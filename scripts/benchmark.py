import os
import time
from pathlib import Path

# Set before JAX is imported (including transitively via lithox).
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu,cuda"

import jax
import jax.numpy as jnp

import lithox as ltx

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "This script requires matplotlib. Install with: pip install lithox[dev]"
    ) from exc
import lithox.defaults as d
from lithox.simulation import compute_aerial_from_mask


def _devices_for(platform: str) -> list[jax.Device]:
    try:
        return list(jax.devices(platform))
    except Exception as exc:
        print(f"Note: jax.devices({platform!r}) failed: {type(exc).__name__}: {exc}")
        return []


def _discover_devices() -> tuple[list[jax.Device], list[jax.Device]]:
    cpu = _devices_for("cpu")
    gpu = _devices_for("gpu")

    if not cpu and not gpu:
        try:
            all_devs = list(jax.devices())
            cpu = [d for d in all_devs if d.platform == "cpu"]
            gpu = [d for d in all_devs if d.platform == "gpu"]
        except Exception as exc:
            print(f"Note: jax.devices() failed: {type(exc).__name__}: {exc}")

    return cpu, gpu


def _time_call(fn, *, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = fn()
        jax.block_until_ready(y)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def main() -> None:
    try:
        import jaxlib  # type: ignore
        jaxlib_version = getattr(jaxlib, "__version__", "unknown")
    except Exception:
        jaxlib_version = "unknown"

    print(f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '(unset)')}")
    cpu, gpu = _discover_devices()

    try:
        default_backend = jax.default_backend()
    except RuntimeError:
        default_backend = "unknown"

    print(f"JAX default_backend={default_backend}  jaxlib={jaxlib_version}")
    if cpu:
        print("CPU:", ", ".join(d.device_kind for d in cpu))
    if gpu:
        print("GPU:", ", ".join(d.device_kind for d in gpu))

    if not gpu:
        print(
            "\nGPU not available (no visible GPU, driver issue, or CUDA plugin cannot init).\n"
            "Check: nvidia-smi\n"
            "CPU-only:  JAX_PLATFORMS=cpu python scripts/benchmark.py\n"
            "CPU+GPU:   JAX_PLATFORMS=cpu,cuda python scripts/benchmark.py  (default in this script)\n"
        )
    if not cpu and gpu:
        print("\nNo CPU device exposed (GPU-only JAX). CPU benchmark skipped.")
    if not cpu and not gpu:
        raise RuntimeError(
            "No JAX devices found. Try:\n"
            "  JAX_PLATFORMS=cpu python scripts/benchmark.py\n"
            "  nvidia-smi  # confirm the GPU is visible to the OS"
        )

    key = jax.random.key(0)

    sizes = [64, 128, 256, 512, 1024]
    batch = 4
    sim = ltx.LithographySimulator()

    kernels = sim.kernels
    kernels_ct = sim.kernels_ct
    scales = sim.scales
    dose = sim.dose
    resist_threshold = sim.resist_threshold
    resist_steepness = sim.resist_steepness
    tau_b = jnp.asarray(d.BINARIZATION_THRESHOLD, jnp.float32)

    def forward(mask_bhw: jax.Array) -> jax.Array:
        aerial = compute_aerial_from_mask(
            mask=mask_bhw.astype(jnp.float32),
            dose=dose,
            kernels_fourier=kernels,
            kernels_fourier_ct=kernels_ct,
            scales=scales,
        )
        resist = jax.nn.sigmoid(resist_steepness * (aerial - resist_threshold))
        return (resist > tau_b).astype(mask_bhw.dtype)

    forward_jit = jax.jit(forward)

    def bench_on_device(dev: jax.Device) -> tuple[list[float], list[float]]:
        compile_times_s: list[float] = []
        step_times_s: list[float] = []

        for n in sizes:
            nonlocal key
            key, sub = jax.random.split(key)
            mask = jax.random.uniform(
                sub, shape=(batch, n, n), dtype=jnp.float32, minval=0.0, maxval=1.0
            )
            mask = jax.device_put(mask, dev)

            def run():
                with jax.default_device(dev):
                    return forward_jit(mask)

            t0 = time.perf_counter()
            y0 = run()
            jax.block_until_ready(y0)
            compile_times_s.append(time.perf_counter() - t0)

            step_times_s.append(_time_call(run, repeats=10))

            print(
                f"[{dev.platform}] n={n:4d}  compile+first={compile_times_s[-1]:.4f}s  "
                f"step={step_times_s[-1]:.4f}s  device_kind={dev.device_kind}"
            )

        return compile_times_s, step_times_s

    cpu_compile, cpu_step = (None, None)
    gpu_compile, gpu_step = (None, None)
    if cpu:
        cpu_compile, cpu_step = bench_on_device(cpu[0])
    if gpu:
        gpu_compile, gpu_step = bench_on_device(gpu[0])

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0), dpi=160)
    if cpu_compile is not None and cpu_step is not None:
        ax.plot(sizes, cpu_compile, marker="o", label="cpu: compile + first run")
        ax.plot(sizes, cpu_step, marker="o", label="cpu: steady-state step")
    if gpu_compile is not None and gpu_step is not None:
        ax.plot(sizes, gpu_compile, marker="o", label="gpu: compile + first run")
        ax.plot(sizes, gpu_step, marker="o", label="gpu: steady-state step")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Image size (N×N)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"lithox benchmark (batch={batch}, float32)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    out_path = Path(__file__).resolve().parent / "benchmark.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
