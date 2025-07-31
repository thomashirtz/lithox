import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import lithox as ltx
from lithox.paths import PACKAGE_DIRECTORY

if __name__ == '__main__':

    # Configuration
    mask_path = PACKAGE_DIRECTORY.parent / "data" / 'mask.png'
    image_size = 1024

    # Load and preprocess the mask image
    raw_mask = (
        Image.open(mask_path)
        .convert("L")
        .resize((image_size, image_size), Image.NEAREST)
    )
    mask = jnp.array(raw_mask, dtype=np.float32) / 255.0

    # Instantiate process-variation simulator
    simulator = ltx.ProcessVariationSimulator()

    # Run simulation
    print("Running process-variation simulation...")
    var_out = simulator(mask)
    print("Simulation completed.")

    # Extract printed min and max
    prt_min = var_out.printed.min
    prt_max = var_out.printed.max

    # Compute 3-class map: 0 = never prints, 1 = max-only, 2 = always prints
    class_map = (prt_max + prt_min).astype(jnp.int32)

    # Plot mask and PVB classes side by side
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4))

    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("Mask")
    axes[0].axis("off")

    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(["black", "red", "gray"])
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N)

    im = axes[1].imshow(class_map, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Process Variation")
    axes[1].axis("off")

    # Add colorbar with class labels
    cbar = fig.colorbar(im, ax=axes[1], ticks=[0.5, 1.5, 2.5], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["Never prints", "Variation band", "Always prints"])
    # cbar.set_label("Classes", rotation=90, labelpad=6)

    plt.tight_layout()
    plt.show()

    # Print fraction summary
    total = class_map.size
    nev = jnp.sum(class_map == 0) / total
    maxonly = jnp.sum(class_map == 1) / total
    always = jnp.sum(class_map == 2) / total
    print(f"Fraction never prints: {nev:.4f}")
    print(f"Fraction variation band: {maxonly:.4f}")
    print(f"Fraction always prints: {always:.4f}")