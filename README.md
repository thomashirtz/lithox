# 🔬 lithox

High performance jax-based photolithography simulation.

## Installation 

```
pip install git+https://github.com/thomashirtz/lithox#egg=lithox
```

## Utilization 

```python
import lithox as ltx

mask_path = r"./data/mask.png"
mask = ltx.load_image(mask_path, size=1024)
simulator = ltx.LithographySimulator()
simulation_result = simulator(mask)
```

<p align="center"> <img src="./scripts/simulation.png" alt="scripts/simulation.png" width="500"/> </p> <p align="center"><em>Example output from the script <code>./scripts/simulation.png</code></em></p>

## Citation

If you use lithox in your work—whether for research, publications, or projects—please cite it as follows:

```
@misc{hirtz2025lithox,
  author       = {Thomas Hirtz},
  title        = {lithox: A JAX-based lithography simulation library},
  year         = {2025},
  howpublished = {\url{https://github.com/thomashirtz/lithox}},
  publisher    = {GitHub},
}
```
