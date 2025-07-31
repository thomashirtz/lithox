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

    # Run the lithography simulation
    print("Running lithography simulation...")
    simulator = ltx.LithographySimulator()
    simulation_result = simulator(mask)
    print("Simulation completed.")

    # Map titles to their corresponding image data
    title_to_data = {
        "Original Mask": mask,
        "Aerial Image": simulation_result.aerial,
        "Resist Image": simulation_result.resist,
        "Printed Image": simulation_result.printed,
    }

    # Plot the results in a 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (title, data) in zip(axes.flatten(), title_to_data.items()):
        plot = ax.imshow(data, cmap="gray")
        ax.set_title(title)
        fig.colorbar(
            plot,
            ax=ax,
            orientation="vertical",
            fraction=0.046,
            pad=0.04,
        )

    plt.tight_layout()
    plt.show()