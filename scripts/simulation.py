import warnings

import matplotlib.pyplot as plt

import lithox as ltx
from lithox.paths import DATA_DIRECTORY

warnings.filterwarnings(
    "ignore",
    message="A JAX array is being set as static!.*"
)

if __name__ == '__main__':

    # Configuration
    mask_path = DATA_DIRECTORY / 'mask.png'
    image_size = 1024

    # Load and preprocess the mask image
    mask = ltx.load_image(mask_path, image_size)

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