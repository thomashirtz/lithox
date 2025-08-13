import warnings
from pathlib import Path

import matplotlib.pyplot as plt

import lithox as ltx

warnings.filterwarnings(
    "ignore",
    message="A JAX array is being set as static!.*"
)

if __name__ == '__main__':

    # Configuration
    mask_path = Path('../data/mask.png')
    target_path = Path('../data/target.png')
    image_size = 1024

    # Load and preprocess the mask image
    mask = ltx.load_image(mask_path, image_size)
    target = ltx.load_image(target_path, image_size)

    # Run the lithography simulation
    print("Running lithography simulation...")
    simulator = ltx.LithographySimulator()
    simulation_result = simulator(mask)
    print("Simulation completed.")

    # Map titles to their corresponding image data
    title_to_data = {
        "Target": target,
        "Original mask": mask,
        "Aerial image": simulation_result.aerial,
        # "Resist Image": simulation_result.resist,
        "Printed image": simulation_result.printed,
    }

    # Plot the results in a 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5), dpi=200)
    for ax, (title, data) in zip(axes.flatten(), title_to_data.items()):
        plot = ax.imshow(data, cmap="magma")
        ax.set_title(title)
        fig.colorbar(
            plot,
            ax=ax,
            orientation="vertical",
            fraction=0.046,
            pad=0.04,
        )
        ax.set_axis_off()

    plt.tight_layout()
    # plt.savefig('simulation.png', dpi=200)
    plt.show()
