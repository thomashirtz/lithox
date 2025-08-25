# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import torch


def convert_pt_to_npy(source_path: str, output_path: str, permute: bool = False) -> None:
    """
    Load a .pt tensor file, permute dimensions from [H, W, C] to [C, H, W],
    and save it as a .npy file, creating the output directory if needed.

    Parameters:
    - source_path: Path to the .pt file to load.
    - output_path: Path where the .npy file will be saved.
    """

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load tensor
    tensor = torch.load(source_path)

    # Permute dimensions and convert to NumPy
    if permute:
        tensor = tensor.permute(2, 0, 1)
    array = tensor.cpu().numpy()

    # Save as .npy
    np.save(output_path, array)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    source_dir = Path('/kernels_pt')
    dest_dir = Path('/lithox/kernels')
    for pt_file in source_dir.glob('*.pt'):
        output_file = dest_dir / f"{pt_file.stem}.npy"
        convert_pt_to_npy(str(pt_file), str(output_file), permute=True)

    source_dir = Path('/scales_pt')
    dest_dir = Path('/lithox/scales')
    for pt_file in source_dir.glob('*.pt'):
        output_file = dest_dir / f"{pt_file.stem}.npy"
        convert_pt_to_npy(str(pt_file), str(output_file), permute=False)
