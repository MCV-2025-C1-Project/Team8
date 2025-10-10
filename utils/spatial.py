import numpy as np

def split_image_blocks(image: np.ndarray, n: int) -> list[np.ndarray]:

    if len(image.shape) < 2:
        raise ValueError("Image must be at least 2D")
    
    if n <= 0:
        raise ValueError("Number of blocks must be positive")
    
    height, width = image.shape[:2]
    
    if n > height or n > width:
        raise ValueError(f"Number of blocks ({n}) cannot exceed image dimensions ({height}x{width})")
    
    block_height = height // n
    block_width = width // n

    blocks = []
    for i in range(n):
        for j in range(n):
            start_row = i * block_height
            end_row = (i + 1) * block_height if i < n - 1 else height
            start_col = j * block_width
            end_col = (j + 1) * block_width if j < n - 1 else width
            block = image[start_row:end_row, start_col:end_col]
            blocks.append(block)

    return blocks
