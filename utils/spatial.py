import numpy as np

def split_image_blocks(image: np.ndarray, n: int) -> list[np.ndarray]:
    height, width = image.shape[:2]
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

if __name__ == "__main__":
    import cv2
    img = cv2.imread("data/BBDD/bbdd_00000.jpg")
    blocks = split_image_blocks(img, 4)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(cv2.cvtColor(blocks[i * 4 + j], cv2.COLOR_BGR2RGB))
            axs[i, j].axis('off')
    plt.show()
