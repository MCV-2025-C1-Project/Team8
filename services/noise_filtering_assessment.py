from dataloader.dataloader import DataLoader
import numpy as np
from preprocessing.preprocessors import PreprocessingMethod
from utils.plots import plot_gaussian_psnr, plot_psnr


class NoiseFilteringAssessment():
    def __init__(self):
        self.dataset = DataLoader()
        self.original_images = []
        self.noisy_images = []
        self.filtered_images = []

        self.preprocessors = None

    def MSE(self, original, assessed):
        """Compute Mean Squared Error between original and assessed images."""
        assert original.shape == assessed.shape, "Original and assessed images must have the same dimensions."
        error = np.sum((original.astype("float") - assessed.astype("float")) ** 2)
        error /= float(original.shape[0] * original.shape[1])
        return error

    def PSNR(self, mse, R=255):
        """Compute Peak Signal-to-Noise Ratio between original and assessed images."""
        if mse == 0:
            return float('inf')
        return 20 * np.log10(R) - 10 * np.log10(mse)
    
    def populate_images(self):
        """Populate original and noisy images from the dataset."""
        for image_id, image, non_augm_image, info, relationship in self.dataset.iterate_images():
            self.original_images.append(non_augm_image)
            self.noisy_images.append(image)

    def run(self, dataset_type, preprocessors: list[PreprocessingMethod]):
        """Run noise filtering assessment."""
        self.dataset.load_dataset(dataset_type)
        self.populate_images()

        for preprocessor in preprocessors:
            print(f"  Evaluating Preprocessing Method: {preprocessor.name}")
            mse_values = []
            psnr_values = []
            for original, noisy in zip(self.original_images, self.noisy_images):
                filtered = preprocessor.apply(noisy)
                mse = self.MSE(original, filtered)
                psnr = self.PSNR(mse)
                mse_values.append(mse)
                psnr_values.append(psnr)
            avg_mse = np.mean(mse_values)
            avg_psnr = np.mean(psnr_values)
            print(f"  Preprocessor: {preprocessor.name}, Average MSE: {avg_mse:.2f}, Average PSNR: {avg_psnr:.2f} dB\n")
    
    def run_single_image(self, dataset_type, image_id, preprocessor: PreprocessingMethod):
        """Run noise filtering assessment on a single image."""
        self.dataset.load_dataset(dataset_type)
        self.populate_images()

        original = self.original_images[image_id-1]
        noisy = self.noisy_images[image_id-1]

        mse_values = []
        psnr_values = []

        if preprocessor == PreprocessingMethod.AVERAGE:
            rng = range(3, 17, 2)
            for k in rng:
                kernel_size = (k, k)
                filtered = preprocessor.apply(noisy, kernel_size=kernel_size)
                mse = self.MSE(original, filtered)
                psnr = self.PSNR(mse)
                mse_values.append(mse)
                psnr_values.append(psnr)
            plot_psnr(
                original_img=original,
                noisy_img=noisy,
                filtered_img=filtered,
                kernel_sizes=list(rng),
                psnr_values=psnr_values,
                title=f"Average filter PSNR assessment wrt. kernel size (Image ID: {image_id})",
            )
        elif preprocessor == PreprocessingMethod.GAUSSIAN:
            # Assess the effect of sigma
            for kernel_size in [(3, 3), (5, 5), (7, 7)]:
                mse_values = []
                psnr_values = []
                rng = np.arange(0.1, 2.1, 0.1)
                for sigma in rng:
                    filtered = preprocessor.apply(noisy, kernel_size=kernel_size, sigma=sigma)
                    mse = self.MSE(original, filtered)
                    psnr = self.PSNR(mse)
                    mse_values.append(mse)
                    psnr_values.append(psnr)
                plot_gaussian_psnr(
                    original_img=original,
                    noisy_img=noisy,
                    filtered_img=filtered,
                    sigma_values=list(rng),
                    psnr_values=psnr_values,
                    title=f"Gaussian filter PSNR assessment wrt. sigma (kernel size={kernel_size}) (Image ID: {image_id})",
                )
        elif preprocessor == PreprocessingMethod.MEDIAN:
            rng = range(3, 17, 2)
            for k in rng:
                kernel_size = (k, k)
                filtered = preprocessor.apply(noisy, kernel_size=kernel_size)
                mse = self.MSE(original, filtered)
                psnr = self.PSNR(mse)
                mse_values.append(mse)
                psnr_values.append(psnr)
            plot_psnr(
                original_img=original,
                noisy_img=noisy,
                filtered_img=filtered,
                kernel_sizes=list(rng),
                psnr_values=psnr_values,
                title=f"Median filter PSNR assessment wrt. kernel size (Image ID: {image_id})",
            )
        