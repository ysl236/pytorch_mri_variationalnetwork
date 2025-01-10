import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob
import os

def calculate_metrics(original_path, reconstructed_path):
    """
    Calculate PSNR and SSIM between two images
    """
    # Read images
    original = cv2.imread(original_path)
    reconstructed = cv2.imread(reconstructed_path)
    
    # Convert to same size if necessary
    if original.shape != reconstructed.shape:
        reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale for SSIM calculation
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    reconstructed_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
    
    # Calculate PSNR
    psnr_value = psnr(original, reconstructed)
    
    # Calculate SSIM
    ssim_value = ssim(original_gray, reconstructed_gray)
    
    return psnr_value, ssim_value

def assess_images(original_dir, reconstructed_dir, extension="png"):
    """
    Calculate average PSNR and SSIM for all images in directories
    """
    # Get all image files
    original_files = sorted(glob.glob(os.path.join(original_dir, f"*.{extension}")))
    reconstructed_files = sorted(glob.glob(os.path.join(reconstructed_dir, f"*.{extension}")))
    
    if len(original_files) != len(reconstructed_files):
        raise ValueError("Number of original and reconstructed images must match")
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    for orig_path, recon_path in zip(original_files, reconstructed_files):
        try:
            psnr_value, ssim_value = calculate_metrics(orig_path, recon_path)
            total_psnr += psnr_value
            total_ssim += ssim_value
            count += 1
            print(f"Image {count}: PSNR = {psnr_value:.2f}, SSIM = {ssim_value:.4f}")
        except Exception as e:
            print(f"Error processing {orig_path}: {str(e)}")
    
    if count > 0:
        mean_psnr = total_psnr / count
        mean_ssim = total_ssim / count
        print("\nAverage Metrics:")
        print(f"Mean PSNR: {mean_psnr:.2f}")
        print(f"Mean SSIM: {mean_ssim:.4f}")
        return mean_psnr, mean_ssim
    return 0, 0

if __name__ == "__main__":
    # Example usage
    original_directory = "eval_result_img_coronal_pd_fs/reference"
    reconstructed_directory = "eval_result_img_coronal_pd_fs/recon"
    
    assess_images(original_directory, reconstructed_directory)
