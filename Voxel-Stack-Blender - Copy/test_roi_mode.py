import numpy as np
import cv2
import processing_core as core
from config import Config
import os

def create_test_images():
    """Creates a prior image with two squares and a current image where one has shrunk."""
    prior = np.zeros((200, 400), dtype=np.uint8)
    # Square 1
    cv2.rectangle(prior, (50, 50), (150, 150), 255, -1)
    # Square 2
    cv2.rectangle(prior, (250, 50), (350, 150), 255, -1)
    
    current = np.zeros((200, 400), dtype=np.uint8)
    # Shrunken Square 1
    cv2.rectangle(current, (70, 70), (130, 130), 255, -1)
    # Unchanged Square 2
    cv2.rectangle(current, (250, 50), (350, 150), 255, -1)
    
    return current, prior

def run_test():
    """Runs the test and saves the output images."""
    # Create a directory for test outputs
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    current_mask, prior_mask = create_test_images()
    
    # Save input images for inspection
    cv2.imwrite(os.path.join(output_dir, "test_current.png"), current_mask)
    cv2.imwrite(os.path.join(output_dir, "test_prior.png"), prior_mask)

    # --- Test Fixed Fade Mode ---
    print("Testing Fixed Fade Mode...")
    config_fixed = Config()
    config_fixed.use_fixed_fade_receding = True
    config_fixed.fixed_fade_distance_receding = 20
    
    debug_info_fixed = {'output_folder': output_dir, 'base_filename': 'fixed_fade_output'}
    
    gradient_fixed = core.process_z_blending(current_mask, prior_mask, config_fixed, debug_info=debug_info_fixed)
    cv2.imwrite(os.path.join(output_dir, "test_output_fixed_fade.png"), gradient_fixed)
    print("Fixed Fade Mode test complete.")

    # --- Test ROI Fade Mode ---
    print("Testing ROI Fade Mode...")
    config_roi = Config()
    config_roi.blending_mode = "roi_fade"
    config_roi.use_fixed_fade_receding = True
    config_roi.fixed_fade_distance_receding = 20
    config_roi.roi_params.min_size = 100
    
    debug_info_roi = {'output_folder': output_dir, 'base_filename': 'roi_fade_output'}
    
    gradient_roi = core.process_z_blending(current_mask, prior_mask, config_roi, debug_info=debug_info_roi)
    cv2.imwrite(os.path.join(output_dir, "test_output_roi_fade.png"), gradient_roi)
    print("ROI Fade Mode test complete.")
    
    print(f"Test finished. Check the images in the '{output_dir}' directory.")

if __name__ == "__main__":
    run_test()
