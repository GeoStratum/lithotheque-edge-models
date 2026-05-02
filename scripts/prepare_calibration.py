import os
import glob
import numpy as np
from PIL import Image
import argparse

def prepare_calibration_dataset(image_dir, output_path, size=(256, 256), num_samples=100):
    """
    Prepares a representative calibration dataset for INT8/W4A8 quantization.
    Extracts num_samples from the directory, resizes them, and saves as a .npy tensor.
    """
    all_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_files.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    
    if len(all_files) == 0:
        print(f"Error: No images found in {image_dir}")
        return

    # Shuffle and sample
    np.random.seed(42)
    np.random.shuffle(all_files)
    selected = all_files[:num_samples]
    
    print(f"Processing {len(selected)} samples...")
    images = []
    for f in selected:
        try:
            img = Image.open(f).convert('RGB')
            img = img.resize(size, Image.Resampling.LANCZOS)
            img_data = np.array(img).astype(np.float32) / 255.0
            
            # Save as NCHW (standard for most calibration readers)
            img_data = np.transpose(img_data, (2, 0, 1)) 
            images.append(img_data)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if images:
        np.save(output_path, np.stack(images))
        print(f"Successfully saved {len(images)} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lithotheque Calibration Preparator")
    parser.add_argument("--dir", type=str, required=True, help="Path to raw images")
    parser.add_argument("--out", type=str, required=True, help="Output .npy path")
    parser.add_argument("--size", type=int, default=256, help="Resize dimension (square)")
    args = parser.parse_args()
    
    prepare_calibration_dataset(args.dir, args.out, size=(args.size, args.size))
