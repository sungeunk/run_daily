import argparse
import os

import cv2
import numpy as np


def blank_image_check(img, threshold):
    # Reading Image
    image = cv2.imread(img, 0)
    
    # Normalize pixel values to range [0,1]
    image_normalized = image.astype(float) / 255.0
    
    # Calculate standard deviation of normalized image
    std_dev = np.std(image_normalized)
    
    # Threshold for normalized values
    if std_dev < threshold:
        return True
    else:
        return False

def blank_image_validator(directory, threshold=0.05):
    validation_results = []
    
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add more extensions if needed
            img_path = os.path.join(directory, filename)
            result = blank_image_check(img_path, threshold)
            
            print(f"Image: {filename} -> is blank: {result}")
            
            validation_results.append(result)
            
    for res in validation_results:
        if res == True:
            raise Exception("Generated images are blank")
                

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Check for blank/empty images')
    parser.add_argument('--threshold', type=float, default=0.05,
                      help='Threshold for standard deviation (default: 0.05). Lower values mean stricter blank detection')
    parser.add_argument('--directory', type=str, default="tests\\utils\\blank_images_samples",
                      help='directory containing images')
    args = parser.parse_args()

    # Directory path
    dir_path = args.directory
    
    # Check all images in directory
    for filename in os.listdir(dir_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add more extensions if needed
            img_path = os.path.join(dir_path, filename)
            result = blank_image_check(img_path, args.threshold)
            
            print(f"Image: {filename} -> is blank: {result}")

if __name__ == "__main__":
    main()