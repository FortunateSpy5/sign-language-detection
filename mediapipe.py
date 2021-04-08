import mediapipe as mp
import cv2
import numpy as np
import os

def get_image_list():
    # Get all the folder names
    image_folders = os.listdir("./images/")

    # Dictionary to store the information
    image_dict = {}

    # Iterating over all the folders
    for folder in image_folders:
        # List of all images
        temp = os.listdir(f"./images/{folder}/")
        # Only extracting the starting and ending no. of all images
        # More effectrive than saving names of all images
        image_dict[folder] = [temp[0][:3], temp[-1][:3]]
    
    return image_dict

def process_images(image_dict):
    print(image_dict)

if __name__ == "__main__":
    image_dict = get_image_list()
    process_images(image_dict)
