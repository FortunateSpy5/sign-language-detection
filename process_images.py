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
        # Some files are missing (or improperly labelled)
        # As a result, only storing the first and last files wouldn't work
        image_dict[folder] = os.listdir(f"./images/{folder}/")

    return image_dict


def get_connections_list(mp_hands):
    # Dictionary to store the collections
    connections_dict = {}
    # mp_hands.HAND_CONNECTIONS is a set, so iterating over it
    for connection in mp_hands.HAND_CONNECTIONS:
        # Extracting the landmarks names
        first, second = connection[0], connection[1]
        # Creating connection name
        connection_name = f"{first.name}_TO_{second.name}"
        # Add connection name and tuple of values to dictionary
        connections_dict[connection_name] = (first.value, second.value)
    return connections_dict


def process_images():
    # mediapipe code
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Run the functions to the get the image directory tree and connection dictionary
    connections_dict = get_connections_list(mp_hands)
    image_dict = get_image_list()

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Iterating over all the folders
        for folder, image_names in image_dict.items():
            print(f"Processing folder: {folder}")
            # Iterating over all the images in folder
            for image_name in image_names:
                # Read image with OpenCV & flip
                image = cv2.imread(f"./images/{folder}/{image_name}")
                image = cv2.flip(image, 1)

                # Process the image
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.multi_hand_landmarks:
                    continue
                
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )
                # Create Image
                cv2.imwrite(f"./processed_images/{folder}/{image_name}", cv2.flip(image, 1))


if __name__ == "__main__":
    process_images()
