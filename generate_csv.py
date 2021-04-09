import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
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

def get_connections_list_2(mp_hands):
    # Adding some connections manually to increase accuracy
    # All landmark names and values: https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
    connections_dict = get_connections_list(mp_hands)
    connections_dict.update({
        "WRIST_TO_THUMB_TIP": (0, 4),
        "WRIST_TO_INDEX_FINGER_TIP": (0, 8),
        "WRIST_TO_MIDDLE_FINGER_TIP": (0, 12),
        "WRIST_TO_RING_FINGER_TIP": (0, 16),
        "WRIST_TO_PINKY_TIP": (0, 20),
        "THUMB_TIP_TO_INDEX_FINGER_TIP": (4, 8),
        "INDEX_FINGER_TIP_TO_MIDDLE_FINGER_TIP": (8, 12),
        "MIDDLE_FINGER_TIP_TO_RING_FINGER_TIP": (12, 16),
        "RING_FINGER_TIP_TO_PINKY_TIP": (16, 20),
    })
    return connections_dict


def get_distance(first, second):
    # Calculate distance from two coordinates
    return np.sqrt(
        (first.x - second.x) ** 2 
        + (first.y - second.y) ** 2 
        # + (first.z - second.z) ** 2
    )

def create_connections_csv():
    # mediapipe code
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Run the functions to the get the image directory tree and connection dictionary
    connections_dict = get_connections_list_2(mp_hands)
    image_dict = get_image_list()

    # List to store all the data to be put in the dataframe
    data = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Iterating over all the folders
        for folder, image_names in image_dict.items():
            print(f"Processing folder: {folder}")
            # Iterating over all the images in folder
            for image_name in image_names:
                # Read image with OpenCV, flip, convert BGR to RGB
                image = cv2.imread(f"./images/{folder}/{image_name}")
                image = cv2.flip(image, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Get coordinates from mediapipe
                results = hands.process(image)
                if not results.multi_hand_landmarks:
                    continue
                coordinates = results.multi_hand_landmarks[0].landmark

                # Calculate distances and append the row to data
                row = []
                for _, values in connections_dict.items():
                    row.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
                row.append(folder)
                data.append(row)
    
    # Create dataframe
    columns = list(connections_dict.keys())
    columns.append('SIGN')
    df = pd.DataFrame(data=data, columns=columns)
    # Export to CSV file
    df.to_csv('connections.csv')

if __name__ == "__main__":
    create_connections_csv()
