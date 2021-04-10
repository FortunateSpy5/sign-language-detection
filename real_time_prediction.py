import cv2
import mediapipe as mp
from generate_csv import get_connections_list, get_distance
from tensorflow import keras
import numpy as np
import pandas as pd

def get_sign_list():
    # Function to get all the values in SIGN column
    df = pd.read_csv('connections.csv', index_col=0)
    return df['SIGN'].unique()

def real_time_prediction():
    sign_list = get_sign_list()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    connections_dict = get_connections_list()

    # Initialize webcam
    # Default is zero, try changing value if it doesn't work
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            # Get image from webcam, change color channels and flip
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            # Get result
            results = hands.process(image)
            if not results.multi_hand_landmarks:
                # If no hand detected, then just display the webcam frame
                cv2.imshow(
                    'Sign Language Detection',
                    frame
                )
            else:
                # If hand detected, superimpose landmarks and default connections
                mp_drawing.draw_landmarks(
                    image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

                # Get landmark coordinates & calculate length of connections
                coordinates = results.multi_hand_landmarks[0].landmark
                data = []
                for _, values in connections_dict.items():
                    data.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
                
                # Scale data
                data = np.array([data])
                data[0] /= data[0].max()
                
                # Load model from h5 file
                model = keras.models.load_model('ann_model.h5')

                # Get prediction
                pred = np.array(model(data))
                pred = sign_list[pred.argmax()]

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                # Display text showing prediction
                image = cv2.putText(
                    image, pred, (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, 
                    (255, 0, 0), 2
                )

                # Display final image
                cv2.imshow(
                    'Sign Language Detection',
                    image
                )

            # Each frame will be displayed for 20ms (50 fps)
            # Press Q on keyboard to quit
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_prediction()
