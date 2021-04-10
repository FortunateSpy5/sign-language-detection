import cv2
import mediapipe as mp
from generate_csv import get_connections_list_2, get_distance
from tensorflow import keras
import numpy as np
import pandas as pd

def get_df_values():
    df = pd.read_csv('connections.csv', index_col=0)
    return df['SIGN'].unique(), list(df.columns).index('WRIST_TO_INDEX_FINGER_MCP')

def real_time_prediction():
    sign_list, scaling_column_index = get_df_values()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    connections_dict = get_connections_list_2(mp_hands)

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            results = hands.process(image)
            if not results.multi_hand_landmarks:
                cv2.imshow(
                    'Sign Language Detection',
                    frame
                )
            else:
                mp_drawing.draw_landmarks(
                    image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

                coordinates = results.multi_hand_landmarks[0].landmark

                data = []
                for _, values in connections_dict.items():
                    data.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
                
                data = np.array([data])
                
                model = keras.models.load_model('ann_model.h5')
                pred = np.array(model(data))
                pred = sign_list[pred.argmax()]

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image = cv2.putText(
                    image, pred, (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, 
                    (255, 0, 0), 2
                )

                cv2.imshow(
                    'Sign Language Detection',
                    image
                )

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_prediction()
