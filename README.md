# Sign Language Detection
**Sign languages**  are languages that use the visual-manual modality to convey meaning. Sign languages are expressed through manual articulations in combination with non-manual elements. Sign languages are full-fledged natural languages with their own grammar and lexicon. Sign languages are not universal and they are not mutually intelligible with each other, although there are also striking similarities among sign languages. Wherever communities of deaf people exist, sign languages have developed as useful means of communication, and they form the core of local Deaf cultures. Although signing is used primarily by the deaf and hard of hearing, it is also used by hearing individuals, such as those unable to physically speak, those who have trouble with spoken language due to a disability or condition, or those with deaf family members, such as children of deaf adults. ([Source](https://en.wikipedia.org/wiki/Sign_language))

This started as a project to detect the various alphabets of the **[Indian Sign Language](https://en.wikipedia.org/wiki/Indo-Pakistani_Sign_Language)** using **[Mediapipe](https://mediapipe.dev)** and **[Tensoflow](https://www.tensorflow.org/)**, with the dataset being taken from [here](https://drive.google.com/drive/folders/1wgXtF6QHKBuXRx3qxuf-o6aOmN87t8G-). However, now I realize that there is nothing in this project (other than a few variable names) that is specific to detecting sign language. This project can be used for wide variety of purposes. 

Basically the way this project works is that all the training images are placed in their respective folders in **images** subdirectory. The file **generate_csv.py** creates a csv (comma separated values) file using the images. Each row is an image, each column stores the length of a particular connection (distance between two joints in hand) and each corresponding class for classification is the name of the folder the image is located in. Then this csv file is used to create an ANN (Artificial Neural Network) model, which can be done in the **ann_model.py** file. After that, this model is used in **real_time_prediction.py** file to predict the class corresponding to the shape of the hand as visible in the webcam. (There is also a process_images.py and cnn_model.py for a CNN implementation, but my main focus was on the ANN model as using CNN on top of Mediapipe felt redundant.)

Let's say that you want to use this project to play a certain game. Suppose that game has 10 possible inputs. Take multiple images, as many as you can, of 10 different hand shapes that you want to use to control the game. Put them in 10 different folders inside the **images** subdirectory. Then execute the files **generate_csv.py** and **ann_model.py**. Then you can modify the **real_time_prediction.py** in such a way that the predictions will be used as the input for the game, then you can play the game with your hand with an almost 100% accuracy as long as you used enough images and any two hand shapes are not too similar to each other.

### Required Libraries
* Mediapipe
* OpenCV
* Numpy
* Pandas
* Scikit-Learn
* Tensorflow

### Dataset
* **Original Source:** **[Github](https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition)**, **[Google Drive](https://drive.google.com/drive/folders/1wgXtF6QHKBuXRx3qxuf-o6aOmN87t8G-)**
* **My Files:** **[Google Drive](https://drive.google.com/drive/folders/1JsiLPimYHM8BGoNrrmJr0tunsTBtg9Al)** (Some images that couldn't be detected properly have been removed. There is also a **processed_images** folder. The processed images are used for CNN and it doesn't necessarily need to be downloaded as they can be generated using the **process_images.py** file)

### Procedure
* Place all the training images in their respective folders in the **images** subcategory. If you are using the folder downloaded from my Google Drive, then you can directly paste the folder in the root directory.
* Run the **generate_csv.py** file. It iterates through all the folders and reads the images one by one. The Mediapipe framework processes the image and determines the 3D coordinates of the landmarks (joints in the hand). The coordinates are used to calculate the distances between specific joints and those distances along with the corresponding classification (name of the folder the image is in) is stored in the **connections.csv** file. (Alternatively, you can use the **connections.csv** file provided in this repository.)
* Run the **ann_model.py** file. It uses the data in the **connections.csv** file to create an Artificial Neural Network (ANN) model that is saved in **ann_model.h5** file. (Alternatively, you can use the **ann_model.h5** file provided in this repository.)
* Finally, run the **real_time_prediction.py**. It use OpenCV to get images from the webcam (Try changing 0 in the line *cap = cv2.VideoCapture(0)* if it doesn't work properly). The image is processed by Mediapipe and distances are calculated as it was done in the previous file. The data is given as input to the ANN model loaded from the **ann_model.h5** file. The original image superimposed with the landmarks and the deafult connection is displayed in a separate window along with the final prediction using OpenCV. This final prediction can be used for some other purpose according to your need.

*For the Convolutional Neural Network (CNN) implementation, first run the process_images.py or download the processed_images folder from my Google Drive link. Then run the cnn_model.py to create the model. Currently, there is no code in this repository that uses the cnn_model.h5 for any purpose.*

### Mediapipe
<p align="center">
  <img src="https://google.github.io/mediapipe/images/mobile/hand_crops.png" alt="Mediapipe" />
</p>

MediaPipe Hands is a high-fidelity hand and finger tracking solution. It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame. The ability to perceive the shape and motion of hands can be a vital component in improving the user experience across a variety of technological domains and platforms. For example, it can form the basis for sign language understanding and hand gesture control, and can also enable the overlay of digital content and information on top of the physical world in augmented reality. While coming naturally to people, robust real-time hand perception is a decidedly challenging computer vision task, as hands often occlude themselves or each other.

MediaPipe Hands utilizes an ML pipeline consisting of multiple models working together: A palm detection model that operates on the full image and returns an oriented hand bounding box. A hand landmark model that operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand keypoints. If you want to learn more about Mediapipe Hands, you can visit [this page](https://google.github.io/mediapipe/solutions/hands).
