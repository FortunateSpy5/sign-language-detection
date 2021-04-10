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
