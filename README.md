# Stress expression recognition
## Introduction
The entire system was constructed using a combination of a convolutional neural network and a linear regression model. The deep model produced exceptional results after testing traditional facial feature extraction methods like Gabor and LBP. The face emotion recognition model was assessed using three expression recognition datasets: FER2013, JAFFE, and CK+. Additionally, the stress detection model was evaluated using our own curated dataset. Furthermore, linear regression and random forest methods were utilized for the evaluation of stress detection.

## Environment deployment
Based on Python3 and Keras2 (TensorFlow backend), the specific dependency installation is as follows (it is recommended to use the conda virtual environment).
```shell script=
cd FacialExpressionRecognition
conda create -n FER python=3.6 -y
conda activate FER
conda install cudatoolkit=10.1 -y
conda install cudnn=7.6.5 -y
pip install -r requirements.txt
```

## Project Description
### **Traditional Method**
- Data Preprocessing
	- Image Denoising
	- Face Detection (HAAR Classifier Detection (opencv))
- Feature Engineering
	- Face Feature Extraction
		- LBP
		- Gabor
- Classifier
	- SVM
### **Deep Method**
- Face Detection
	- HAAR Classifier
	- MTCNN (better effect)
- Convolutional Neural Network
	- For Feature Extraction + Classification

## Model application
When compared to traditional methods, convolutional neural networks outperform them. This model is utilized to develop a recognition system, offering a **GUI interface and real-time camera detection** (ensuring sufficient fill light for cameras). During prediction, a picture is subjected to horizontal flipping, a 15-degree deflection, and translation to obtain multiple probability distributions. These distributions are then weighted and summed to derive the final probability distribution. Presently, the label with the highest probability is selected (using data enhancement for inference).

### **GUI interface**

Note that **GUI interface prediction only shows the facial expression that is most likely to be a face, but all detected faces will be framed and marked on the picture. The marked pictures are in the output directory.**

Execute the following command to launch the GUI program, which is designed using PyQT. The resulting effect on a sample picture (sourced from the Internet) is displayed below.

```shell
python src/gui.py
```

### **Real-time detection**
Real-time detection is developed with Opencv to predict real-time video streams from cameras. If you don't have a camera and would like to test with a video, you can modify the command line parameters.

To run real-time detection using the camera, use the following command (exit using the ESC key). If you wish to specify a video for detection, use the second command below.
```shell
python src/recognition_camera.py
```

```shell
python src/recognition_pressure_camera.py
```
