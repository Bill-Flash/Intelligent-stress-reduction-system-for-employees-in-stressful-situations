
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
from model import CNN2, CNN3
from utils import index2emotion, cv2_img_add_text
from blazeface import blaze_detect
# from Regression.stress_test import load_model2, predict
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, default=0, help="data source, 0 for camera 1 for video")
parser.add_argument("--video_path", type=str, default=None)
opt = parser.parse_args()

if opt.source == 1 and opt.video_path is not None:
    filename = opt.video_path
else:
    filename = None


# Identify model
class StressPredictor(nn.Module):
    def __init__(self, input_dim):
        super(StressPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # output 1 value for regression

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model2(model_path, input_dim):
    model = StressPredictor(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for tensors, labels in dataloader:
            outputs = model(tensors).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def load_model():
    """
    加载本地模型
    :return:
    """
    model = CNN3()
    model.load_weights('/Users/fangzhihao/Desktop/港大学习/sem2/project/FacialExpressionRecognition-master/models/cnn3_best_weights.h5')
    return model

def pressure_predict(data, model="lg"):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # get 32 same data in a batch
    data = np.tile(data, (32, 1))
    label = [3] * 32
    if model == "lg": 
        model_path = '/Users/fangzhihao/Desktop/港大学习/sem2/project/FacialExpressionRecognition-master/stress_predictor_model.pth'
        input_dim = 8
        model = load_model2(model_path, input_dim)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        dataset = TensorDataset(data_tensor, label_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        preds, labels = predict(model, dataloader)
    if model == "rf":
        model_path = '/Users/fangzhihao/Desktop/港大学习/sem2/project/FacialExpressionRecognition-master/src/Regression/rf_model.pkl'
        model = joblib.load(model_path)
        preds = model.predict(data)
    return preds[0] + 1


def generate_faces(face_img, img_size=48):
    """
    :param face_img: 
    :param img_size: 
    :return:
    """

    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img)
    resized_images.append(face_img[2:45, :])
    resized_images.append(face_img[1:47, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_expression():
    """
    :return:
    """
    model = load_model()

    border_color = (0, 0, 0)  
    font_color = (255, 255, 255)  
    capture = cv2.VideoCapture(0)  
    if filename:
        capture = cv2.VideoCapture(filename)

    while True:
        _, frame = capture.read()  
        frame = cv2.cvtColor(cv2.resize(frame, (800, 600)), cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        # cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')  
        faces = blaze_detect(frame)

        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame_gray[y: y + h, x: x + w] 
                faces = generate_faces(face)
                results = model.predict(faces)
                result_sum = np.sum(results, axis=0).reshape(-1)
                level = pressure_predict(result_sum, model="lg")
                level_str = "Stress Level: " + str(level)
                label_index = np.argmax(result_sum, axis=0)
                emotion = index2emotion(label_index)
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
                frame = cv2_img_add_text(frame, level_str, x+30, y+30, font_color, 20)
  
                # cv2.putText(frame, emotion, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 4)
        cv2.imshow("expression recognition(press esc to exit)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(30)  
        if key == 27:
            break
    capture.release()  
    cv2.destroyAllWindows()  


if __name__ == '__main__':
    predict_expression()