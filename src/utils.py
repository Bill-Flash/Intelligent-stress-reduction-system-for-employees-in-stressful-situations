from tensorflow.keras.preprocessing.image import load_img, img_to_array


def get_fer2013_images():
    """
    :return:
    """
    import pandas as pd
    import numpy as np
    import scipy.misc as sm
    import os

    emotions = {
        '0': 'anger',
        '1': 'disgust',
        '2': 'fear',
        '3': 'happy',
        '4': 'sad',
        '5': 'surprised',
        '6': 'neutral',
    }

    def save_image_from_fer2013(file):
        faces_data = pd.read_csv(file)
        root = '../data/fer2013/'

        data_number = 0
        for index in range(len(faces_data)):

            emotion_data = faces_data.loc[index][0]  # emotion
            image_data = faces_data.loc[index][1]  # pixels
            usage_data = faces_data.loc[index][2]  # usage

            image_array = np.array(list(map(float, image_data.split()))).reshape((48, 48))

            folder = root + usage_data
            emotion_name = emotions[str(emotion_data)]
            image_path = os.path.join(folder, emotion_name)
            if not os.path.exists(folder):
                os.mkdir(folder)
            if not os.path.exists(image_path):
                os.mkdir(image_path)

            image_file = os.path.join(image_path, str(index) + '.jpg')
            sm.toimage(image_array).save(image_file)
            data_number += 1
        print('There are' + str(data_number) + 'pictures in total')

    save_image_from_fer2013('../data/fer2013/fer2013.csv')


def get_jaffe_images():
    """
    :return:
    """
    import cv2
    import os
    emotions = {
        'AN': 0,
        'DI': 1,
        'FE': 2,
        'HA': 3,
        'SA': 4,
        'SU': 5,
        'NE': 6
    }
    emotions_reverse = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']

    def detect_face(img):
        """
        :param img:
        :return:
        """
        cascade = cv2.CascadeClassifier('../data/params/haarcascade_frontalface_alt.xml')
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:

            return []
        rects[:, 2:] += rects[:, :2]
        return rects

    folder = '../data/jaffe'
    files = os.listdir(folder)
    images = []
    labels = []
    index = 0
    for file in files:
        img_path = os.path.join(folder, file)
        img_label = emotions[str(img_path.split('.')[-3][:2])]
        labels.append(img_label)
        img = cv2.imread(img_path, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects_ = detect_face(img_gray)
        for x1, y1, x2, y2 in rects_:
            cv2.rectangle(img, (x1+10, y1+20), (x2-10, y2), (0, 255, 255), 2)
            img_roi = img_gray[y1+20: y2, x1+10: x2-10]
            img_roi = cv2.resize(img_roi, (48, 48))
            images.append(img_roi)

        # icons.append(cv2.resize(img_gray, (48, 48)))

        index += 1
    if not os.path.exists('../data/jaffe/Training'):
        os.mkdir('../data/jaffe/Training')
    for i in range(len(images)):
        path_emotion = '../data/jaffe/Training/{}'.format(emotions_reverse[labels[i]])
        if not os.path.exists(path_emotion):
            os.mkdir(path_emotion)
        cv2.imwrite(os.path.join(path_emotion, '{}.jpg'.format(i)), images[i])
    print("load jaffe dataset")


def expression_analysis(distribution_possibility):
    """
    :param distribution_possibility:
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    # 定义8种表情
    emotions = {
        '0': 'anger',
        '1': 'disgust',
        '2': 'fear',
        '3': 'happy',
        '4': 'sad',
        '5': 'surprised',
        '6': 'neutral',
        '7': 'contempt'
    }
    y_position = np.arange(len(emotions))
    plt.figure()
    plt.bar(y_position, distribution_possibility, align='center', alpha=0.5)
    plt.xticks(y_position, list(emotions.values()))
    plt.ylabel('possibility')
    plt.title('predict result')
    if not os.path.exists('../results'):
        os.mkdir('../results')
    plt.show()
    # plt.savefig('../results/rst.png')


def load_test_image(path):
    """
    :param path:
    :return:
    """
    img = load_img(path, target_size=(48, 48), color_mode="grayscale")
    img = img_to_array(img) / 255.
    return img


def index2emotion(index=0, kind='cn'):
    """
    :param index:
    :return:
    """
    emotions = {
        'anger': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happy',
        'sad': 'sad',
        'surprised': 'surprised',
        'neutral': 'neutral',
        'contempt': 'contempt'

    }
    if kind == 'cn':
        return list(emotions.keys())[index]
    else:
        return list(emotions.values())[index]


def cv2_img_add_text(img, text, left, top, text_color=(0, 255, 0), text_size=20):
    """
    :param img:
    :param text:
    :param left:
    :param top:
    :param text_color:
    :param text_size
    :return:
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype(
        "/Users/fangzhihao/Desktop/港大学习/sem2/project/FacialExpressionRecognition-master/assets/simsun.ttc", text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=font_text)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_faces_from_gray_image(img_path):
    """
    :param img_path:
    :return:
    """
    import cv2
    face_cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    faces_gray = []
    for (x, y, w, h) in faces:
        face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]
        face_img_gray = cv2.resize(face_img_gray, (48, 48))
        faces_gray.append(face_img_gray)
    return faces_gray


def get_feature_map(model, layer_index, channels, input_img=None):
    """
    Visualize the feature maps learned by each convolutional layer
    :param model:
    :param layer_index:
    :param channels:
    :param input_img:
    :return:
    """
    if not input_img:
        input_img = load_test_image('../data/demo.jpg')
        input_img.shape = (1, 48, 48, 1)
    from keras import backend as K
    layer = K.function([model.layers[0].input], [model.layers[layer_index+1].output])
    feature_map = layer([input_img])[0]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 8))
    for i in range(channels):
        img = feature_map[:, :, :, i]
        plt.subplot(4, 8, i+1)
        plt.imshow(img[0], cmap='gray')
    plt.show()


if __name__ == '__main__':
    from model import CNN3
    model = CNN3()
    model.load_weights('../models/cnn3_best_weights.h5')
    get_feature_map(model, 1, 32)

