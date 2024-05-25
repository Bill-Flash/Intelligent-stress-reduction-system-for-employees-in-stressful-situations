import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
# import dlib


def add_noise(input_data):
    """
    Increase noise interference
    :param input_data:
    :return:
    """
    for i in range(5000):
        x = np.random.randint(0, input_data.shape[0])
        y = np.random.randint(0, input_data.shape[1])
        input_data[x][y][:] = 255
    return input_data


def histogram_equalization(img):
    """
    :param img:
    :return:
    """
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    cv.equalizeHist(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
    return img


def adaptive_histogram_equalization(img):
    """
    :param img:
    :return:
    """
    ycrcb = img
    channels = cv.split(ycrcb)
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels[0] = clahe.apply(channels[0])
    cv.merge(channels, ycrcb)
    img = ycrcb
    return img


def detection(img):
    """
    :param img:
    :return:
    """
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)  # Use detector for face detection, dets is the returned result
    print("Number of faces detected: {}".format(len(dets)))  # Print the number of recognized faces
    for index, face in enumerate(dets):
        # Label faces in pictures and display
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)
    return dets


def predictor(img, dets):
    """
    :param img:
    :param dets:
    :return:
    """
    # shape_predictor_68_face_landmarks.dat is a model for face calibration, it is based on HOG features.
    predictor_path = "../model/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    shape_list = []
    for index, face in enumerate(dets):
        shape = predictor(img, face)
        for _, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            cv.circle(img, pt_pos, 1, (0, 255, 0), 1)
        shape_list.append(shape)
    return shape_list


def gray_norm(img):
    """
    灰度归一化
    :param img:
    :return:
    """
    min_value = np.min(img)
    max_value = np.max(img)
    if max_value == min_value:
        return img
    (n, m) = img.shape
    for i in range(n):
        for j in range(m):
            img[i, j] = int(255 * (img[i][j] - min_value) / (max_value - min_value))
    return img


def normailiztaion(img, dets, shape_list):
    """
    :param img:
    :param dets:
    :param shape_list:
    :return:
    """
    img = gray_norm(img)

    img_list = []
    pt_pos_list = []
    for index, face in enumerate(dets):
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        img1 = img[top:bottom, left:right]
        size = (48, 48)
        img1 = cv.resize(img1, size, interpolation=cv.INTER_LINEAR)

        pos = []
        for _, pt in enumerate(shape_list[index].parts()):
            pt_pos = (int((pt.x - left) / (right - left) * 90), int((pt.y - top) / (bottom - top) * 100))
            pos.append(pt_pos)
            cv.circle(img1, pt_pos, 2, (255, 0, 0), 1)
        pt_pos_list.append(pos)
        img_list.append(img1)
    return img_list, pt_pos_list


def deal(img):
    """
    :param img:
    :return:
    img: The image after being cropped
    dets: Face bounding information
    shape: The positions of feature points in the original image
    img_list: The cropped images
    pt_pos_list: The positions of feature points in each image
    """
    img = cv.blur(img, (5, 5))
    dets = detection(img)
    shape_list = predictor(img, dets)
    adaptive_histogram_equalization(img)
    img_list, pt_pos_list = normailiztaion(img, dets, shape_list)
    return img, dets, shape_list, img_list, pt_pos_list


def test():
    lena = cv.cvtColor(cv.imread('zhangyu.jpg', flags=1), cv.COLOR_BGR2RGB)
    print(lena.shape)

    lena.flags.writeable = True

    plt.suptitle('preprocess')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot(321)
    plt.imshow(lena)
    plt.title('origin_image')
    plt.axis('off')

    noise_image = add_noise(lena)
    plt.subplot(322)
    plt.imshow(noise_image)
    plt.title('noise_image')
    plt.axis('off')

    blur_image = cv.blur(noise_image, (5, 5))
    plt.subplot(323)
    plt.imshow(blur_image)
    plt.title('AF_image')
    plt.axis('off')

    median_blur_image = cv.medianBlur(noise_image, 5)
    plt.subplot(324)
    plt.imshow(median_blur_image)
    plt.title('MF_image')
    plt.axis('off')

    # adaptive_median_blur_image = cv.ad

    plt.subplot(325)
    plt.imshow(histogram_equalization(median_blur_image))
    plt.title('equalization_image')
    plt.axis('off')

    plt.subplot(326)
    plt.imshow(adaptive_histogram_equalization(median_blur_image))
    plt.title('adaptive_equalization_image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    test()
