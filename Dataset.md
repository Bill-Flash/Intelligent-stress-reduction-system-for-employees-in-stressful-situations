1. FER-2013 (35685 48x48 pixel grayscale images of happiness, neutrality, sadness, anger, surprise, disgust, and fear)
https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
The specific method can refer to this Zhihu:
https://www.zhihu.com/question/417644001/answer/3445136257

2. AffectNet is a large facial database that contains faces marked with "influence" (a psychological term for facial expressions). To adapt to common memory limitations, the resolution has been reduced to 96x96
Advantages of AffectNet dataset:
Classification annotation: Approximately 450000 images were manually annotated with eight basic emotions (happiness, sadness, surprise, fear, disgust, anger, contempt, neutral). Continuous annotation: Some images are also annotated with emotional dimensions, including Valence and Arousal.
http://mohammadmahoor.com/affectnet/
https://www.kaggle.com/datasets/noamsegal/affectnet-training-data

3. KDEF is a facial expression library created by the Karolinska Institute in Sweden. This dataset contains images of 70 models (half male and half female) captured from different angles displaying six basic emotions (happiness, sadness, anger, fear, surprise, and disgust) and a neutral expression.
Advantages of the KDEF dataset:
Multi-angle shooting: Each model's expression is shot from five different angles (front, half side, side, diagonal up and diagonal down), increasing the diversity of data.
High resolution: All images are provided in a high-resolution format, suitable for detailed image analysis.
Rich expressions: including expressions of all basic emotions, suitable for conducting complex emotional analysis research.
Standardized environment: All photos are taken under standardized lighting and background conditions to ensure data consistency.
https://www.kaggle.com/datasets/muhammadnafian/kdef-dataset

5. The Japanese Female Facial Expression (JAFFE) Dataset contains 213 grayscale images of 7 basic facial expressions (anger, disgust, fear, happiness, sadness, surprise, and neutrality) performed by 10 Japanese female actors. Each expression has three different degrees. This dataset is small but has high annotation accuracy. (I think because Japanese people are more in line with our Asian appearance, it may be more accurate for our actual matching) We have not yet applied to download the dataset, so we need to download it when using it. https://paperswithcode.com/dataset/jaffe
The Japanese Female Facial Expression (JAFFE) Dataset (zenodo. org)
6. Expand the Cohn Kanade (CK+) dataset to include 593 video sequences from 123 different themes, ranging in age from 18 to 50 years old, with different genders and traditions. Each video displays a facial transition from a neutral expression to a target peak expression, recorded at 30 frames per second (FPS) with a resolution of 640x490 or 640x480 pixels. In these videos, 327 were marked as one of the seven categories of facial expressions: anger, contempt, disgust, fear, happiness, sadness, and surprise.
https://paperswithcode.com/dataset/ck

6. The MMI facial expression database contains over 2900 videos and high-resolution static images of 75 themes. It fully annotates the presence of AU in the video (event encoding) and partially encodes it at the frame level, indicating whether the AU in each frame is in the neutral, starting, vertex, or offset stage. A small portion is annotated as audio-visual laughter. (If you need to use video as the dataset later, you can use this, or cut the video into each frame as the image dataset.)
https://paperswithcode.com/dataset/mmi

All dataset download links:
https://drive.google.com/drive/folders/1u0zcRj6s9iZBVyrcE3zlzgarrjPZveBi?usp=sharing
