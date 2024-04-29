1. FER - 2013(快乐、中性、悲伤、愤怒、惊讶、厌恶、恐惧35,685 个 48x48 像素灰度图像)
https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
具体做法可以参考这个知乎：
https://www.zhihu.com/question/417644001/answer/3445136257

2. AffectNet是一个大型面部数据库，其中包含带有“影响”（面部表情的心理学术语）标记的面部。为了适应常见的内存限制，分辨率降低至 96x96
AffectNet数据集的优点：
分类标注：大约有450,000张图像被手工标注了八种基本情绪（快乐、悲伤、惊讶、恐惧、厌恶、愤怒、轻蔑、中性）。连续标注：部分图像还被标注了情感维度，包括价值（Valence）和激活度（Arousal）。
http://mohammadmahoor.com/affectnet/
https://www.kaggle.com/datasets/noamsegal/affectnet-training-data

3. KDEF 由瑞典卡罗林斯卡研究所创建的一个面部表情库。该数据集包含了从不同角度拍摄的70位模特（男女各半）展示六种基本情绪（快乐、悲伤、愤怒、恐惧、惊讶和厌恶）以及一个中性表情的图像。
KDEF数据集的优点：
多角度拍摄：每个模特的表情都从五个不同的角度（正面、半侧面、侧面、斜上和斜下）进行拍摄，增加了数据的多样性。
高分辨率：所有图片都以高分辨率格式提供，适合进行详细的图像分析。
丰富的表情：包括所有基本情绪的表情，适用于进行复杂的情绪分析研究。
标准化的环境：所有照片都在标准化的光照和背景条件下拍摄，确保数据的一致性。
https://www.kaggle.com/datasets/muhammadnafian/kdef-dataset

4. The Japanese Female Facial Expression (JAFFE) Dataset 日本女性面部表情数据集，包含了213张由10名日本女性演员表演的7种基本面部表情（愤怒、厌恶、恐惧、快乐、悲伤、惊讶和中性）的灰度图像。每种表情都有3个不同的程度。此数据集较小，但具有较高的标注准确率。（我觉得因为日本人比较符合我们亚洲人长相，可能对于我们的实际匹配更精确）暂时还未申请下载数据集，使用时需要下载。https://paperswithcode.com/dataset/jaffe
The Japanese Female Facial Expression (JAFFE) Dataset (zenodo.org)
5. 扩展 Cohn-Kanade （CK+） 数据集包含来自 123 个不同主题的 593 个视频序列，年龄从 18 岁到 50 岁不等，具有不同的性别和传统。每个视频都显示了从中性表情到目标峰值表情的面部转变，以每秒 30 帧 （FPS） 录制，分辨率为 640x490 或 640x480 像素。在这些视频中，有 327 个被标记为七种表情类别之一：愤怒、蔑视、厌恶、恐惧、快乐、悲伤和惊讶。
https://paperswithcode.com/dataset/ck


6.MMI 面部表情数据库包含 2900 多个视频和 75 个主题的高分辨率静态图像。它针对视频中 AU 的存在进行了完全注释（事件编码），并在帧级别上进行了部分编码，指示每个帧的 AU 是否处于中性、起始、顶点或偏移阶段。一小部分被注释为视听笑声。（若后面需要用到视频作为数据集可以用这个，或者将视频截成每一帧作为图片数据集）
https://paperswithcode.com/dataset/mmi

所有数据集下载链接：
https://drive.google.com/drive/folders/1u0zcRj6s9iZBVyrcE3zlzgarrjPZveBi?usp=sharing
