# IEMOCAP-Emotion-Detection
### 计53 尹宇峰  计54 马子轩

code文件夹下是所有实验代码，data文件夹下是所有数据，包括语音和对应的文字，以及标注。

### 代码说明

`features.py`, `helper.py`, `mocap_data_collect.py`是用于提取声音特征和文本特征的文件。`speech.py`是利用声音特征训练的神经网络模型，准确率是52%。`text.py`是利用文本特征训练的神经网络模型，准确率是60%。

在命令行中先运行`python mocap_data_collect.py`提取特征，然后分别运行`python speech.py`和`python text.py`来训练神经网络。

### 数据说明

因为文件太大了，无法上传至网络学堂，所以助教如需访问，请从网盘下载，链接: https://pan.baidu.com/s/1PItiiyJnTAzPa7IYCZ6e5Q 密码: cu55，并将数据放置于data文件夹下面。一共有5个Session，每个Session中，`dialog/wav`保存的是音频数据，`dialog/transcriptions`保存的是文本数据，`dialog/EmoEvaluation`保存的是标注数据。
