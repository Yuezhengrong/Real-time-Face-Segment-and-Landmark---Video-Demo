<h2 align="center">Real-time Face Segment and Landmark - Video Demo</h2>

<div align="center"><i>Real-time Face Segment and Landmark - Video Demo</i></div>

<br />

<img src="new.gif" width="100%">
<div align="center">Real-time Face Segment and Landmark is a Combination method for <b>real-time</b> Person Segment and Landmark Detection with <b>only RGB video input</b></div>
<div align="center">Real-time Face Segment and Landmark 是一个<b>仅需RGB视频输入</b>的<b>实时</b>人像分割和关键点检测的组合方法</div>

<br />

## Real-time Face Segment and Landmark - Video Demo
We utilize MODNet to complete person segmentation, and use Mediapipe to complete character facial landmark detection.

### 1. Requirements
The basic requirements for this demo are:
- Anaconda
- Python 3+
- requiremen.txt

**NOTE**: If your device does not satisfy the above conditions, please try our [online Colab demo](https://colab.research.google.com/drive/1Pt3KDSc2q7WxFvekCnCLD8P0gBEbxm6J?usp=sharing).


### 2. Run Demo
We recommend creating a new conda virtual environment to run this demo, as follow:

1. Clone the this repository

2. Create a conda virtual environment named `seg_mark` (if it doesn't exist) and activate it. Here we use `python=3.9` as an example:
     ```
    conda create -n seg_mark python=3.9
    source activate seg_mark
    ```

3. Install the required python dependencies (please make sure your CUDA version is supported by the PyTorch version installed):
    ```
    pip install -r demo/video_matting/webcam/requirements.txt
    ```
4. Execute the main code:
    ```
    python run.py
    ```

### 3. Acknowledgement
We thank [@google](https://google.github.io/mediapipe/), [@ZHKKKe el at.](https://github.com/ZHKKKe/MODNet) for their contributions to this code.
