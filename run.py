from PIL import Image
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
import torchvision.transforms as transforms
from modnet import MODNet
from save_gif import create_gif
i = 0
image_list=[]
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation
# 调用关键点检测模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,  # 检测静态图片设置为False,检测视频设置为True，默认为False
                                  max_num_faces=3,  # 能检测的最大人脸数，默认为1
                                  refine_landmarks=True,  # 定位嘴唇、眼睛、瞳孔的关键点，设置为True，否则设置为False
                                  # 当模型配置refine_landmarks=True,获得478个人脸关键点；
                                  # 当模型配置refine_landmarks=False,获得468个人脸关键点,缺少瞳孔的关键点。
                                  min_detection_confidence=0.5,  # 人脸检测的置信度
                                  min_tracking_confidence=0.5)  # 人脸追踪的置信度（检测图像时可以忽略）

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print('Load MODNet...')
pretrained_ckpt = 'modnet_webcam_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
if GPU:
    print('Use GPU...')
    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
    print('Use CPU...')
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

modnet.eval()

print('Init Cam...')
cap = cv2.VideoCapture('o4.mp4')  # 0内置摄像头，1外接摄像头，'o4.mp4'文件
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# 开始人像segment
print('Starting ...')
while (True):
    _, frame = cap.read()
    # 原图frame_np
    frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
    frame_np = frame_np[:, 120:792, :]
    frame_np = cv2.flip(frame_np, 1)

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :]
    if GPU:
        frame_tensor = frame_tensor.cuda()

    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)  # 模型输出结果

    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)  # 转化为(B,C,W,H)
    # matte_np就是三通道mask，人像不变，背景为黑色，转化为Opencv image的shape(W,H,C)=(512, 672, 3),>0.1的位置为True，小于的为False
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0) > 0.1

    # 生成背景图bg_image
    BG_COLOR = (166, 219, 255)  # BGR 粉(213, 174, 246)、蓝(255, 219, 166)
    bg_image = np.zeros(frame_np.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    # 满足条件的地方等于frame_np，不满足的地方等于bg_image,(frame_np和bg_image都是RGB)
    output_image = np.where(matte_np, frame_np, bg_image)

    # 转化成opencv的BGR格式
    output_image = cv2.cvtColor(np.uint8(output_image), cv2.COLOR_RGB2BGR)

    # 在segment基础上加landmark
    # 使用FaceMesh模型的process方法检测图像中的关键点
    results = face_mesh.process(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    # 获得关键点结果
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 利用mp_drawing绘制图像

            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                      # 选取关键点
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      # 绘制关键点，若为None，表示不绘制关键点，也可以指定点的颜色、粗细、半径
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(236, 228, 233), thickness=1,
                                                                                   circle_radius=2),
                                      # 绘制连接线的格式
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_IRISES,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            # FACEMESH_CONTOURS可以选取轮廓点,将其连接成线
            mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    # 如果没有检测到关键点，在黑色背景上显示“NO FACE TO DETECT”
    else:
        output_image = np.zeros(output_image.shape, dtype='uint8')
        output_image = cv2.putText(output_image, str("NO FACE TO DETECT"), (300, 400),
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)

    # save gif
    save_gif = False
    if(save_gif):
        i = i+1  # 控制变量，达到20帧，生成gif
        frame_RGB = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # 因为opencv使用的是BGR,而imageio使用RGB
        image_list.append(frame_RGB)
        if i <= 20:
            cv2.imwrite(f'save_frames/{i}.jpg', frame_RGB)


    cv2.imshow('SegMark - WebCam [Press \'Q\' To Exit]', output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Exit...')
