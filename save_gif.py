import os
import cv2
import imageio
def create_gif(img_path="save_frames", gif_name='new.gif', duration = 0.08):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 每帧图像间隔时间
    :return:
    '''
    frames = []
    for image_name in os.listdir(img_path):
        temp = os.path.join(img_path, image_name)
        print(temp)
        img = cv2.imread(temp)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


if __name__ == '__main__':
    create_gif()