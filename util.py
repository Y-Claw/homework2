import cv2

def load_pic(pic_path):
    image = cv2.imread(pic_path)
    return image