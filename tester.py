import cv2


im = cv2.imread('./images/13.jpg')
size = im.shape[:2][::-1]
author = cv2.VideoWriter('./images/demo.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, size)
for i in range(10):
    author.write(im)
