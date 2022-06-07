import numpy as np
import cv2
import time

one = np.ones((255, 255, 3))
pts = np.array([[[1, 1], [10, 10], [120, 120], [200, 200]]])
bef = time.time()
for i in range(1000000):
    cv2.fillPoly(one, pts, color=(0, 0, 0))
aft = time.time()
print('time taken is:', aft - bef)

