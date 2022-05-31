from unittest import result
import cv2
import time
from matplotlib.transforms import BboxBase
import numpy as np
from iwpod_net.src.keras_utils import load_model, detect_lp_width
from iwpod_net.src.utils import im2single
import os.path as osp
import sys
from CenterTrack_ROOT.src._init_paths import add_path

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)


from crop_detector import Detector
from opts import opts



class blacker():
    def __init__(self, read_path, write_path):
        self.vid = cv2.VideoCapture(read_path)
        #for i in range(100):
        #    self.vid.read()
        size = cv2.VideoCapture(read_path).read()[1].shape[:2][::-1]
        print(size)
        self.writer = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.vid.get(cv2.CAP_PROP_FPS), size)
        self.iwpod_net = load_model('iwpod_net/weights/iwpod_net')
        ct_opt = opts().init()
        self.ct = Detector(ct_opt)
        self.count = 1
    
    # Black out the license plates in the input image COMPLETE
    def black(self, crop, pts):
        parsed = []
        if pts == []:
            return
        for i in range(4):
            parsed.append([pts[0][i], pts[1][i]])
        parsed = np.array(parsed).reshape((1, 4, 2)).astype(np.int32)
        cv2.fillPoly(crop, pts=parsed, color=(0, 0, 0))

    def locate_vehicles(self, frame):
        if self.count % 7 == 1:
            ret, f_time = self.ct.run(frame, 0)
        else:
            ret, f_time = self.ct.run(frame, 0)
        bboxes = []
        for item in self.ct.tracker.tracks:
            bboxes.append(item['bbox'])
            cv2.rectangle(frame, ((int)(item['bbox'][0]), (int)(item['bbox'][1])), ((int)(item['bbox'][2]), (int)(item['bbox'][3])), (0, 255, 0), 2)
        self.count += 1
        return bboxes

    def crop_vehicle(self, frame, bboxes):
        crops = []
        for box in bboxes:
            crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            crops.append(crop)
        return crops

    # Find license plates in crops of the input image COMPLETE
    def locate_LP(self, crops):
        ptss = []
        lp_output_resolution = tuple((240, 80))
        for crop in crops:
            print(crop.shape)
            iwh = np.array(crop.shape[1::-1],dtype=float).reshape((2,1))
            ASPECTRATIO = max(1, min(2.75, 1.0*crop.shape[1]/crop.shape[0]))  # width over height
            WPODResolution = 256
            Llp, LlpImgs, _ = detect_lp_width(self.iwpod_net, im2single(crop), WPODResolution*ASPECTRATIO, 2**1, lp_output_resolution, 0.01)
            for i, img in enumerate(LlpImgs):
                pts = Llp[i].pts * iwh
                self.black(crop, pts)
        return ptss

    def step(self):
        ret, frame = self.vid.read()
        if not ret:
            return False
        #cv2.imshow('frame', frame)
        #cv2.waitKey(0)
        bboxes_vehicle = self.locate_vehicles(frame)
        crops = self.crop_vehicle(frame, bboxes_vehicle)
        self.locate_LP(crops)
        self.writer.write(frame)
        return True

    def run(self):
        while self.step():
            print(self.count)
            pass
        self.writer.release()
        self.vid.release()
        

if __name__ == '__main__':
    black = blacker('./images/demo.mp4', './images/night_black.mp4')
    bef = time.time()
    black.run()
    aft = time.time()
    print('Time: ', aft - bef)
    im = cv2.imread('./images/15.jpg')
    print(im.shape)
    black.black([im], black.locate_LP([im]))
    cv2.imshow('im', im)
    cv2.waitKey(0)
#bef = time.time()
#for i in range(10):
#    obj.step()
#after = time.time()
#print(after-bef)
#obj.writer.release()
