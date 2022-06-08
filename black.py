from unittest import result
import cv2
import time
from matplotlib.transforms import BboxBase
import numpy as np
from iwpod_net.src.keras_utils import *
from iwpod_net.src.utils import *
import os.path as osp
import sys
from CenterTrack_ROOT.src._init_paths import add_path
import tensorflow as tf
from torchvision.ops import nms 
import torch

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)


from crop_detector import Detector
from opts import opts

#def find_LP():
    
    # rewrite detect_LP_width

    # resize

    # put through prediction 

    # reconsturct batch

    # convert back to original coordinates

    # 

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
        scores = []
        for item in self.ct.tracker.tracks:
            scores.append(item['score'])
            bboxes.append(item['bbox'])
            # cv2.rectangle(frame, ((int)(item['bbox'][0]), (int)(item['bbox'][1])), ((int)(item['bbox'][2]), (int)(item['bbox'][3])), (0, 255, 0), 2)
        print(len(scores))
        scores = torch.Tensor(scores)
        bboxes = torch.Tensor(bboxes)
        idx = nms(bboxes, scores, 0.5)
        print(len(idx))
        bboxes = bboxes[idx]
        self.count += 1
        return bboxes

    def crop_vehicle(self, frame, bboxes):
        crops = []
        for box in bboxes:
            crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            if 0 in crop.shape:
                continue
            crops.append(crop)
        return crops

    # Find license plates in crops of the input image COMPLETE
    def locate_LP(self, crops):
        ptss = []
        lp_output_resolution = tuple((240, 80))
        #crops_resized = np.array((len(crops), 56, 64, 3))
        for crop in crops:
            #crops_resized[i] = cv2.resize(crops[i],(64, 56), interpolation = cv2.INTER_CUBIC)

            
            
            iwh = np.array(crop.shape[1::-1],dtype=float).reshape((2,1))
            ASPECTRATIO = max(1, min(2.75, 1.0*crop.shape[1]/crop.shape[0]))  # width over height
            WPODResolution = 256
            Llp, LlpImgs, _ = detect_lp_width(self.iwpod_net, im2single(crop), WPODResolution*ASPECTRATIO, 2**1, lp_output_resolution, 0.01)
            for i, img in enumerate(LlpImgs):
                pts = Llp[i].pts * iwh
                self.black(crop, pts)
            
        return ptss

    def locate_LP_alt(self, crops):
        ptss = []
        lp_output_resolution = tuple((240, 80))
        crops_resized = np.zeros((len(crops), 96, 96, 3))
        iwhs = np.zeros((len(crops), 2, 1))
        for i in range(len(crops)):
            crops_resized[i] = im2single(tf.image.resize_with_pad(crops[i], 96, 96).numpy().astype(np.uint8))
            #cv2.imshow('hu', tf.image.resize_with_pad(crops[i], 56, 64).numpy().astype(np.uint8))
            #cv2.waitKey(0)
            s = max(crops[i].shape[1::-1])
            iwhs[i] = np.array([s, s],dtype=float).reshape((2,1))
        results = self.iwpod_net.predict(crops_resized)
        
        for i in range(len(crops)):
            label, TLps = reconstruct_new(crops[i], crops_resized[i], results[i], lp_output_resolution, 0.01)
            label, TLps = FindBestLP(label, TLps)
            for j, img in enumerate(TLps):
                pts = label[j].pts * iwhs[i]
                w, h = crops[i].shape[1::-1]
                if h < w:
                    pts[1][:] -= 0.5 * (w-h)
                else:
                    pts[0][:] -= 0.5 * (h-w)
                self.black(crops[i], pts)
        return ptss

    def step(self):
        time_one = time.time()
        ret, frame = self.vid.read()
        time_two = time.time()
        print('read took time:', time_two - time_one)
        if not ret:
            return False
        #cv2.imshow('frame', frame)
        #cv2.waitKey(0)
        bboxes_vehicle = self.locate_vehicles(frame)
        crops = self.crop_vehicle(frame, bboxes_vehicle)
        time_three = time.time()
        print('ct took time:', time_three - time_two)
        self.locate_LP_alt(crops)
        time_four = time.time()
        print('LP took time:', time_four - time_three)
        self.writer.write(frame)
        time_five = time.time()
        print('write took time:', time_five - time_four)
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
#bef = time.time()
#for i in range(10):
#    obj.step()
#after = time.time()
#print(after-bef)
#obj.writer.release()
