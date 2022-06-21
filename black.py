import cv2
import time
import numpy as np
import os.path as osp
from CenterTrack_ROOT.src._init_paths import add_path
from torchvision.ops import nms 
import torch
import torchvision
import imp
MainModel = imp.load_source('MainModel', "load_model.py")

this_dir = osp.dirname(__file__)


from crop_detector import Detector
from opts import opts


class blacker():
    def __init__(self, read_path, write_path):
        self.vid = cv2.VideoCapture(read_path)
        size = cv2.VideoCapture(read_path).read()[1].shape[:2][::-1]
        print(size)
        ct_opt = opts().init()
        self.ct = Detector(ct_opt)
        self.crop_size = np.array([300, 300])
        self.writer = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.vid.get(cv2.CAP_PROP_FPS), size)
        self.model = torch.load('plate.pth').to('cuda')
        self.count = 0
    
    # Black out the license plates in the input image COMPLETE
    def black(self, crop, pts):
        cv2.fillPoly(crop, [pts], (0, 0, 0))


    def decode(self, result, threshold=0.1):
        result = result.to('cpu').numpy()
        net_stride = 2**4
        side = 7.75

        Probs = result[..., 0]
        Affines = result[...,-6:]

        try:
            xx, yy = np.where(Probs>threshold)
            print('hey')
            print(xx)
            if(len(xx) == 0):
                print(Probs)
                print(max(Probs))
                return False
        except:
            return False

        WH = self.crop_size
        MN = WH/net_stride

        base = np.matrix([[-.5,-.5,1.],[.5,-.5,1.],[.5,.5,1.],[-.5,.5,1.]]).T
        conf = np.zeros((len(xx)))
        ptss = np.zeros((len(xx), 4, 2))

        for i in range(len(xx)):
            y, x = xx[i], yy[i]
            affine = Affines[y, x]
            prob = Probs[y, x]

            mn = np.array([float(x) + .5, float(y) + .5])

            A = np.reshape(affine, (2, 3))
            A[0, 0] = max(0., A[0, 0])
            A[1, 1] = max(0., A[1, 1])
            pts = np.array(A*base)
            pts_MN_center_mn = pts * side
            pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
            
            pts_prop = pts_MN/MN.reshape((2, 1))

            pts_prop = pts_prop.T.reshape(4, 2)

            conf[i] = prob
            ptss[i] = pts_prop

        return ptss[np.argmax(conf)]
        

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
        scores = torch.Tensor(scores)
        bboxes = torch.Tensor(bboxes)
        idx = nms(bboxes, scores, 0.5)
        bboxes = bboxes[idx]
        self.count += 1
        print(len(bboxes))
        return bboxes

    def crop_vehicle(self, frame, bboxes):
        crops = []
        for box in bboxes:
            crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            print(crop.shape)
            if 0 in crop.shape:
                continue
            crops.append(crop)
        return crops

    def plate_detection(self, crops):
        crops_resized = torch.zeros((len(crops), 3, self.crop_size[0], self.crop_size[1]), device='cuda')
        for i in range(len(crops)):
            print(i)
            h, w = crops[i].shape[:2]
            resize_dim = []
            pad_dim = []
            if h > w:
                resize_dim = (int(self.crop_size[1] * w / h), self.crop_size[0])
                pad_dim =  (0, self.crop_size[1] - resize_dim[0] , 0, 0)
            else:
                resize_dim = (self.crop_size[1], int(self.crop_size[0] * h / w))
                pad_dim = (0, 0, 0, self.crop_size[0] - resize_dim[1])

            resized = torch.from_numpy(cv2.resize(crops[i], resize_dim)).permute(2, 0, 1).to('cuda')
            resized = torch.nn.functional.pad(resized, pad_dim).type(torch.float32) / 255
            crops_resized[i] = resized

        with torch.no_grad():
            results = self.model(crops_resized)

        results = results.permute(0, 2, 3, 1)
        for i in range(len(crops)):
            pts = self.decode(results[i], 0.01)
            if isinstance(pts, bool):
                continue
            long_side = max(crops[i].shape[:2])
            pts = (pts * long_side).astype(np.int32)
            self.black(crops[i], pts)

    def step(self):
        torch.cuda.synchronize()
        time_one = time.time()
        ret, frame = self.vid.read()
        torch.cuda.synchronize()
        time_two = time.time()
        print('read took time:', time_two - time_one)
        if not ret:
            return False
        bboxes_vehicle = self.locate_vehicles(frame)
        crops = self.crop_vehicle(frame, bboxes_vehicle)
        torch.cuda.synchronize()
        time_three = time.time()
        print('ct took time:', time_three - time_two)
        self.plate_detection(crops)
        torch.cuda.synchronize()
        time_four = time.time()
        print('LP took time:', time_four - time_three)
        self.writer.write(frame)
        torch.cuda.synchronize()
        time_five = time.time()
        print('write took time:', time_five - time_four)
        return True

    def run(self):
        while self.step() and self.count < 5:
            print(self.count)
            pass
        self.writer.release()
        self.vid.release()
        

if __name__ == '__main__':
    black = blacker('./images/original.mkv', './images/edited.mkv')
    #im = np.array([cv2.imread('./images/hey.jpg')])
    #print(im.shape)
    #black.plate_detection(im)
    black.run()

    #input = torch.from_numpy(crops.astype(np.float32)).permute(0, 3, 1, 2).to('cuda')
    #print(res.permute(0, 2, 3, 1))
    
