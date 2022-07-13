from polygon import do_something as piou
import os
import torch
import imp
import numpy as np
import cv2
from shapely.geometry import box, Polygon

MainModel = imp.load_source('MainModel', "../load_model.py")

this_dir = os.path.dirname(__file__)

class Chair():
    def __init__(self, model_pth, gt_pth, threshold=0.5, success_criteria=[0.5, 0.1]):
        self.model_pth = model_pth
        self.gt_pth = gt_pth
        self.threshold = 0.5
        self.success_criteria = success_criteria
    
    def compile(self, ):
        self.model = torch.load(self.model_pth)
        self.gt = self.get_gt()
        self.preds = self.get_preds()
        self.size = len(self.gt)
    
    def decode(self, result, threshold=0.1):
        result = result.to('cpu').numpy()
        net_stride = 2**4
        side = 7.75

        Probs = result[..., 0]
        Affines = result[...,-6:]
        print(max(Probs.reshape(-1)))
        try:
            xx, yy = np.where(Probs>threshold)
            if(len(xx) == 0):
                return False
        except:
            return False

        WH = self.crop_size[::-1]
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

        return max(conf), ptss[np.argmax(conf)]

    def get_preds(self, ):
        img_pth = 'dummy'
        imgs = os.listdir(img_pth)
        preds = []
        for img in imgs:
            res = self.model(torch.from_numpy(cv2.imread(os.path.join(img_pth, img))).permute(2, 0, 1).to('cuda'))
            conf, pts = self.decode(res)
            result = {'conf': conf, 'pts': pts}
            preds.append(result)
        return preds

    def get_gt(self, ):
        txts = os.listdir(self.gt_pth)
        gt = []
        for txt in txts:
            if 'txt' not in txt:
                continue
            with open(os.path.join(self.gt_pth, txt)) as f:
                contents = f.read()
                list = contents.split(',')
                coord = np.array(list[1:9], dtype=np.float)
                coord = np.reshape(coord, (4, 2), order = 'F')
                gt.append(coord)
        return gt
    
    def get_iou(self, pts1, pts2):
        poly1 = Polygon(pts1)
        poly2 = Polygon(pts2)
        i = poly1.intersection(poly2)
        u = poly1.union(poly2)
        return i/u

    

    def get_metrics(self, ):
        miss_count = 0
        tot_iou = 0
        successes = 0
        for i in range(self.size):
            poly1 = Polygon(self.preds[i]['pts'])
            poly2 = Polygon(self.gt[i])
            i = poly1.intersection(poly2)
            u = poly1.union(poly2)
            iou = i.area/u.area
            tot_iou += iou
            if self.preds[i]['conf'] < self.threshold:
                miss_count += 1
            else:
                lp_iou = i.area / poly2.area
                frame = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
                pred_iou = poly1.area / frame.area
                if lp_iou > self.success_criteria[0] and pred_iou < self.success_criteria[1]:
                    successes += 1
            if iou == 0:
                miss_count += 1
        return miss_count/self.size, tot_iou/self.size, successes / self.size


        




if __name__ == '__main__':

    #model = torch.load('./dummy.pth').to('cuda')
    dum = Chair('', './dataset')
    img = np.zeros((200, 200, 3))
    pts = torch.from_numpy(dum.get_gt()[90]).to(float)
    ptss = torch.from_numpy(dum.get_gt()[89]).to(float)
    polygon1_shape = Polygon(pts)
    print(pts)
    polygon2_shape = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.union(polygon2_shape).area
    print(polygon1_shape.area)
    print(intersection)
    print(polygon_union)
    IOU = intersection / polygon_union
    print(IOU)

