# import numpy as np
import cv2
import torch
import numpy as np
import sys
# sys.path.append('/deep_sort')
import os
# os.chdir('/content/deep_sort')
# from deep.feature_extractor import Extractor
from deep.feature_extractor import FastReIDExtractor
from sort.nn_matching import NearestNeighborDistanceMetric
from sort.preprocessing import non_max_suppression
from sort.detection import Detection
from sort.tracker import Tracker

from hand_knn.hand_detect import hand_pose_recognition, init_knn

dirname = os.path.dirname(__file__)

#load yolo5 weight
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False).to(device)

fast_reidconfig = ''
class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.5, max_age=200, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        # self.extractor = Extractor(model_path, use_cuda=use_cuda)
        self.extractor = FastReIDExtractor(os.path.join(dirname, 'deep/fastreid/config/bagtricks_R50.yml'),os.path.join(dirname, 'deep/checkpoint/market_bot_R50.pth'), use_cuda=use_cuda)
        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    def update(self, bbox_tlwh, bbox_tlbr, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_tlbr, ori_img)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._xywh_to_tlbr(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int16))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    def _xywh_to_tlwh(self, bbox_xywh): # 把中心点宽高转换成左上点宽高
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh

    def _xywh_to_tlbr(self, bbox_xywh): # 把中心点宽高转换成左上右下点
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_tlbr(self, bbox_tlwh): # 把左上点宽高转换成左上右下点
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _tlbr_to_tlwh(self, bbox_tlbr): # 把左上右下点转换成左上点宽高
        x1,y1,x2,y2 = bbox_tlbr
        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h

    def draw_boxes(self, img, bbox_tlbr, identities=None, offset=(0,0)):
        for i,box in enumerate(bbox_tlbr):
            x1,y1,x2,y2 = box
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0    
            color = self.compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
            cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
            cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        return img

    def compute_color_for_labels(self, label):
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)
        
    def _get_features(self, bbox_tlbr, ori_img):
        im_crops = []
        for x1,y1,x2,y2 in bbox_tlbr:
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


model_path = os.path.join(dirname, 'deep/checkpoint/ckpt.t7')
max_dist = 0.2
min_confidence = 0.3
nms_max_overlap = 0.5
max_iou_distance = 0.7
max_age = 70
n_init = 3
nn_budget = 100
use_cuda = torch.cuda.is_available()

deepsort = DeepSort(model_path, max_dist, min_confidence, 
                    nms_max_overlap, max_iou_distance, 
                    max_age, n_init, nn_budget, use_cuda)

spider_label = -1

if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    neigh, embedder = init_knn(os.path.join(dirname, 'hand_knn/dataset_embedded.npz'))
    time = 0
    while True:
        ret, frame = vid.read()
        #TODO: convert frame to PILLOW!!
        outputs_OD = model([frame]) 
        
        mask = outputs_OD.xywh[0][:,-1] == 0
        tlwh = outputs_OD.xywh[0][:,:4][mask].to('cpu').to(torch.int16)
        tlbr = outputs_OD.xyxy[0][:,:4][mask].to('cpu').to(torch.int16)
        conf = outputs_OD.xywh[0][:,4][mask].to('cpu')                               
        img  = outputs_OD.imgs[0]
        outputs_MOT = deepsort.update(tlwh,tlbr,conf, img)
        print(time,spider_label)
        if len(outputs_MOT) > 0:
            bbox_tlbr = outputs_MOT[:, :4]
            identities = outputs_MOT[:, -1]
            if spider_label== -1:
                
                # iterate human's object box
                for output in outputs_MOT:
                    top,left,bottom,right  = output[:4] #tlbr,id

                    # print("left is : ", left)
                    # print("right is : ", right)
                    # print("bottom is : ", bottom)
                    # print("top is : ", top)
                    # if left>right:
                    #     person_image = im[right:left,bottom:top, :]
                    # else:
                    #     person_image = im[left:right,bottom:top, :]
                    person_image = img[top:bottom,left:right, :]
                    if person_image.shape[0]==0:
                        continue
                    hand_class, person_image = hand_pose_recognition(person_image, neigh, embedder)

                    if len(hand_class) > 1:
                        if (hand_class[0] == 'spider' or hand_class[1] == 'spider'):
                            spider_label = output[-1]
                            print("spider_label is : ", spider_label)
                    elif hand_class == 'spider': 
                        spider_label = output[-1]
                        print("spider_label is : ", spider_label)
            
            t_xywh = None   #TODO: ask TA how to handle None
            bbox_xyxy = outputs_MOT[:, :4]
            identities = outputs_MOT[:, -1]
            if spider_label > -1:
                mask = outputs_MOT[:,-1]==spider_label
                identities[mask] = identities[mask]+10000
                if len(outputs_MOT[:,:4][mask])>0:
                    time = 0
                    tt, tl, tb, tr = outputs_MOT[:,:4][mask][0]
                    tx = int((tt+tb)/2)
                    ty = int((tl+tr)/2)
                    tw = int(tr-tl)
                    th = int(tb-tt)
                    t_xywh = [tx,ty,tw,th]
                    # img = cv2.circle(img, (tx,ty), radius=0, color=(0, 0, 255), thickness=-1)
                    # print('spider bounding box: ',np.array([tx,ty,tw,th]))
                    #return t_xywh,1
                else:
                    time += 1
                    if time > 100:
                        spider_label = -1
                #     return t_xywh,0
            # else:
                
                #return t_xywh,0
            
        
            img = deepsort.draw_boxes(img, bbox_xyxy, identities)

        
        # img = deepsort.draw_boxes(img, bbox_tlbr, identities)
        

        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()