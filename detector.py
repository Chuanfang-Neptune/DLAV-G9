import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import torch.nn.functional as F
from deepsort import *
from hand_knn.hand_detect import hand_pose_recognition, init_knn

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_c):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.box = torch.nn.Linear(n_hidden, n_output-1)   # output layer
        self.logit = torch.nn.Linear(n_hidden, 1)
        
        self.conv1 = torch.nn.Sequential(         # input shape (3, 80, 60)
            torch.nn.Conv2d(
                in_channels = n_c,            # input height
                out_channels = 8,             # n_filters
                kernel_size = 5,              # filter size
                stride = 2,                   # filter movement/step
                padding = 0,                  
            ),                              
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(kernel_size = 2),    
        )
        self.conv2 = torch.nn.Sequential(       
            torch.nn.Conv2d(in_channels = 8, 
                            out_channels = 16, 
                            kernel_size = 5, 
                            stride = 2, 
                            padding = 0),      
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(2),                
        )
        
        self.conv3 = torch.nn.Sequential(       
            torch.nn.Conv2d(in_channels = 16, 
                            out_channels = 8, 
                            kernel_size = 1, 
                            stride = 1, 
                            padding = 0),      
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(2),                
        )
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = feat.view(feat.size(0), -1)
        x2 = F.relu(self.hidden(feat))      # activation function for hidden layer
        
        out_box = F.relu(self.box(x2))            # linear output
        out_logit = torch.sigmoid(self.logit(x2))
        
        return out_box, out_logit
        
class Detector(object):
    """docstring for Detector"""
    def __init__(self):
        super(Detector, self).__init__()
        # TODO: MEAN & STD
        # self.mean = [[[[0.5548078,  0.56693329, 0.53457436]]]] 
        # self.std = [[[[0.26367019, 0.26617227, 0.25692861]]]]
        # self.img_size = 100 
        # self.img_size_w = 80
        # self.img_size_h = 60
        # self.min_object_size = 10
        # self.max_object_size = 40 
        # self.num_objects = 1
        # self.num_channels = 3
        # self.model = Net(n_feature = 1632, n_hidden = 128, n_output = 5, n_c = 3)     # define the network
        self.filter_kernel_width = 20
        self.time = 0
        self.neigh, self.embedder = init_knn(os.path.join(dirname, 'hand_knn/dataset_embedded.npz'))
        self.spider_label = -1
        self.t_xywh_list = [[80,60,40,40]] * self.filter_kernel_width
        self.weight = [1]*20
        # 调整gamma看效果，这里先设置为1.1
        for i in range(20):
          self.weight[i] = 1 / 1.1**(20-i)
        #deepsort
        weights=[0.2, 0.2, 0.2]
        model_path = os.path.join(dirname, 'deep/checkpoint/ckpt.t7')
        max_dist = 0.2
        min_confidence = 0.3
        nms_max_overlap = 0.5
        max_iou_distance = 0.7
        max_age = 70
        n_init = 3
        nn_budget = 100
        use_cuda = torch.cuda.is_available()

        self.deepsort = DeepSort(model_path, max_dist, min_confidence, 
                            nms_max_overlap, max_iou_distance, 
                            max_age, n_init, nn_budget, use_cuda)
    def load(self, PATH):
        # self.model = torch.load(PATH)
        # self.model.eval()

        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def forward(self, img):   
        ##Add a dimension
        #print('type:',type(img))
        #frame = np.expand_dims(img.transpose((1,0,2)), 0) #/ 255
        frame = img
        
        t_xywh = [80,60,40,40]
        interestedperson = [0]
        
        #TODO: convert frame to PILLOW!!
        outputs_OD = model([frame]) 
        mask = outputs_OD.xywh[0][:,-1] == 0
        tlwh = outputs_OD.xywh[0][:,:4][mask].to('cpu').to(torch.int16)
        tlbr = outputs_OD.xyxy[0][:,:4][mask].to('cpu').to(torch.int16)
        conf = outputs_OD.xywh[0][:,4][mask].to('cpu')                               
        img  = outputs_OD.imgs[0]
        
        outputs_MOT = self.deepsort.update(tlwh,tlbr,conf, img)
        if len(outputs_MOT) > 0:
            identities = outputs_MOT[:, -1]
            if self.spider_label== -1:
                print('\033[0;31m No Spider. Reinitialize!!!!! \033[0m')
                t_xywh = [80,60,10,10]
                interestedperson = [0]
                for output in outputs_MOT:
                    top,left,bottom,right  = output[:4] #tlbr,id
                    person_image = img[top:bottom,left:right, :]
                    if person_image.shape[0]==0:
                        continue
                    hand_class, person_image = hand_pose_recognition(person_image, self.neigh, self.embedder)

                    if len(hand_class) > 1:
                        if (hand_class[0] == 'spider' or hand_class[1] == 'spider'):
                            self.spider_label = output[-1]
                            print("spider_label is : ", self.spider_label)
                    elif hand_class == 'spider': 
                        self.spider_label = output[-1]
                        self.t_xywh_list = [[80,60,40,40]] * self.filter_kernel_width
                        print("spider_label is : ", self.spider_label)
            
            else:
                mask = outputs_MOT[:,-1]==self.spider_label
                identities[mask] = identities[mask]+10000
                if len(outputs_MOT[:,:4][mask])>0:
                    self.time = 0
                    tt, tl, tb, tr = outputs_MOT[:,:4][mask][0]
                    tx = int((tt+tb)/2)
                    ty = int((tl+tr)/2)
                    tw = int(tr-tl)
                    th = int(tb-tt)
                    t_xywh = [tx,ty,tw,th]
                    
                    interestedperson = [1]
                    # img = cv2.circle(img, (tx,ty), radius=0, color=(0, 0, 255), thickness=-1)
                    # print('spider bounding box: ',np.array([tx,ty,tw,th]))
                    print('Tracking')
                else:
                    self.time += 1
                    if self.time > 30:
                        self.spider_label = -1
                    
                    t_xywh = [80,60,40,40]
                    #t_xywh_list = t_xywh_list[:-1]+t_xywh
                    interestedperson = [0]
                    print('Spider out of range')
        if interestedperson[0] == 1:
            self.t_xywh_list = self.t_xywh_list[1:]+[t_xywh]
            #print(np.stack(self.t_xywh_list))
            print(np.mean(np.stack(self.t_xywh_list),axis=0).astype(int))
        #print(self.t_xywh_list)
        
            #draw box
            #debug plot
            # pil_image = Image.frombytes('RGB', (160, 120), recvd_image)
            #opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            #opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
            #bbox_tlbr = outputs_MOT[:, :4]
            #identities = outputs_MOT[:, -1]
            #opencvImage = deepsort.draw_boxes(opencvImage, bbox_tlbr, identities)
            # else:
                
                #return t_xywh,0
            
        
            # img = deepsort.draw_boxes(img, bbox_xyxy, identities)
        #cv2.imshow('Test window',opencvImage)
        #cv2.waitKey(1)
        
        # img = deepsort.draw_boxes(img, bbox_tlbr, identities)
        
        return np.average(np.stack(self.t_xywh_list),axis=0,weights=self.weight).astype(int), interestedperson

