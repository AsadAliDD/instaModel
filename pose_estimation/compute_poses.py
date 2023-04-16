import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses, get_similarity
from val import normalize, pad_width
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import pandas as pd
from tqdm import tqdm 

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider,path, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    # previous_poses = []
    # delay = 33

    # frame = image_provider.__iter__().__next__()
    # vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

    poses=[]

    for filename,img in tqdm(zip(image_provider,path)):
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)
        current_poses = []
        confidences =[]
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
            confidences.append(pose.confidence)


        current_poses=np.array(current_poses)
        confidences=np.array(confidences)
        # Geting top 2 confident pose
        # ind=np.argsort(confidences)[::-1][:2]
        # print (ind)
      
        # top_poses=current_poses[ind]
        # ind=np.argpartition(confidences, -2)
        try:
            ind=np.argsort(confidences)[::-1][:2]
        #     # ind=np.argpartition(confidences, -2)[-2:]
        #     print ("YES")
        #     print (ind)
            top_poses=current_poses[ind]
        except:
            print ("Only 1 pose")
            top_poses=current_poses

        poses.append(top_poses)
        # print (top_poses)



        # Visulization
        bin_img=np.zeros(img.shape)
        for pose in top_poses:
            # if(pose.confidence>20):
            pose.draw(img,bin_img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        # print ("OG: ",orig_img.shape)
        # print ("Test: ",img.shape)
        for pose in top_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                

        
        # filename for saving
        filename=filename.rsplit('.',1)[0]+"_pose.png"
        print (filename)
        cv2.imwrite(filename, bin_img)  
        


        # cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
        # f,ax=plt.subplots(1,2)
        # ax[0].imshow(img)
        # ax[1].imshow(bin_img)
        # f.suptitle("Pose Mask")



        # fig=go.Figure(go.Image(z=img))
        # fig.show()

        # print ("Pose Mask")
        # fig=go.Figure(go.Image(z=bin_img))
        # fig.show()
        # plt.imshow(bin_img)
    return poses
    #     # vid_writer.write(img)
    #     key = cv2.waitKey(delay)
    #     if key == 27:  # esc
    #         return
    #     elif key == 112:  # 'p'
    #         if delay == 33:
    #             delay = 0
    #         else:
    #             delay = 33
    # # vid_writer.release()
    # cv2.destroyAllWindows()






def save_poses(chkpt_path,images):

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(chkpt_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(images)
    poses=run_demo(net,images, frame_provider, 256, 1, 1,1 )
    return poses





