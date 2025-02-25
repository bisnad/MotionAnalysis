import torch
from torch import nn
import numpy as np

from common.quaternion import qmul, qrot, qnormalize_np, slerp
from common.quaternion_torch import qfix

config = {"skeleton": None,
          "model": None,
          "seq_window_length": 20,
          "seq_window_overlap": 10
          }

class MotionSynthesis():
    
    def __init__(self, config):
        self.skeleton = config["skeleton"]
        self.model = config["model"]
        self.seq_window_length = config["seq_window_length"]
        self.seq_window_overlap = config["seq_window_overlap"]
        
        self.seq_window_offset = self.seq_window_length - self.seq_window_overlap
        
        self.cluster_label = 0
        self.mocap_excerpts = torch.from_numpy(self.model.get_cluster_mocap_excerpts(self.cluster_label))
        self.mocap_excerpt_count = self.mocap_excerpts.shape[0]

        self.seq_length = self.mocap_excerpts[0].shape[0]
        self.joint_count = self.mocap_excerpts[0].shape[1]
        self.joint_dim = self.mocap_excerpts[0].shape[2]
        self.pose_dim = self.joint_count * self.joint_dim
        
        self.joint_children = self.skeleton["children"]
        
        self._create_edge_list()

        self.pred_pose = None
        self.synth_pose_wpos = None
        
        self.mocap_excerpt_index = 0
        self.excerpt_frame_index = 0
        
    def _create_edge_list(self):
        
        self.edge_list = []
        
        for parent_joint_index in range(len(self.joint_children)):
            for child_joint_index in self.joint_children[parent_joint_index]:
                self.edge_list.append([parent_joint_index, child_joint_index])
                
    def setClusterLabel(self, label):
        
        self.mocap_excerpt_index = 0
        self.excerpt_frame_index = 0
        
        self.cluster_label = min(label, self.model.get_label_count() - 1)
        self.mocap_excerpts = torch.from_numpy(self.model.get_cluster_mocap_excerpts(self.cluster_label))
        self.mocap_excerpt_count = self.mocap_excerpts.shape[0]
        
    def selectMotionFeature(self, fleatureName):
        
        self.model.select_motion_feature(fleatureName)
        self.model.create_clusters()
        
        self.cluster_label = 0
        self.mocap_excerpt_index = 0
        self.excerpt_frame_index = 0
        
        self.mocap_excerpts = torch.from_numpy(self.model.get_cluster_mocap_excerpts(self.cluster_label))
        self.mocap_excerpt_count = self.mocap_excerpts.shape[0]
        
    def update(self):
        
        self._gen()
        
        #print("self.pred_pose s ", self.pred_pose.shape)
        
        self.synth_pose_wpos = self.pred_pose

    def _gen(self):
        
        if self.excerpt_frame_index >= self.seq_window_length:
            self.mocap_excerpt_index += 1
            self.excerpt_frame_index  = self.seq_window_overlap

            if self.mocap_excerpt_index >= self.mocap_excerpt_count:
                self.mocap_excerpt_index = 0
        
        if self.mocap_excerpt_index == 0 and self.excerpt_frame_index < self.seq_window_offset:
            
            #print("result_seq[", rfI, "] = excerpts[", eI, "][", feI, "]")
            
            self.pred_pose = self.mocap_excerpts[self.mocap_excerpt_index][self.excerpt_frame_index]
            
        elif self.mocap_excerpt_index == self.mocap_excerpt_count - 1 and self.excerpt_frame_index  > self.seq_window_overlap:
            
            #print("result_seq[", rfI, "] = excerpts[", eI, "][", feI, "]")
            
            self.pred_pose = self.mocap_excerpts[self.mocap_excerpt_index][self.excerpt_frame_index] 
        elif self.mocap_excerpt_index < self.mocap_excerpt_count - 1:
            if self.excerpt_frame_index  < self.seq_window_offset:
                
                #print("result_seq[", rfI, "] = excerpts[", eI, "][", feI, "]")
                
                self.pred_pose = self.mocap_excerpts[self.mocap_excerpt_index][self.excerpt_frame_index]
            else:

                bI = self.excerpt_frame_index  - self.seq_window_offset
                blendValue = bI / (self.seq_window_overlap - 1)
                
                #print("result_seq[", rfI, "] = excerpts[", eI, "][", feI, "] + excerpts[", (eI+1), "][", (feI - window_offset), "] blend ", blendValue)

                for jI in range(self.joint_count):
                    
                    pos1 = self.mocap_excerpts[self.mocap_excerpt_index][self.excerpt_frame_index , jI]
                    pos2 = self.mocap_excerpts[self.mocap_excerpt_index + 1][self.excerpt_frame_index  - self.seq_window_offset, jI]
                    pos_blend = pos1 * (1.0 - blendValue) + pos2 * blendValue
                    
                    self.pred_pose[jI] = pos_blend
        
        self.excerpt_frame_index += 1
