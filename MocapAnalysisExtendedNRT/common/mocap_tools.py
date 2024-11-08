import pandas
import math
import numpy as np
import transforms3d as t3d
from common import bvh_tools as bvh
from common.quaternion import qfix
import copy

class Mocap_Tools:

    def bvh_to_mocap(self, bvh_data):
        
        mocap_data = {}
        
        mocap_data["frame_rate"] = 1.0 / bvh_data.framerate
        
        rot_channels = list(bvh_data.skeleton.values())[0]["channels"][3:]
        rot_channel_names = ["Xrotation", "Yrotation", "Zrotation"]
        mocap_data["rot_sequence"]  = [ rot_channel_names.index(rot_channel) for rot_channel in rot_channels ]
        
        self._create_skeleton_data(bvh_data, mocap_data)
        self._create_motion_data(bvh_data, mocap_data)
        
        return mocap_data
    
    def mocap_to_bvh(self, mocap_data):
        
        bvh_data = bvh.BVH_Data()
        bvh_data.framerate = 1.0 / mocap_data["frame_rate"]
        bvh_data.root_name = mocap_data["skeleton"]["root"]
        
        bvh_channel_names, bvh_channels = self._create_bvh_channel_names(mocap_data)
        bvh_data.channel_names = bvh_channel_names
        
        bvh_skeleton = self._create_bvh_skeleton(mocap_data, bvh_channels)
        bvh_data.skeleton = bvh_skeleton
        
        bvh_frames = self._create_bvh_frames(mocap_data, bvh_channels)
        bvh_data.values = bvh_frames
        
        return bvh_data
    
    def local_to_world(self, rot_local, pos_local, skeleton):
        
        root_name = skeleton["root"]
        joint_names = skeleton["joints"]
        root_index = joint_names.index(root_name)
        parents = skeleton["parents"]
        children = skeleton["children"]
        offsets = skeleton["offsets"]
        
        root_positions = pos_local[:, root_index, :]

        frame_count = rot_local.shape[0]
        joint_count = rot_local.shape[1]
        
        positions_world = []
        rotations_world = []
        
        for fI in range(frame_count):
            
            frame_positions_world = []
            frame_rotations_world = []
            
            for jI in range(joint_count):
                
                if parents[jI] == -1:
                    frame_positions_world.append(root_positions[fI])
                    frame_rotations_world.append(rot_local[fI, 0])
                else:
                    frame_positions_world.append(t3d.quaternions.rotate_vector(offsets[jI], frame_rotations_world[parents[jI]]) + frame_positions_world[parents[jI]])
                
                    if len(children[jI]) > 0:
                        frame_rotations_world.append(t3d.quaternions.qmult(frame_rotations_world[parents[jI]], rot_local[fI, jI]))
                    else:
                        frame_rotations_world.append(t3d.quaternions.qeye())
                        
                #print("jI ", jI, " fpw ", frame_positions_world[jI].shape, " frw ", frame_rotations_world[jI].shape)
                
            frame_positions_world = np.stack(frame_positions_world, axis=0)
            frame_rotations_world = np.stack(frame_rotations_world, axis=0)
            
            #print("fI ", fI, " fpw ", frame_positions_world.shape, " frw ", frame_rotations_world.shape)
            
            positions_world.append(frame_positions_world)
            rotations_world.append(frame_rotations_world)
            
        positions_world = np.stack(positions_world, axis=0)
        rotations_world = np.stack(rotations_world, axis=0)
        
        return positions_world, rotations_world

    def euler_to_quat(self, rotations_euler, rot_sequence):
        
        # rotations_euler shape: F x J x D (F: frame count, J: joint count, D: 3 (with angles in degrees))
        # rot_sequence shape: D (D: 3 (integer indices))
        
        frame_count = rotations_euler.shape[0]
        joint_count = rotations_euler.shape[1]
        
        rotations_quat = []
        
        for fI in range(frame_count):
            
            joint_rotations_quat = []
            
            for jI in range(joint_count):
        
                # convert degrees to radians
                euler_x = rotations_euler[fI, jI, 0]/180.0 * math.pi;
                euler_y = rotations_euler[fI, jI, 1]/180.0 * math.pi;
                euler_z = rotations_euler[fI, jI, 2]/180.0 * math.pi;

                # convert euler rotation to quaternion
                quat_identity = t3d.quaternions.qeye()

                quat_x = t3d.quaternions.axangle2quat([1, 0, 0], euler_x)
                quat_y = t3d.quaternions.axangle2quat([0, 1, 0], euler_y)
                quat_z = t3d.quaternions.axangle2quat([0, 0, 1], euler_z)
                
                joint_rotation_quat = t3d.quaternions.qeye()
                
                rotations = [quat_x, quat_y, quat_z]
                for rot_index in rot_sequence:
                    joint_rotation_quat = t3d.quaternions.qmult(joint_rotation_quat, rotations[rot_index])
                
                joint_rotations_quat.append(joint_rotation_quat)
                
            joint_rotations_quat = np.stack(joint_rotations_quat, axis=0)
            
            rotations_quat.append(joint_rotations_quat)
                
        
        rotations_quat = np.stack(rotations_quat, axis=0)
        rotations_quat = qfix(rotations_quat)
        
        return rotations_quat


    def quat_to_euler(self, rotations_quat, rot_sequence):
        
        # rotations_quat shape: F x J x D (F: frame count, J: joint count, D: 4 )
        # rot_sequence shape: D (D: 3 (integer indices))
        
        frame_count = rotations_quat.shape[0]
        joint_count = rotations_quat.shape[1]
        
        rotations_euler = []
        
        for fI in range(frame_count):
            
            joint_rotations_euler = []
            
            for jI in range(joint_count):
                
                rotation_quat = rotations_quat[fI, jI]
                #rotation_euler = np.array(t3d.euler.quat2euler(rotation_quat, axes="sxyz"))
                rotation_euler = np.array(t3d.euler.quat2euler(rotation_quat, axes="syxz"))
                rotation_euler  *= 180.0 / math.pi
                rotation_euler = np.array((rotation_euler[1], rotation_euler[0], rotation_euler[2]))

                joint_rotations_euler.append(rotation_euler)

            rotations_euler.append(joint_rotations_euler)
                
        rotations_euler = np.stack(rotations_euler, axis=0)
                
        return rotations_euler
    
    def mocap_excerpt(self, mocap_data, start_frame=-1, end_frame=-1):
        
        mocap_data_excerpt = copy.deepcopy(mocap_data)
        
        motion_data = mocap_data_excerpt["motion"]

        full_frame_count = motion_data[list(motion_data.keys())[0]].shape[0]

        if start_frame == -1:
            start_frame = 0
    
        if end_frame == -1 or end_frame > full_frame_count:
            end_frame = full_frame_count
            
        for key in motion_data.keys():
            
            motion_data[key] = motion_data[key][start_frame:end_frame, ...]
            
            #values = values[start_frame:end_frame, ...]
            
        return mocap_data_excerpt
            
 
    def _create_skeleton_data(self, bvh_data, mocap_data):
        
        skeleton_data = {}
        
        joint_names = [ joint_name for joint_name in bvh_data.skeleton.keys() ]
        joint_parent_names = [ joint_info["parent"] for joint_name, joint_info in bvh_data.skeleton.items() ]
        joint_parent_indices = [joint_names.index(joint_parent_name) if joint_parent_name in joint_names else -1 for joint_parent_name in joint_parent_names ]
        
        joint_child_indices = []
        for joint_info in bvh_data.skeleton.values():
            child_indices = []
            for child_name in joint_info["children"]:
                child_indices.append(joint_names.index(child_name))
            joint_child_indices.append(child_indices)
            
        joint_offsets = [ joint_info["offsets"] for joint_info in bvh_data.skeleton.values() ]
        joint_offsets = np.stack(joint_offsets, axis=0)
        
        skeleton_data["root"] = bvh_data.root_name
        skeleton_data["joints"] = joint_names
        skeleton_data["parents"] = joint_parent_indices
        skeleton_data["children"] = joint_child_indices
        skeleton_data["offsets"] = joint_offsets
        
        mocap_data["skeleton"] = skeleton_data
        
        return mocap_data
    
    def _create_motion_data(self, bvh_data, mocap_data):
        
        motion = {}
        
        joint_names = mocap_data["skeleton"]["joints"]
        joint_count = len(joint_names)
        rot_sequence = mocap_data["rot_sequence"]
        
        bvh_frames = bvh_data.values
        frame_count = bvh_frames.shape[0]
        
        bvh_frames_column_names = [ column for column in bvh_data.values.columns ]
        bvh_channels = set(bvh_data.channel_names)
        bvh_channel_joint_names = set([channel[0] for channel in bvh_channels])
        bvh_channel_value_names = ["Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"]
        
        motion_translation = []
        motion_euler_rotation = []

        for joint_name in joint_names:

            if joint_name in bvh_channel_joint_names:
                joint_frames_combined = []      
            
                for i, value_name in enumerate(bvh_channel_value_names):
                    column_name = joint_name + "_" + value_name
                
                    if column_name in bvh_frames_column_names:
                        joint_frames_combined.append(np.array(bvh_frames[column_name]))
                    
                        #print("colname ", column_name, " values ", np.array(bvh_frames[column_name])[0])
                    
                    else:
                        joint_frames_combined.append(np.zeros(frame_count))

                joint_translations = joint_frames_combined[:3]
                joint_rotations = joint_frames_combined[3:]
                
                joint_translations = np.array(joint_translations)
                joint_rotations = np.array(joint_rotations)

                joint_translations = np.transpose(joint_translations)
                joint_rotations = np.transpose(joint_rotations)
                
                motion_translation.append(joint_translations)
                motion_euler_rotation.append(joint_rotations)
            else:
                
                joint_translations = np.zeros((frame_count, 3))
                joint_rotations = np.zeros((frame_count, 3))
                
                motion_translation.append(joint_translations)
                motion_euler_rotation.append(joint_rotations)
                
        motion_translation = np.stack(motion_translation, axis=1)
        motion_euler_rotation = np.stack(motion_euler_rotation, axis=1)
        
        motion["pos_local"] = motion_translation
        motion["rot_local_euler"] = motion_euler_rotation
        #motion["rot_local"] = self.euler_to_quat(motion_euler_rotation, rot_sequence)

        mocap_data["motion"] = motion
        
        return mocap_data

    def _create_bvh_channel_names(self, mocap_data):
        
        joints = mocap_data["skeleton"]["joints"]
        children = mocap_data["skeleton"]["children"]
        rot_sequence = mocap_data["rot_sequence"]
        
        pos_channel_names = ["Xposition", "Yposition", "Zposition"]
        rot_channel_names = ["Xrotation", "Yrotation", "Zrotation"]

        bvh_channels = pos_channel_names + [ rot_channel_names[i] for i in rot_sequence ]
        
        bvh_channel_names = []
        
        for jI, joint_name in enumerate(joints):
            
            # ignore nub joints
            if len(children[jI]) == 0:
                continue

            for channel_name in bvh_channels:
                
                bvh_channel_names.append((joint_name, channel_name))
            
        
        return bvh_channel_names, bvh_channels

    
    def _create_bvh_skeleton(self, mocap_data, bvh_channels):
        
        joints = mocap_data["skeleton"]["joints"]
        parents  = mocap_data["skeleton"]["parents"]
        children  = mocap_data["skeleton"]["children"]
        offsets = mocap_data["skeleton"]["offsets"]
        
        bvh_skeleton = {}
        
        for jI, joint_name in enumerate(joints):
            
            bvh_joint = {}
            
            # add name of parent joint
            if parents[jI] != -1:
                bvh_joint["parent"]  = joints[parents[jI]]
            else:
                bvh_joint["parent"] = None
            
            # add names of child joints
            bvh_joint["children"] = [ joints[child] for child in children[jI] ]
            
            # add joint offset
            bvh_joint["offsets"] = offsets[jI].tolist()
            
            # add joint channels
            if len(children[jI]) > 0:
                bvh_joint["channels"] = bvh_channels
            else: 
                bvh_joint["channels"] = []
            
                    
            bvh_skeleton[joint_name] = bvh_joint
            
        
        return bvh_skeleton
    
    def _create_bvh_frames(self, mocap_data, bvh_channels):
        
        joints = mocap_data["skeleton"]["joints"]
        children = mocap_data["skeleton"]["children"]
        pos_local = mocap_data["motion"]["pos_local"]
        rot_euler = mocap_data["motion"]["rot_local_euler"]
        rot_sequence = mocap_data["rot_sequence"]
        
        pos_channels = ["Xposition", "Yposition", "Zposition"]
        rot_channels = ["Xrotation", "Yrotation", "Zrotation"]
        
        bvh_frames = {}
        
        for jI, joint_name in enumerate(joints):
            
            if len(children[jI]) == 0:
                continue
            
            for channel_name in bvh_channels:
                
                col_name = joint_name + "_" + channel_name
                channel_values = []
                
                if channel_name in pos_channels:
                    channel_values = pos_local[:, jI, pos_channels.index(channel_name)].tolist()
                    
                elif channel_name in rot_channels:
                    channel_values = rot_euler[:, jI, rot_channels.index(channel_name)].tolist()

                else:
                    continue
                
                bvh_frames[col_name] = channel_values
                
        dataFrame = pandas.DataFrame(bvh_frames)
        
        return dataFrame