import cv2
import argparse
import mediapipe as mp
import numpy as np
import time
import datetime
import os
import sys
import urllib.request
import math
from scipy.spatial.transform import Rotation as R
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSizePolicy, QSlider, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap

import motion_sender 

try:
    import fbx
except ImportError:
    print("WARNING: Autodesk 'fbx' module not found. FBX export will not work. Install with 'pip install fbx'.")

# ==============================================================================
# 1. CONSTANTS & TOPOLOGY
# ==============================================================================

VISUAL_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

_base_pose_kinematic = [
    edge for edge in VISUAL_POSE_CONNECTIONS 
    if edge not in [(11, 23), (12, 24), (23, 24)] 
]
KINEMATIC_POSE_CONNECTIONS = _base_pose_kinematic + [
    (0, 9), (0, 10),   
    (0, 11), (0, 12),  
    (0, 23), (0, 24)   
]

VISUAL_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), 
    (0, 5), (5, 6), (6, 7), (7, 8), 
    (5, 9), (9, 10), (10, 11), (11, 12), 
    (9, 13), (13, 14), (14, 15), (15, 16), 
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
]

KINEMATIC_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       
    (0, 5), (5, 6), (6, 7), (7, 8),       
    (0, 9), (9, 10), (10, 11), (11, 12),  
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20) 
]

def build_parents_from_edges(num_nodes, edges, root_node=0):
    parents = [-1] * num_nodes
    adj = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited, queue = {root_node}, [root_node]
    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                parents[neighbor] = curr
                queue.append(neighbor)
    return parents

def configure_topology(enable_pose, enable_hands):
    if enable_pose and enable_hands:
        num_nodes = 75
        
        hand_plane_edges = [(15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20)]
        kp_edges = [e for e in KINEMATIC_POSE_CONNECTIONS if e not in hand_plane_edges]
        
        parents = [-2] * num_nodes 
        
        p_pose = build_parents_from_edges(33, kp_edges, 0)
        for i in range(33):
            if i in [17, 18, 19, 20, 21, 22]:
                parents[i] = -2 
            else:
                parents[i] = p_pose[i]
        
        p_hand = build_parents_from_edges(21, KINEMATIC_HAND_CONNECTIONS, 0)
        for i in range(21):
            parents[33 + i] = 15 if p_hand[i] == -1 else p_hand[i] + 33
            parents[54 + i] = 16 if p_hand[i] == -1 else p_hand[i] + 54
            
        slices = {'pose': (0, 33), 'Left': (33, 54), 'Right': (54, 75)}
        attachments = {'Left': 15, 'Right': 16}
        
    elif enable_pose:
        num_nodes = 33
        parents = build_parents_from_edges(33, KINEMATIC_POSE_CONNECTIONS, 0)
        slices = {'pose': (0, 33)}
        attachments = {}
        
    elif enable_hands:
        num_nodes = 42
        parents = [-2] * num_nodes
        p_hand = build_parents_from_edges(21, KINEMATIC_HAND_CONNECTIONS, 0)
        for i in range(21):
            parents[i] = -1 if p_hand[i] == -1 else p_hand[i]
            parents[21 + i] = -1 if p_hand[i] == -1 else p_hand[i] + 21
        slices = {'Left': (0, 21), 'Right': (21, 42)}
        attachments = {'Left': -1, 'Right': -1}
        
    return np.array(parents), num_nodes, slices, attachments

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return x
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class PoseFilter:
    def __init__(self, num_nodes, min_cutoff=0.5, beta=1.0):
        self.num_nodes = num_nodes
        self.filters = None
        self.min_cutoff = min_cutoff
        self.beta = beta

    def update_params(self, min_cutoff, beta):
        self.min_cutoff = min_cutoff
        self.beta = beta
        if self.filters is not None:
            for i in range(self.num_nodes):
                for j in range(3):
                    self.filters[i][j].min_cutoff = min_cutoff
                    self.filters[i][j].beta = beta

    def __call__(self, t, pos_world):
        if self.filters is None:
            self.filters = [[OneEuroFilter(t, pos_world[i, j], min_cutoff=self.min_cutoff, beta=self.beta) 
                            for j in range(3)] for i in range(self.num_nodes)]
            return pos_world
        filtered_pos = np.zeros_like(pos_world)
        for i in range(self.num_nodes):
            for j in range(3):
                filtered_pos[i, j] = self.filters[i][j](t, pos_world[i, j])
        return filtered_pos

# ==============================================================================
# 2. OSC & FBX EXPORT
# ==============================================================================

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9007
osc_sender = motion_sender.OscSender(motion_sender.config)

def osc_send_tracked_data(tracked_data):
    if not tracked_data: return
    pose_data = tracked_data["skeleton_0"]["pose"]
    osc_sender.send("/mocap/0/joint/pos_world", pose_data["pos_world"])
    osc_sender.send("/mocap/0/joint/pos_local", pose_data["pos_rel"])
    osc_sender.send("/mocap/0/joint/rot_world", pose_data["quat_world"])
    osc_sender.send("/mocap/0/joint/rot_local", pose_data["quat_rel"])

def export_fbx(filename, frames_data, parents, fps=30.0):
    if 'fbx' not in globals(): return
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    scene = fbx.FbxScene.Create(manager, "MocapScene")

    num_joints = len(parents)
    nodes = []

    try:
        skel_root_enum = fbx.FbxSkeleton.EType.eRoot
        skel_limb_enum = fbx.FbxSkeleton.EType.eLimbNode
    except AttributeError:
        skel_root_enum = fbx.FbxSkeleton.eRoot
        skel_limb_enum = fbx.FbxSkeleton.eLimbNode

    try:
        interp_linear_enum = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
    except AttributeError:
        try: interp_linear_enum = fbx.FbxAnimCurveDef.eInterpolationLinear
        except AttributeError: interp_linear_enum = 4 

    for i in range(num_joints):
        if parents[i] == -2:
            nodes.append(None)
            continue
        node = fbx.FbxNode.Create(scene, f"Joint_{i}")
        skeleton = fbx.FbxSkeleton.Create(scene, f"Skel_{i}")
        skeleton.SetSkeletonType(skel_root_enum if parents[i] == -1 else skel_limb_enum)
        node.SetNodeAttribute(skeleton)
        
        # Rest Offset Lock
        tx, ty, tz = frames_data[0]['pos_rel'][i] * 100.0 
        node.LclTranslation.Set(fbx.FbxDouble3(float(tx), float(ty), float(tz)))
        
        nodes.append(node)

    for i in range(num_joints):
        if parents[i] == -2: continue
        if parents[i] != -1 and nodes[parents[i]] is not None:
            nodes[parents[i]].AddChild(nodes[i])
        else:
            scene.GetRootNode().AddChild(nodes[i])

    anim_stack = fbx.FbxAnimStack.Create(scene, "MotionStack")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "MotionLayer")
    anim_stack.AddMember(anim_layer)

    curves = []
    for i in range(num_joints):
        if parents[i] == -2:
            curves.append(None)
            continue
        n = nodes[i]
        curves_i = []
        for prop in [n.LclTranslation, n.LclRotation]:
            for axis in ["X", "Y", "Z"]:
                c = prop.GetCurve(anim_layer, axis, True)
                c.KeyModifyBegin()
                curves_i.append(c)
        curves.append(curves_i)

    fbx_time = fbx.FbxTime()
    for frame_idx, frame in enumerate(frames_data):
        fbx_time.SetSecondDouble(frame_idx / fps)
        pos_rel, quat_rel = frame['pos_rel'], frame['quat_rel']
        for i in range(num_joints):
            if parents[i] == -2: continue
            tx, ty, tz = pos_rel[i] * 100.0 
            rx, ry, rz = R.from_quat(quat_rel[i]).as_euler('xyz', degrees=True)
            for curve, val in zip(curves[i], [tx, ty, tz, rx, ry, rz]):
                k_idx = curve.KeyAdd(fbx_time)[0]
                curve.KeySet(k_idx, fbx_time, float(val), interp_linear_enum)

    for i in range(num_joints):
        if parents[i] == -2: continue
        for c in curves[i]: c.KeyModifyEnd()

    unroll_filter = fbx.FbxAnimCurveFilterUnroll()
    for i in range(num_joints):
        if parents[i] == -2: continue
        rot_curve_node = nodes[i].LclRotation.GetCurveNode(anim_layer, False)
        if rot_curve_node:
            try: unroll_filter.Apply(rot_curve_node)
            except Exception:
                status = fbx.FbxStatus()
                unroll_filter.Apply(rot_curve_node, status)

    exporter = fbx.FbxExporter.Create(manager, "")
    if exporter.Initialize(filename, -1, manager.GetIOSettings()): 
        exporter.Export(scene)
    
    print(f"\n---> Successfully exported FBX to: {filename}")

# ==============================================================================
# 3. KINEMATICS & MATH
# ==============================================================================

def extract_landmarks(landmark_list):
    return np.array([[lm.x, -lm.y, -lm.z] for lm in landmark_list])

def get_shortest_quat(v1, v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0]) 
        
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    elif dot < -0.999999:
        ortho = np.cross(np.array([1.0, 0.0, 0.0]), v1)
        if np.linalg.norm(ortho) < 1e-6:
            ortho = np.cross(np.array([0.0, 1.0, 0.0]), v1)
        ortho = ortho / np.linalg.norm(ortho)
        return np.array([ortho[0], ortho[1], ortho[2], 0.0])
    
    cross = np.cross(v1, v2)
    q = np.array([cross[0], cross[1], cross[2], 1.0 + dot])
    return q / np.linalg.norm(q)


def compute_kinematics(pos_world, parents, offsets):
    num_nodes = len(pos_world)
    pos_rel = np.zeros_like(pos_world)
    quat_world = np.zeros((num_nodes, 4))
    quat_rel = np.zeros((num_nodes, 4))

    quat_world[:, 3] = 1.0
    quat_rel[:, 3] = 1.0

    children = {i: [] for i in range(num_nodes)}
    for i, p in enumerate(parents):
        if p >= 0: 
            children[p].append(i)

    topo_order = []
    queue = [i for i, p in enumerate(parents) if p == -1]
    while queue:
        curr = queue.pop(0)
        topo_order.append(curr)
        queue.extend(children[curr])

    for i in topo_order:
        p = parents[i]
        child_list = children[i]
        
        if p == -1:
            if len(child_list) >= 2:
                v_rest = np.array([offsets[c] for c in child_list])
                v_curr = np.array([pos_world[c] - pos_world[i] for c in child_list])
                valid_idx = (np.linalg.norm(v_rest, axis=1) > 1e-6) & (np.linalg.norm(v_curr, axis=1) > 1e-6)
                
                if sum(valid_idx) >= 2:
                    try:
                        rot, _ = R.align_vectors(v_curr[valid_idx], v_rest[valid_idx])
                        quat_world[i] = rot.as_quat()
                    except Exception:
                        quat_world[i] = np.array([0.0, 0.0, 0.0, 1.0])
                else:
                    quat_world[i] = np.array([0.0, 0.0, 0.0, 1.0])
            elif len(child_list) == 1:
                v_rest = offsets[child_list[0]]
                v_curr = pos_world[child_list[0]] - pos_world[i]
                quat_world[i] = get_shortest_quat(v_rest, v_curr)
            else:
                quat_world[i] = np.array([0.0, 0.0, 0.0, 1.0])
                
        else:
            if len(child_list) == 0:
                quat_world[i] = quat_world[p]
            else:
                c = child_list[0]
                v_rest = offsets[c]
                v_curr = pos_world[c] - pos_world[i]
                
                try:
                    v_expected = R.from_quat(quat_world[p]).apply(v_rest)
                    q_swing = get_shortest_quat(v_expected, v_curr)
                    r_swing = R.from_quat(q_swing)
                    r_parent = R.from_quat(quat_world[p])
                    quat_world[i] = (r_swing * r_parent).as_quat()
                except Exception:
                    quat_world[i] = quat_world[p] 

    for i in range(num_nodes):
        if parents[i] == -2: continue 
        p = parents[i]
        if p != -1:
            try:
                r_p_inv = R.from_quat(quat_world[p]).inv()
                quat_rel[i] = (r_p_inv * R.from_quat(quat_world[i])).as_quat()
            except Exception:
                quat_rel[i] = np.array([0.0, 0.0, 0.0, 1.0])
            pos_rel[i] = offsets[i] 
        else:
            quat_rel[i] = quat_world[i]
            pos_rel[i] = pos_world[i]

    return pos_rel, quat_world, quat_rel

def fallback_missing_joints(current_pos, last_pos, start_idx, end_idx, attach_idx=-1):
    if attach_idx != -1:
        offset = current_pos[attach_idx] - last_pos[attach_idx]
        current_pos[start_idx:end_idx] = last_pos[start_idx:end_idx] + offset
    else:
        current_pos[start_idx:end_idx] = last_pos[start_idx:end_idx]

def download_models(model_type):
    pose_model = f"pose_landmarker_{model_type}.task"
    models = {
        pose_model: f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{model_type}/float16/latest/{pose_model}",
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    }
    for filename, url in models.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)

def draw_landmarks_custom(image, landmarks, connections, color_edge=(0, 255, 0)):
    h, w, _ = image.shape
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(image, pt1, pt2, color_edge, 2)

# ==============================================================================
# 4. GUI & THREADING
# ==============================================================================

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_fps_signal = pyqtSignal(float)

    def __init__(self, args, parents, num_nodes, slices, attachments):
        super().__init__()
        self.args = args
        self.parents = parents
        self.num_nodes = num_nodes
        self.slices = slices
        self.attachments = attachments
        self._run_flag = True

        self.is_recording = False
        self.is_counting_down = False
        self.countdown_start_time = 0
        self.recorded_frames = []
        
        self.rest_offsets = np.zeros((self.num_nodes, 3))
        self.offsets_captured = np.array([p == -2 for p in self.parents], dtype=bool)

        self.pose_landmarker = None
        self.hand_landmarker = None
        self.tracking_confidence = 0.7
        self.recreate_landmarker = False
        self.pose_filter = None

        self.init_pose_landmarker()

        if self.args.hands:
            options_hand = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2, min_hand_detection_confidence=0.2, min_tracking_confidence=0.2)
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

    def init_pose_landmarker(self):
        if self.pose_landmarker:
            self.pose_landmarker.close()
        options_pose = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=f'pose_landmarker_{self.args.model}.task'),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=self.tracking_confidence)
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)
        self.recreate_landmarker = False

    def set_tracking_confidence(self, conf):
        self.tracking_confidence = conf
        self.recreate_landmarker = True

    def set_filter_params(self, min_cutoff, beta):
        if self.pose_filter is not None:
            self.pose_filter.update_params(min_cutoff, beta)

    def reset_calibration(self):
        self.offsets_captured = np.array([p == -2 for p in self.parents], dtype=bool)
        print("Rest pose calibration reset! Please hold a flat hand/T-Pose.")

    def run(self):
        source = int(self.args.source) if self.args.source.isdigit() else self.args.source
        cap = cv2.VideoCapture(source)

        last_timestamp_ms = -1
        last_pos_world = np.zeros((self.num_nodes, 3))

        prev_time = time.time()
        fps_filter = 0.0

        while self._run_flag and cap.isOpened():
            if self.recreate_landmarker:
                self.init_pose_landmarker()

            success, frame = cap.read()
            if not success: 
                break 

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_filter = 0.9 * fps_filter + 0.1 * fps
            self.update_fps_signal.emit(fps_filter)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(time.perf_counter() * 1000)
            if timestamp_ms <= last_timestamp_ms: timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            pose_res = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms) if self.pose_landmarker else None
            hand_res = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms) if self.hand_landmarker else None

            current_pos_world = np.copy(last_pos_world)
            
            left_wrist_pose = None
            right_wrist_pose = None
            if not self.args.pose and pose_res and pose_res.pose_world_landmarks:
                pose_world_lms = extract_landmarks(pose_res.pose_world_landmarks[0])
                left_wrist_pose = pose_world_lms[15]
                right_wrist_pose = pose_world_lms[16]

            if self.args.pose:
                start, end = self.slices['pose']
                if pose_res and pose_res.pose_world_landmarks:
                    current_pos_world[start:end] = extract_landmarks(pose_res.pose_world_landmarks[0])
                    if pose_res.pose_landmarks:
                        draw_landmarks_custom(frame_rgb, pose_res.pose_landmarks[0], VISUAL_POSE_CONNECTIONS, (255, 255, 255))
                else:
                    fallback_missing_joints(current_pos_world, last_pos_world, start, end, -1)

            if self.args.hands:
                updated_hands = set()
                if hand_res and hand_res.hand_world_landmarks:
                    for i, hand_lms in enumerate(hand_res.hand_world_landmarks):
                        handedness = hand_res.handedness[i][0].category_name
                        if handedness in self.slices:
                            start, end = self.slices[handedness]
                            hand_arr = extract_landmarks(hand_lms)
                            attach_idx = self.attachments[handedness]
                            
                            if attach_idx != -1:
                                hand_arr += (current_pos_world[attach_idx] - hand_arr[0])
                            elif handedness == 'Left' and left_wrist_pose is not None:
                                hand_arr += (left_wrist_pose - hand_arr[0])
                            elif handedness == 'Right' and right_wrist_pose is not None:
                                hand_arr += (right_wrist_pose - hand_arr[0])

                            current_pos_world[start:end] = hand_arr
                            updated_hands.add(handedness)
                            if hand_res.hand_landmarks:
                                draw_landmarks_custom(frame_rgb, hand_res.hand_landmarks[i], VISUAL_HAND_CONNECTIONS, (0, 255, 0))

                for handedness in ['Left', 'Right']:
                    if handedness in self.slices and handedness not in updated_hands:
                        start, end = self.slices[handedness]
                        if not self.args.pose and ((handedness == 'Left' and left_wrist_pose is not None) or (handedness == 'Right' and right_wrist_pose is not None)):
                            offset = (left_wrist_pose if handedness == 'Left' else right_wrist_pose) - last_pos_world[start]
                            current_pos_world[start:end] = last_pos_world[start:end] + offset
                        else:
                            fallback_missing_joints(current_pos_world, last_pos_world, start, end, self.attachments[handedness])

            if self.pose_filter is None:
                self.pose_filter = PoseFilter(self.num_nodes, min_cutoff=0.1, beta=5.0)

            t_sec = timestamp_ms / 1000.0
            filtered_pos_world = self.pose_filter(t_sec, current_pos_world)

            if not np.all(self.offsets_captured):
                for i in range(self.num_nodes):
                    if not self.offsets_captured[i]:
                        active = False
                        for slice_name, (start, end) in self.slices.items():
                            if start <= i < end:
                                if slice_name == 'pose' and pose_res and pose_res.pose_world_landmarks:
                                    active = True
                                elif slice_name in ['Left', 'Right'] and slice_name in (updated_hands if self.args.hands else []):
                                    dist = np.linalg.norm(current_pos_world[start + 9] - current_pos_world[start])
                                    if dist > 0.04: 
                                        active = True
                                break
                                
                        if active:
                            p = self.parents[i]
                            if p == -1:
                                self.rest_offsets[i] = current_pos_world[i].copy()
                                self.rest_offsets[i, [0, 2]] = 0.0 
                                self.offsets_captured[i] = True
                            else:
                                self.rest_offsets[i] = current_pos_world[i] - current_pos_world[p]
                                self.offsets_captured[i] = True

            if np.any(self.offsets_captured):
                pos_rel, quat_world, quat_rel = compute_kinematics(filtered_pos_world, self.parents, self.rest_offsets)
            else:
                pos_rel, quat_world, quat_rel = np.zeros_like(filtered_pos_world), np.zeros((self.num_nodes, 4)), np.zeros((self.num_nodes, 4))
                quat_world[:, 3] = quat_rel[:, 3] = 1.0 

            last_pos_world = current_pos_world

            tracked_data = {"skeleton_0": {"pose": {
                "pos_world": current_pos_world, "pos_rel": pos_rel, "quat_world": quat_world, "quat_rel": quat_rel, "parents": self.parents
            }}}
            osc_send_tracked_data(tracked_data)

            if self.is_counting_down:
                remaining = 5 - int(time.time() - self.countdown_start_time)
                if remaining > 0:
                    cv2.putText(frame_rgb, f"RECORDING IN {remaining}...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                else:
                    self.is_counting_down = False
                    self.is_recording = True
                    self.recorded_frames = [] 
            elif self.is_recording:
                self.recorded_frames.append({'pos_rel': pos_rel, 'quat_rel': quat_rel})
                cv2.putText(frame_rgb, "RECORDING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(q_img)

        cap.release()
        if self.pose_landmarker: self.pose_landmarker.close()
        if self.hand_landmarker: self.hand_landmarker.close()

    def stop(self):
        self._run_flag = False
        self.wait()

    def start_recording(self):
        self.recorded_frames = []
        if self.args.source.isdigit():
            self.is_counting_down = True
            self.countdown_start_time = time.time()
        else:
            self.is_counting_down = False
            self.is_recording = True

    def stop_recording(self):
        self.is_counting_down = False
        self.is_recording = False
        if len(self.recorded_frames) > 0:
            filename = datetime.datetime.now().strftime("mocap_%Y%m%d_%H%M%S.fbx")
            export_fbx(filename, self.recorded_frames, self.parents)
            self.recorded_frames = []


class MainWindow(QMainWindow):
    def __init__(self, args, parents, num_nodes, slices, attachments):
        super().__init__()
        self.setWindowTitle("MediaPipe Pose Estimation")
        self.resize(800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0) 

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: black;") 
        self.layout.addWidget(self.image_label, stretch=1) 

        self.filter_layout = QHBoxLayout()
        self.filter_layout.setContentsMargins(10, 5, 10, 0)

        self.filter_layout.addWidget(QLabel("One-Euro Filter - Min Cutoff:"))
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.001, 5.0)
        self.cutoff_spin.setSingleStep(0.05)
        self.cutoff_spin.setDecimals(3)
        self.cutoff_spin.setValue(0.1)
        self.cutoff_spin.valueChanged.connect(self.on_filter_changed)
        self.filter_layout.addWidget(self.cutoff_spin)

        self.filter_layout.addWidget(QLabel("Beta (Speed):"))
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 50.0)
        self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setDecimals(2)
        self.beta_spin.setValue(5.0)
        self.beta_spin.valueChanged.connect(self.on_filter_changed)
        self.filter_layout.addWidget(self.beta_spin)

        self.filter_layout.addStretch(1)
        self.layout.addLayout(self.filter_layout)

        self.controls_layout = QHBoxLayout()
        self.controls_layout.setContentsMargins(10, 5, 10, 10) 

        self.btn_reset_calib = QPushButton("Reset Calibration")
        self.controls_layout.addWidget(self.btn_reset_calib)

        self.btn_start_record = QPushButton("Start Recording")
        self.btn_start_record.clicked.connect(self.start_recording)
        self.controls_layout.addWidget(self.btn_start_record)

        self.btn_stop_record = QPushButton("Stop Recording")
        self.btn_stop_record.clicked.connect(self.stop_recording)
        self.btn_stop_record.setEnabled(False)
        self.controls_layout.addWidget(self.btn_stop_record)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        self.controls_layout.addWidget(self.btn_exit)

        self.conf_label = QLabel("Tracking Conf: 0.70")
        self.controls_layout.addWidget(self.conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(70)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        self.controls_layout.addWidget(self.conf_slider)

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.controls_layout.addWidget(self.fps_label)

        self.layout.addLayout(self.controls_layout)

        self.thread = VideoThread(args, parents, num_nodes, slices, attachments)
        self.btn_reset_calib.clicked.connect(self.thread.reset_calibration)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_fps_signal.connect(self.update_fps)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    @pyqtSlot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def on_conf_changed(self, value):
        conf = value / 100.0
        self.conf_label.setText(f"Tracking Conf: {conf:.2f}")
        self.thread.set_tracking_confidence(conf)

    def on_filter_changed(self):
        cutoff = self.cutoff_spin.value()
        beta = self.beta_spin.value()
        self.thread.set_filter_params(cutoff, beta)

    def start_recording(self):
        self.thread.start_recording()
        self.btn_start_record.setEnabled(False)
        self.btn_stop_record.setEnabled(True)

    def stop_recording(self):
        self.thread.stop_recording()
        self.btn_start_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--hands", action="store_true")
    parser.add_argument("--model", type=str, choices=["lite", "full", "heavy"], default="heavy", help="Pose model type to use")
    args = parser.parse_args()
    if not (args.pose or args.hands): args.pose = args.hands = True

    download_models(args.model)
    PARENTS, NUM_NODES, SLICES, ATTACHMENTS = configure_topology(args.pose, args.hands)

    app = QApplication(sys.argv)
    window = MainWindow(args, PARENTS, NUM_NODES, SLICES, ATTACHMENTS)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()