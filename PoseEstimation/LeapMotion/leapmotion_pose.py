import sys
import time
import math
import datetime
import threading
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSizePolicy, QDoubleSpinBox, QLineEdit, QSpinBox)
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

import leap
from leap.events import TrackingEvent
from pythonosc.udp_client import SimpleUDPClient

try:
    import fbx
except ImportError:
    print("WARNING: Autodesk fbx module not found. FBX export will not work. Install with pip install fbx.")

# ==========================================
# CONFIGURATION & TOPOLOGY
# ==========================================
DEFAULT_OSC_IP = "127.0.0.1"
DEFAULT_OSC_PORT = 9007

NUM_NODES = 44
PARENTS = [-1] * NUM_NODES

# 0: Palm, 1: Wrist
# 2-5: Thumb, 6-9: Index, 10-13: Middle, 14-17: Ring, 18-21: Pinky
HAND_PARENTS = [
    1, -1, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 1, 14, 15, 16, 1, 18, 19, 20
]

for i in range(22):
    PARENTS[i] = HAND_PARENTS[i] # Left Hand
    PARENTS[i + 22] = HAND_PARENTS[i] + 22 if HAND_PARENTS[i] != -1 else -1 # Right Hand

HAND_CONNECTIONS = [
    (1, 0),
    (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9),
    (1, 10), (10, 11), (11, 12), (12, 13),
    (1, 14), (14, 15), (15, 16), (16, 17),
    (1, 18), (18, 19), (19, 20), (20, 21)
]

VISUAL_CONNECTIONS = []
for u, v in HAND_CONNECTIONS:
    VISUAL_CONNECTIONS.append((u, v))
    VISUAL_CONNECTIONS.append((u + 22, v + 22))

# ==========================================
# OSC SENDER
# ==========================================
class OscSender:
    def __init__(self, ip=DEFAULT_OSC_IP, port=DEFAULT_OSC_PORT):
        self.oscsender = SimpleUDPClient(ip, port)
        
    def send(self, address, values):
        osc_values = np.reshape(values, -1).tolist()
        self.oscsender.send_message(address, osc_values)

# ==========================================
# FILTERS
# ==========================================
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

# ==========================================
# KINEMATICS & MATH
# ==========================================
def get_shortest_quat(v1, v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    v1, v2 = v1 / v1_norm, v2 / v2_norm
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    if dot > 0.999999: return np.array([0.0, 0.0, 0.0, 1.0])
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
        if p >= 0: children[p].append(i)

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
                    except:
                        quat_world[i] = np.array([0.0, 0.0, 0.0, 1.0])
                elif len(child_list) == 1:
                    v_rest = offsets[child_list[0]]
                    v_curr = pos_world[child_list[0]] - pos_world[i]
                    quat_world[i] = get_shortest_quat(v_rest, v_curr)
            elif len(child_list) == 1:
                v_rest = offsets[child_list[0]]
                v_curr = pos_world[child_list[0]] - pos_world[i]
                quat_world[i] = get_shortest_quat(v_rest, v_curr)
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
                except:
                    quat_world[i] = quat_world[p]

    for i in range(num_nodes):
        p = parents[i]
        if p != -1:
            try:
                r_p_inv = R.from_quat(quat_world[p]).inv()
                quat_rel[i] = (r_p_inv * R.from_quat(quat_world[i])).as_quat()
            except:
                quat_rel[i] = np.array([0.0, 0.0, 0.0, 1.0])
            pos_rel[i] = offsets[i]
        else:
            quat_rel[i] = quat_world[i]
            pos_rel[i] = pos_world[i]
            
    return pos_rel, quat_world, quat_rel

def extract_hand_data(hand):
    joints = np.zeros((22, 3))
    joints[0] = [hand.palm.position.x, hand.palm.position.y, hand.palm.position.z]
    joints[1] = [hand.arm.next_joint.x, hand.arm.next_joint.y, hand.arm.next_joint.z]
    
    fingers = [hand.thumb, hand.index, hand.middle, hand.ring, hand.pinky]
    idx = 2
    for finger in fingers:
        for bone in [finger.metacarpal, finger.proximal, finger.intermediate, finger.distal]:
            joints[idx] = [bone.next_joint.x, bone.next_joint.y, bone.next_joint.z]
            idx += 1
    return joints

# ==========================================
# FBX EXPORTER
# ==========================================
def export_fbx(filename, frames_data, parents, fps=115.0):
    if 'fbx' not in globals(): return
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    scene = fbx.FbxScene.Create(manager, "MocapScene")
    num_joints = len(parents)
    nodes = []

    try: skel_root_enum = fbx.FbxSkeleton.EType.eRoot; skel_limb_enum = fbx.FbxSkeleton.EType.eLimbNode
    except AttributeError: skel_root_enum = fbx.FbxSkeleton.eRoot; skel_limb_enum = fbx.FbxSkeleton.eLimbNode
    
    try: interp_linear_enum = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
    except AttributeError:
        try: interp_linear_enum = fbx.FbxAnimCurveDef.eInterpolationLinear
        except AttributeError: interp_linear_enum = 4

    for i in range(num_joints):
        node = fbx.FbxNode.Create(scene, f"Joint_{i}")
        skeleton = fbx.FbxSkeleton.Create(scene, f"Skel_{i}")
        skeleton.SetSkeletonType(skel_root_enum if parents[i] == -1 else skel_limb_enum)
        node.SetNodeAttribute(skeleton)
        
        # Determine Local Translation offsets based on the first recorded frame
        if len(frames_data) > 0:
            tx, ty, tz = frames_data[0][0][i] / 10.0 # scale mm to cm for FBX
            node.LclTranslation.Set(fbx.FbxDouble3(float(tx), float(ty), float(tz)))
        nodes.append(node)

    for i in range(num_joints):
        if parents[i] != -1 and nodes[parents[i]] is not None:
            nodes[parents[i]].AddChild(nodes[i])
        else:
            scene.GetRootNode().AddChild(nodes[i])

    anim_stack = fbx.FbxAnimStack.Create(scene, "MotionStack")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "MotionLayer")
    anim_stack.AddMember(anim_layer)

    curves = []
    for i in range(num_joints):
        n = nodes[i]
        curves_i = []
        for prop in (n.LclTranslation, n.LclRotation):
            for axis in ("X", "Y", "Z"):
                c = prop.GetCurve(anim_layer, axis, True)
                c.KeyModifyBegin()
                curves_i.append(c)
        curves.append(curves_i)

    fbx_time = fbx.FbxTime()
    for frame_idx, frame in enumerate(frames_data):
        fbx_time.SetSecondDouble(frame_idx / fps)
        pos_rel, quat_rel = frame
        
        for i in range(num_joints):
            tx, ty, tz = pos_rel[i] / 10.0
            rx, ry, rz = R.from_quat(quat_rel[i]).as_euler('xyz', degrees=True)
            
            for curve, val in zip(curves[i], (tx, ty, tz, rx, ry, rz)):
                k_idx = curve.KeyAdd(fbx_time)[0]
                curve.KeySet(k_idx, fbx_time, float(val), interp_linear_enum)

    for i in range(num_joints):
        for c in curves[i]: c.KeyModifyEnd()

    # Apply unroll filter to prevent FBX gimbal lock
    unroll_filter = fbx.FbxAnimCurveFilterUnroll()
    for i in range(num_joints):
        rot_curve_node = nodes[i].LclRotation.GetCurveNode(anim_layer, False)
        if rot_curve_node:
            try: unroll_filter.Apply(rot_curve_node)
            except Exception:
                status = fbx.FbxStatus()
                unroll_filter.Apply(rot_curve_node, status)

    exporter = fbx.FbxExporter.Create(manager, "")
    if exporter.Initialize(filename, -1, manager.GetIOSettings()):
        exporter.Export(scene)
        print(f"--- Successfully exported FBX to {filename}")

# ==========================================
# LEAP LISTENER
# ==========================================
class LeapListener(leap.Listener):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.latest_event = None
        
    def on_tracking_event(self, event: TrackingEvent):
        with self.lock:
            self.latest_event = event

# ==========================================
# PROCESSING THREAD
# ==========================================
class ProcessingThread(QThread):
    update_frame = pyqtSignal(np.ndarray, dict)

    def __init__(self):
        super().__init__()
        self.run_flag = True
        self.is_recording = False
        self.recorded_frames = []
        self.rest_offsets = np.zeros((NUM_NODES, 3))
        self.offsets_captured = np.zeros(NUM_NODES, dtype=bool)
        
        self.pose_filter = None
        self.filter_min_cutoff = 0.1
        self.filter_beta = 5.0
        
        self.osc_sender = OscSender()
        self.listener = LeapListener()
        self.connection = leap.Connection()
        self.connection.add_listener(self.listener)
        
        self.proc_fps = 0.0
        self.last_pos_world = np.zeros((NUM_NODES, 3))

    def update_osc_settings(self, ip, port):
        self.osc_sender = OscSender(ip, port)

    def set_filter_params(self, min_cutoff, beta):
        self.filter_min_cutoff = min_cutoff
        self.filter_beta = beta
        if self.pose_filter:
            self.pose_filter.update_params(min_cutoff, beta)

    def reset_calibration(self):
        self.offsets_captured.fill(False)
        print("Rest pose calibration reset! Please hold flat hands.")

    def start_recording(self):
        self.recorded_frames = []
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False
        if len(self.recorded_frames) > 0:
            filename = datetime.datetime.now().strftime("leap_mocap_%Y%m%d_%H%M%S.fbx")
            export_fbx(filename, self.recorded_frames, PARENTS, fps=115.0)

    def project_3d_to_2d(self, pos, w, h):
        # Map Leap coords (-250 to 250 mm x/z, 50 to 500 mm y) into a 2D image plane
        x = int(np.interp(pos[0], [-250, 250], [0, w]))
        y = int(np.interp(pos[1], [50, 500], [h, 0]))  # OpenCV Y is flipped relative to Leap Up
        return (x, y)

    def run(self):
        # Connection must be wrapped as a context manager for tracking to pump properly
        with self.connection.open():
            prev_time = time.perf_counter()
            
            while self.run_flag:
                now = time.perf_counter()
                dt = now - prev_time
                if dt < (1.0 / 60.0): # limit drawing loop to ~60 FPS
                    time.sleep(0.001)
                    continue
                    
                self.proc_fps = 0.9 * self.proc_fps + 0.1 * (1.0 / dt)
                prev_time = now

                with self.listener.lock:
                    event = self.listener.latest_event
                    
                current_pos_world = np.copy(self.last_pos_world)
                active_nodes = np.zeros(NUM_NODES, dtype=bool)

                if event:
                    for hand in event.hands:
                        # Ensures correct index offsets dynamically
                        is_left = "left" in str(hand.type).lower()
                        offset = 0 if is_left else 22
                        joints = extract_hand_data(hand)
                        current_pos_world[offset:offset+22] = joints
                        active_nodes[offset:offset+22] = True

                # Filtering
                if self.pose_filter is None:
                    self.pose_filter = PoseFilter(NUM_NODES, min_cutoff=self.filter_min_cutoff, beta=self.filter_beta)
                filtered_pos_world = self.pose_filter(now, current_pos_world)

                # Calibration/Rest Capture
                if not np.all(self.offsets_captured[active_nodes]):
                    for i in range(NUM_NODES):
                        if active_nodes[i] and not self.offsets_captured[i]:
                            p = PARENTS[i]
                            if p == -1: 
                                self.rest_offsets[i] = current_pos_world[i].copy()
                            else: 
                                self.rest_offsets[i] = current_pos_world[i] - current_pos_world[p]
                            self.offsets_captured[i] = True

                # Kinematics Math
                if np.any(self.offsets_captured):
                    pos_rel, quat_world, quat_rel = compute_kinematics(filtered_pos_world, PARENTS, self.rest_offsets)
                else:
                    pos_rel = np.zeros_like(filtered_pos_world)
                    quat_world = np.zeros((NUM_NODES, 4)); quat_world[:, 3] = 1.0
                    quat_rel = np.zeros((NUM_NODES, 4)); quat_rel[:, 3] = 1.0

                self.last_pos_world = current_pos_world

                # OSC Sender
                if np.any(active_nodes):
                    self.osc_sender.send("/mocap/0/joint/pos_world", current_pos_world)
                    self.osc_sender.send("/mocap/0/joint/pos_local", pos_rel)
                    self.osc_sender.send("/mocap/0/joint/rot_world", quat_world)
                    self.osc_sender.send("/mocap/0/joint/rot_local", quat_rel)

                # Recording
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                if self.is_recording:
                    self.recorded_frames.append((pos_rel, quat_rel))
                    cv2.putText(img, "RECORDING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                # Visual Calibration Alert
                if not np.all(self.offsets_captured):
                    cv2.putText(img, "UNTRACKED JOINTS. HOLD BOTH HANDS IN VIEW", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw Visuals based on Active Nodes
                for u, v in VISUAL_CONNECTIONS:
                    if active_nodes[u] and active_nodes[v]:
                        pt1 = self.project_3d_to_2d(filtered_pos_world[u], 640, 480)
                        pt2 = self.project_3d_to_2d(filtered_pos_world[v], 640, 480)
                        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                for i in range(NUM_NODES):
                    if active_nodes[i]:
                        pt = self.project_3d_to_2d(filtered_pos_world[i], 640, 480)
                        cv2.circle(img, pt, 4, (255, 255, 255), -1)

                stats = {"proc_fps": self.proc_fps}
                self.update_frame.emit(img, stats)

    def stop(self):
        self.run_flag = False
        self.wait()

# ==========================================
# GUI / MAIN WINDOW
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leap Motion - Kinematics & FBX")
        self.resize(800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.image_label, stretch=1)
        
        self.filter_layout = QHBoxLayout()
        self.filter_layout.addWidget(QLabel("1-Euro Min Cutoff:"))
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.001, 5.0); self.cutoff_spin.setSingleStep(0.05)
        self.cutoff_spin.setValue(0.1); self.cutoff_spin.valueChanged.connect(self.on_filter_changed)
        self.filter_layout.addWidget(self.cutoff_spin)
        
        self.filter_layout.addWidget(QLabel("Beta:"))
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 50.0); self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setValue(5.0); self.beta_spin.valueChanged.connect(self.on_filter_changed)
        self.filter_layout.addWidget(self.beta_spin)
        self.layout.addLayout(self.filter_layout)
        
        self.osc_layout = QHBoxLayout()
        self.osc_layout.addWidget(QLabel("OSC IP:"))
        self.ip_input = QLineEdit(DEFAULT_OSC_IP)
        self.osc_layout.addWidget(self.ip_input)
        self.osc_layout.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535); self.port_spin.setValue(DEFAULT_OSC_PORT)
        self.osc_layout.addWidget(self.port_spin)
        self.btn_apply_osc = QPushButton("Apply OSC")
        self.btn_apply_osc.clicked.connect(self.on_apply_osc)
        self.osc_layout.addWidget(self.btn_apply_osc)
        self.layout.addLayout(self.osc_layout)
        
        self.controls_layout = QHBoxLayout()
        self.btn_reset_calib = QPushButton("Reset Calibration")
        self.controls_layout.addWidget(self.btn_reset_calib)
        self.btn_start_record = QPushButton("Start Recording")
        self.btn_start_record.clicked.connect(self.start_recording)
        self.controls_layout.addWidget(self.btn_start_record)
        self.btn_stop_record = QPushButton("Stop Recording")
        self.btn_stop_record.clicked.connect(self.stop_recording)
        self.btn_stop_record.setEnabled(False)
        self.controls_layout.addWidget(self.btn_stop_record)
        
        self.fps_label = QLabel("FPS: 0.0")
        self.controls_layout.addWidget(self.fps_label)
        self.layout.addLayout(self.controls_layout)
        
        self.thread = ProcessingThread()
        self.thread.update_frame.connect(self.refresh_ui)
        self.btn_reset_calib.clicked.connect(self.thread.reset_calibration)
        
        # Retain image array in memory for PyQT garbage collection safely
        self.current_image = None
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray, dict)
    def refresh_ui(self, img_array, stats):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        self.current_image = img_array.copy()
        
        qimg = QImage(self.current_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.image_label.setPixmap(pixmap)
        self.fps_label.setText(f"FPS: {stats['proc_fps']:.1f}")

    def on_filter_changed(self):
        self.thread.set_filter_params(self.cutoff_spin.value(), self.beta_spin.value())

    def on_apply_osc(self):
        self.thread.update_osc_settings(self.ip_input.text().strip(), self.port_spin.value())

    def start_recording(self):
        self.thread.start_recording()
        self.btn_start_record.setEnabled(False)
        self.btn_stop_record.setEnabled(True)

    def stop_recording(self):
        self.thread.stop_recording()
        self.btn_start_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())