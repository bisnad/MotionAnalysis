import sys
import time
import math
import os
import json
import threading
import numpy as np
import cv2
import sounddevice as sd
import soundfile as sf
import argparse
from scipy.spatial.transform import Rotation as R

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSizePolicy, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
import leap

# For rendering final mp4 with audio
import moviepy.editor as mp

# ==========================================
# CONSTANTS & TOPOLOGY
# ==========================================
NUM_NODES = 44
PARENTS = [-1] * NUM_NODES
HAND_PARENTS = [1, -1, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 1, 14, 15, 16, 1, 18, 19, 20]

for i in range(22):
    PARENTS[i] = HAND_PARENTS[i] # Left
    PARENTS[i + 22] = HAND_PARENTS[i] + 22 if HAND_PARENTS[i] != -1 else -1 # Right

HAND_CONNECTIONS = [(1, 0), (1, 2), (2, 3), (3, 4), (4, 5), (1, 6), (6, 7), (7, 8), (8, 9),
                    (1, 10), (10, 11), (11, 12), (12, 13), (1, 14), (14, 15), (15, 16), (16, 17),
                    (1, 18), (18, 19), (19, 20), (20, 21)]

VISUAL_CONNECTIONS = []
for u, v in HAND_CONNECTIONS:
    VISUAL_CONNECTIONS.append((u, v))
    VISUAL_CONNECTIONS.append((u + 22, v + 22))

# ==========================================
# MATH & KINEMATICS
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
        if np.linalg.norm(ortho) < 1e-6: ortho = np.cross(np.array([0.0, 1.0, 0.0]), v1)
        ortho = ortho / np.linalg.norm(ortho)
        return np.array([ortho[0], ortho[1], ortho[2], 0.0])
    cross = np.cross(v1, v2)
    q = np.array([cross[0], cross[1], cross[2], 1.0 + dot])
    return q / np.linalg.norm(q)

def compute_kinematics(pos_world, parents, offsets):
    num_nodes = len(pos_world)
    pos_rel = np.zeros_like(pos_world)
    quat_world = np.zeros((num_nodes, 4)); quat_world[:, 3] = 1.0
    quat_rel = np.zeros((num_nodes, 4)); quat_rel[:, 3] = 1.0

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
                    except: pass
                elif len(child_list) == 1:
                    quat_world[i] = get_shortest_quat(offsets[child_list[0]], pos_world[child_list[0]] - pos_world[i])
            elif len(child_list) == 1:
                quat_world[i] = get_shortest_quat(offsets[child_list[0]], pos_world[child_list[0]] - pos_world[i])
        else:
            if len(child_list) == 0:
                quat_world[i] = quat_world[p]
            else:
                c = child_list[0]
                try:
                    v_expected = R.from_quat(quat_world[p]).apply(offsets[c])
                    q_swing = get_shortest_quat(v_expected, pos_world[c] - pos_world[i])
                    quat_world[i] = (R.from_quat(q_swing) * R.from_quat(quat_world[p])).as_quat()
                except:
                    quat_world[i] = quat_world[p]

    for i in range(num_nodes):
        p = parents[i]
        if p != -1:
            try:
                quat_rel[i] = (R.from_quat(quat_world[p]).inv() * R.from_quat(quat_world[i])).as_quat()
            except: pass
            pos_rel[i] = offsets[i]
        else:
            quat_rel[i] = quat_world[i]
            pos_rel[i] = pos_world[i]
            
    return pos_rel, quat_world, quat_rel

def extract_hand_data(hand):
    joints = np.zeros((22, 3))
    joints[0] = [hand.palm.position.x, hand.palm.position.y, hand.palm.position.z]
    joints[1] = [hand.arm.next_joint.x, hand.arm.next_joint.y, hand.arm.next_joint.z]
    idx = 2
    for finger in [hand.thumb, hand.index, hand.middle, hand.ring, hand.pinky]:
        for bone in [finger.metacarpal, finger.proximal, finger.intermediate, finger.distal]:
            joints[idx] = [bone.next_joint.x, bone.next_joint.y, bone.next_joint.z]
            idx += 1
    return joints

# ==========================================
# FILTERS
# ==========================================
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007):
        self.min_cutoff = min_cutoff; self.beta = beta; self.d_cutoff = 1.0
        self.x_prev = x0; self.dx_prev = 0.0; self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return x
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat; self.dx_prev = dx_hat; self.t_prev = t
        return x_hat

# ==========================================
# PROCESSING THREAD
# ==========================================
class ProcessingThread(QThread):
    update_frame = pyqtSignal(np.ndarray, dict)

    def __init__(self):
        super().__init__()
        self.run_flag = True
        self.recording_mode = False
        
        # Audio / File Metadata
        self.audio_file = ""
        self.audio_data = None
        self.audio_fs = 44100
        self.audio_player = None
        self.recording_start_time = 0
        self.video_writer = None
        self.temp_video_path = ""
        
        # Mocap data storage
        self.mocap_data = {
            "/mocap/0/joint/pos_world_values": [], "/mocap/0/joint/pos_world_timestamps": [],
            "/mocap/0/joint/pos_local_values": [], "/mocap/0/joint/pos_local_timestamps": [],
            "/mocap/0/joint/rot_world_values": [], "/mocap/0/joint/rot_world_timestamps": [],
            "/mocap/0/joint/rot_local_values": [], "/mocap/0/joint/rot_local_timestamps": []
        }
        
        self.rest_offsets = np.zeros((NUM_NODES, 3))
        self.offsets_captured = np.zeros(NUM_NODES, dtype=bool)
        
        self.filter_min_cutoff = 0.1
        self.filter_beta = 5.0
        self.filters = None
        
        self.osc_sender = udp_client.SimpleUDPClient("127.0.0.1", 9007)
        self.listener_lock = threading.Lock()
        self.latest_event = None
        self.connection = leap.Connection()
        
        class LeapListener(leap.Listener):
            def __init__(self, thread_ref):
                super().__init__()
                self.thread_ref = thread_ref
            def on_tracking_event(self, event):
                with self.thread_ref.listener_lock:
                    self.thread_ref.latest_event = event
                    
        self.leap_listener = LeapListener(self)
        self.connection.add_listener(self.leap_listener)

    def play_audio(self, filepath):
        try:
            self.audio_data, self.audio_fs = sf.read(filepath)
            sd.play(self.audio_data, self.audio_fs)
        except Exception as e:
            print(f"Audio Play Error: {e}")

    def start_recording(self, filepath):
        print(f"Starting recording with audio: {filepath}")
        self.audio_file = filepath
        self.recording_mode = True
        
        for k in self.mocap_data:
            self.mocap_data[k] = []
            
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        os.makedirs("recordings", exist_ok=True)
        self.temp_video_path = f"recordings/temp_{base_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.temp_video_path, fourcc, 30.0, (640, 480))
        
        # Adding a minor delay before starting as requested
        time.sleep(0.5) 
        self.recording_start_time = time.perf_counter()
        self.play_audio(filepath)

    def finish_recording(self):
        self.recording_mode = False
        print("Finalizing Recording...")
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        # Write files
        base_name = os.path.splitext(os.path.basename(self.audio_file))[0]
        out_audio = f"recordings/{base_name}_audio.wav"
        out_npz = f"recordings/{base_name}_motion.npz"
        out_video = f"recordings/{base_name}_anim.mp4"
        out_meta = f"recordings/{base_name}_meta.json"
        
        # 1. Save Audio
        if self.audio_data is not None:
            sf.write(out_audio, self.audio_data, self.audio_fs)
            
        # 2. Save NPZ arrays
        npz_dict = {
            k: np.array(v) for k, v in self.mocap_data.items()
        }
        # Correctly renaming the redundant keys from the prompt requirements
        npz_dict["/mocap/0/joint/rot_world_timestamps"] = npz_dict.pop("/mocap/0/joint/rot_world_timestamps", np.array(self.mocap_data["/mocap/0/joint/rot_world_timestamps"]))
        np.savez_compressed(out_npz, **npz_dict)
        
        # 3. Save MP4 with Audio via MoviePy
        try:
            video_clip = mp.VideoFileClip(self.temp_video_path)
            audio_clip = mp.AudioFileClip(out_audio)
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(out_video, codec="libx264", audio_codec="aac", logger=None)
            video_clip.close()
            audio_clip.close()
            os.remove(self.temp_video_path)
        except Exception as e:
            print(f"Error compiling video/audio: {e}")
            
        # 4. Save Metadata
        frames = len(self.mocap_data["/mocap/0/joint/pos_world_timestamps"])
        meta = {
            "audio_wav": f"{base_name}_audio.wav",
            "motion_npz": f"{base_name}_motion.npz",
            "anim_mp4": f"{base_name}_anim.mp4",
            "fps": 30.0,
            "framecount": frames
        }
        with open(out_meta, 'w') as f:
            json.dump(meta, f, indent=4)
            
        print("Recording saved successfully!")

    def project_2d(self, pos):
        x = int(np.interp(pos[0], [-250, 250], [0, 640]))
        y = int(np.interp(pos[1], [50, 500], [480, 0]))
        return (x, y)

    def run(self):
        last_pos_world = np.zeros((NUM_NODES, 3))
        with self.connection.open():
            prev_time = time.perf_counter()
            
            while self.run_flag:
                now = time.perf_counter()
                dt = now - prev_time
                if dt < (1.0 / 30.0): # ~30fps for stable video rendering
                    time.sleep(0.001)
                    continue
                prev_time = now

                # Audio state check for auto-stop
                if self.recording_mode and self.audio_data is not None:
                    duration = len(self.audio_data) / self.audio_fs
                    if (now - self.recording_start_time) > duration:
                        self.finish_recording()

                with self.listener_lock:
                    event = self.latest_event
                    
                current_pos_world = np.copy(last_pos_world)
                active_nodes = np.zeros(NUM_NODES, dtype=bool)

                if event:
                    for hand in event.hands:
                        offset = 0 if "left" in str(hand.type).lower() else 22
                        joints = extract_hand_data(hand)
                        current_pos_world[offset:offset+22] = joints
                        active_nodes[offset:offset+22] = True

                # Filter
                if self.filters is None:
                    self.filters = [[OneEuroFilter(now, current_pos_world[i,j]) for j in range(3)] for i in range(NUM_NODES)]
                filtered_world = np.zeros_like(current_pos_world)
                for i in range(NUM_NODES):
                    for j in range(3):
                        self.filters[i][j].min_cutoff = self.filter_min_cutoff
                        self.filters[i][j].beta = self.filter_beta
                        filtered_world[i,j] = self.filters[i][j](now, current_pos_world[i,j])

                # Calibration
                if not np.all(self.offsets_captured[active_nodes]):
                    for i in range(NUM_NODES):
                        if active_nodes[i] and not self.offsets_captured[i]:
                            p = PARENTS[i]
                            if p == -1: self.rest_offsets[i] = current_pos_world[i].copy()
                            else: self.rest_offsets[i] = current_pos_world[i] - current_pos_world[p]
                            self.offsets_captured[i] = True

                # Kinematics
                if np.any(self.offsets_captured):
                    pos_rel, quat_world, quat_rel = compute_kinematics(filtered_world, PARENTS, self.rest_offsets)
                else:
                    pos_rel = np.zeros_like(filtered_world)
                    quat_world = np.zeros((NUM_NODES, 4)); quat_world[:, 3] = 1.0
                    quat_rel = np.zeros((NUM_NODES, 4)); quat_rel[:, 3] = 1.0

                last_pos_world = current_pos_world

                # Rendering
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                if self.recording_mode:
                    t_stamp = now - self.recording_start_time
                    self.mocap_data["/mocap/0/joint/pos_world_values"].append(filtered_world)
                    self.mocap_data["/mocap/0/joint/pos_world_timestamps"].append(t_stamp)
                    self.mocap_data["/mocap/0/joint/pos_local_values"].append(pos_rel)
                    self.mocap_data["/mocap/0/joint/pos_local_timestamps"].append(t_stamp)
                    self.mocap_data["/mocap/0/joint/rot_world_values"].append(quat_world)
                    self.mocap_data["/mocap/0/joint/rot_world_timestamps"].append(t_stamp)
                    self.mocap_data["/mocap/0/joint/rot_local_values"].append(quat_rel)
                    self.mocap_data["/mocap/0/joint/rot_local_timestamps"].append(t_stamp)
                    
                    cv2.putText(img, "RECORDING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                for u, v in VISUAL_CONNECTIONS:
                    if active_nodes[u] and active_nodes[v]:
                        cv2.line(img, self.project_2d(filtered_world[u]), self.project_2d(filtered_world[v]), (0, 255, 0), 2)
                for i in range(NUM_NODES):
                    if active_nodes[i]:
                        cv2.circle(img, self.project_2d(filtered_world[i]), 4, (255, 255, 255), -1)

                if self.video_writer and self.recording_mode:
                    self.video_writer.write(img)

                stats = {"proc_fps": 1.0 / max(dt, 1e-6)}
                self.update_frame.emit(img, stats)

    def stop(self):
        self.run_flag = False
        self.connection.remove_listener(self.leap_listener)
        if self.recording_mode:
            self.finish_recording()
        self.wait()

# ==========================================
# GUI / APP
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leap Motion - OSC Audio Recorder")
        self.resize(640, 560)
        
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.layout = QVBoxLayout(self.central)
        
        self.image_label = QLabel()
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label, stretch=1)
        
        self.fps_label = QLabel("FPS: 0.0")
        self.layout.addWidget(self.fps_label)
        
        self.thread = ProcessingThread()
        self.thread.update_frame.connect(self.refresh_ui)
        self.thread.start()

        # Start OSC Server
        self.osc_dispatcher = Dispatcher()
        self.osc_dispatcher.map("/play", self.osc_play)
        self.osc_dispatcher.map("/record", self.osc_record)
        
        self.osc_server = ThreadingOSCUDPServer(("0.0.0.0", 8000), self.osc_dispatcher)
        self.osc_thread = threading.Thread(target=self.osc_server.serve_forever, daemon=True)
        self.osc_thread.start()
        print("OSC Server listening on port 8000. Send /play <file> or /record <file>")

    def osc_play(self, address, filepath):
        print(f"Received /play command: {filepath}")
        self.thread.play_audio(filepath)

    def osc_record(self, address, filepath):
        print(f"Received /record command: {filepath}")
        self.thread.start_recording(filepath)

    @pyqtSlot(np.ndarray, dict)
    def refresh_ui(self, img_array, stats):
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.image_label.setPixmap(pixmap)
        self.fps_label.setText(f"FPS: {stats['proc_fps']:.1f}")

    def closeEvent(self, event):
        self.osc_server.shutdown()
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())