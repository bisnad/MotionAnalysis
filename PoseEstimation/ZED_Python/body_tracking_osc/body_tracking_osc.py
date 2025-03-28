########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse

import motion_sender

# joint map for boddy 34 to match joint numbers from live capture with those in an fbx recording
joint_map = [ 0, 1, 2, 11, 12, 13, 14, 15, 16, 17, 3, 26, 27, 28, 29, 30, 31, 4, 5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 32, 22, 23, 24, 25, 33 ]
# joint map for body 38 to match joint numbers from live capture with those in an fbx recording
# joint_map = [ 0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 12, 14, 16, 30, 32, 34, 36, 11, 13, 15, 17, 31, 33, 35, 37, 18, 20, 22, 24, 26, 28, 19, 21, 23, 25, 27, 29 ]


def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")


def update_osc(_bodies, sender):

    for _body in _bodies.body_list:
        
        #print("_body t ", type(_body))
        
        body_id = _body.id
        joint_pos2d_world = _body.keypoint_2d
        joint_pos3d_world = _body.keypoint
        joint_rot_local = _body.local_orientation_per_joint
        joint_pos_local = _body.local_position_per_joint
        root_rot_world = _body.global_root_orientation

        joint_count = joint_pos2d_world.shape[0]
        if joint_count == 18: # Body18
            root_joint_index = 1
        else: # Body34 or Body38 
            root_joint_index = 0
            
        root_pos_world = joint_pos3d_world[root_joint_index]
        
        """
        print("id ", body_id)
        print("pos2d_world s ", joint_pos2d_world.shape)
        print("joint_pos3d_world s ", joint_pos3d_world.shape)
        if joint_rot_local is not None:
            print("joint_rot_local s ", joint_rot_local.shape)
        if joint_pos_local is not None:
            print("joint_pos_local s ", joint_pos_local.shape)        
        if root_rot_world is not None:
            print("root_rot_world s ", root_rot_world.shape)  
        if root_pos_world is not None:
            print("root_pos_world s ", root_pos_world.shape)  
        """


        joint_pos2d_world_osc = joint_pos2d_world[joint_map, :]
        joint_pos3d_world_osc = joint_pos3d_world[joint_map, :]
        joint_rot_local_osc = joint_rot_local[joint_map, :]
        joint_pos_local_osc = joint_pos_local[joint_map, :]


        # quat conversion from x y z w to w x y z
        joint_rot_local_osc = np.roll(joint_rot_local_osc, 1, axis=1)
        root_rot_world_osc = np.roll(root_rot_world, 1, axis=0)

        #sender.send(f"/mocap/{body_id}/joint/pos2d_world", joint_pos2d_world)
        #sender.send(f"/mocap/{body_id}/joint/pos_world", joint_pos3d_world)
        #sender.send(f"/mocap/{body_id}/joint/rot_local", joint_rot_local)
        #sender.send(f"/mocap/{body_id}/joint/pos_local", joint_pos_local)
        
        sender.send(f"/mocap/{body_id}/joint/pos2d_world", joint_pos2d_world_osc)
        sender.send(f"/mocap/{body_id}/joint/pos_world", joint_pos3d_world_osc)
        sender.send(f"/mocap/{body_id}/joint/rot_local", joint_rot_local_osc)
        sender.send(f"/mocap/{body_id}/joint/pos_local", joint_pos_local_osc)
        
        sender.send(f"/mocap/{body_id}/joint/root_rot_world", root_rot_world_osc)
        sender.send(f"/mocap/{body_id}/joint/root_pos_world", root_pos_world)
        
def main():
    
    """
    OSC Sender
    """
    
    motion_sender.config["ip"] = "127.0.0.1"
    motion_sender.config["port"] = 9007
    
    osc_sender = motion_sender.OscSender(motion_sender.config)
    
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = True            # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
    #body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM 
    #body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE 

    if len(joint_map) == 34:
        body_param.body_format = sl.BODY_FORMAT.BODY_34  # Choose the BODY_FORMAT you wish to use
    elif len(joint_map) == 38:
        body_param.body_format = sl.BODY_FORMAT.BODY_38  # Choose the BODY_FORMAT you wish to use
    else:
        print("number of joints not supported")

    #body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use
    #body_param.body_format = sl.BODY_FORMAT.BODY_34  # Choose the BODY_FORMAT you wish to use
    #body_param.body_format = sl.BODY_FORMAT.BODY_38  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40
    #body_runtime_param.skeleton_smoothing = 0

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)
    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    image = sl.Mat()
    
    key_wait = 10 
    while viewer.is_available():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)
            # Update OSC
            update_osc(bodies, osc_sender)
            # Update GL view
            viewer.update_view(image, bodies) 
            # Update OCV view
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            key = cv2.waitKey(key_wait)
            if key == 113: # for 'q' key
                print("Exiting...")
                break
            if key == 109: # for 'm' key
                if (key_wait>0):
                    print("Pause")
                    key_wait = 0 
                else : 
                    print("Restart")
                    key_wait = 10 
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 