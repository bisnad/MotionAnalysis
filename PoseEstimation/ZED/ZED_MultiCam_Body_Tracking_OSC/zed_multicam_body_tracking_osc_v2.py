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
import time
import ogl_viewer.viewer as gl
import numpy as np

import motion_sender

# joint map for body 34 to match joint numbers from live capture with those in an fbx recording
joint_map = [ 0, 1, 2, 11, 12, 13, 14, 15, 16, 17, 3, 26, 27, 28, 29, 30, 31, 4, 5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 32, 22, 23, 24, 25, 33 ]

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

if __name__ == "__main__":
        
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    if len(sys.argv) < 2:
        print("This sample display the fused body tracking of multiple cameras.")
        print("It needs a Localization file in input. Generate it with ZED 360.")
        print("The cameras can either be plugged to your devices, or already running on the local network.")
        exit(1)

    filepath = sys.argv[1]
    fusion_configurations = sl.read_fusion_configuration_file(filepath, sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER)
    if len(fusion_configurations) <= 0:
        print("Invalid file.")
        exit(1)

    senders = {}
    network_senders = {}

    # common parameters
    init_params = sl.InitParameters()
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_resolution = sl.RESOLUTION.HD1080

    communication_parameters = sl.CommunicationParameters()
    communication_parameters.set_for_shared_memory()

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True

    body_tracking_parameters = sl.BodyTrackingParameters()
    body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    #body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_18
    body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_34
    body_tracking_parameters.enable_body_fitting = True # False
    body_tracking_parameters.enable_tracking = True # False

    for conf in fusion_configurations:
        print("Try to open ZED", conf.serial_number)
        init_params.input = sl.InputType()
        # network cameras are already running, or so they should
        if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
            network_senders[conf.serial_number] = conf.serial_number

        # local camera needs to be run form here, in the same process than the fusion
        else:
            init_params.input = conf.input_type
            
            senders[conf.serial_number] = sl.Camera()

            init_params.set_from_serial_number(conf.serial_number)
            status = senders[conf.serial_number].open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print("Error opening the camera", conf.serial_number, status)
                del senders[conf.serial_number]
                continue

            status = senders[conf.serial_number].enable_positional_tracking(positional_tracking_parameters)
            if status != sl.ERROR_CODE.SUCCESS:
                print("Error enabling the positional tracking of camera", conf.serial_number)
                del senders[conf.serial_number]
                continue

            status = senders[conf.serial_number].enable_body_tracking(body_tracking_parameters)
            if status != sl.ERROR_CODE.SUCCESS:
                print("Error enabling the body tracking of camera", conf.serial_number)
                del senders[conf.serial_number]
                continue

            senders[conf.serial_number].start_publishing(communication_parameters)

        print("Camera", conf.serial_number, "is open")
    
    if len(senders) + len(network_senders) < 1:
        print("No enough cameras")
        exit(1)

    print("Senders started, running the fusion...")
        
    init_fusion_parameters = sl.InitFusionParameters()
    init_fusion_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion_parameters.coordinate_units = sl.UNIT.METER
    init_fusion_parameters.output_performance_metrics = False
    init_fusion_parameters.verbose = True
    communication_parameters = sl.CommunicationParameters()
    fusion = sl.Fusion()
    camera_identifiers = []

    fusion.init(init_fusion_parameters)
        
    print("Cameras in this configuration : ", len(fusion_configurations))

    # warmup
    bodies = sl.Bodies()        
    for serial in senders:
        zed = senders[serial]
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies)

    for i in range(0, len(fusion_configurations)):
        conf = fusion_configurations[i]
        uuid = sl.CameraIdentifier()
        uuid.serial_number = conf.serial_number
        print("Subscribing to", conf.serial_number, conf.communication_parameters.comm_type)

        status = fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
        if status != sl.FUSION_ERROR_CODE.SUCCESS:
            print("Unable to subscribe to", uuid.serial_number, status)
        else:
            camera_identifiers.append(uuid)
            print("Subscribed.")

    if len(camera_identifiers) <= 0:
        print("No camera connected.")
        exit(1)

    body_tracking_fusion_params = sl.BodyTrackingFusionParameters()
    body_tracking_fusion_params.enable_tracking = True
    body_tracking_fusion_params.enable_body_fitting = False
    
    fusion.enable_body_tracking(body_tracking_fusion_params)

    rt = sl.BodyTrackingFusionRuntimeParameters()
    rt.skeleton_minimum_allowed_keypoints = 7
    viewer = gl.GLViewer()
    viewer.init()
    
    """
    OSC Sender
    """
    
    motion_sender.config["ip"] = "127.0.0.1"
    motion_sender.config["port"] = 9007
    
    osc_sender = motion_sender.OscSender(motion_sender.config)

    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    single_bodies = [sl.Bodies]

    while (viewer.is_available()):
        for serial in senders:
            zed = senders[serial]
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_bodies(bodies)
        if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            
            # Retrieve detected objects
            fusion.retrieve_bodies(bodies, rt)
            # for debug, you can retrieve the data send by each camera, as well as communication and process stat just to make sure everything is okay
            # for cam in camera_identifiers:
            #     fusion.retrieveBodies(single_bodies, rt, cam); 
            viewer.update_bodies(bodies)
            
            # Update OSC
            update_osc(bodies, osc_sender)
            
    for sender in senders:
        senders[sender].close()
        
    viewer.exit()