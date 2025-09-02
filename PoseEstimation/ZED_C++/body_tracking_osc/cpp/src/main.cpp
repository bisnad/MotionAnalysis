///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/*****************************************************************************************
 ** This sample demonstrates how to detect human bodies and retrieves their 3D position **
 **         with the ZED SDK and display the result in an OpenGL window.                **
 *****************************************************************************************/

 // OSC include
#pragma comment(lib, "ws2_32.lib")
#include <array>
#include <string>

#include "osc/OscOutboundPacketStream.h"
#include "ip/UdpSocket.h"

std::string osc_send_address = "127.0.0.1";
int osc_send_port = 9007;

// joint map for body 34 to match joint numbers from live capture with those in an fbx recording
std::array<int, 34> joint_map = { 0, 1, 2, 11, 12, 13, 14, 15, 16, 17, 3, 26, 27, 28, 29, 30, 31, 4, 5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 32, 22, 23, 24, 25, 33 };
// joint map for body 38 to match joint numbers from live capture with those in an fbx recording
// std::array<int, 38> joint_map = { 0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 12, 14, 16, 30, 32, 34, 36, 11, 13, 15, 17, 31, 33, 35, 37, 18, 20, 22, 24, 26, 28, 19, 21, 23, 25, 27, 29 };

// joint parent indices for body 34
std::array<int, 34> joint_parent_indices = { -1, 0, 1, 2, 3, 4, 5, 6, 7, 6, 2, 10, 11, 11, 11, 11, 11, 2, 17, 18, 19, 20, 21, 20, 0, 24, 25, 26, 26, 0, 29, 30, 31, 31 };
// joint parent indices for body 38
//std::array<int, 38> joint_parent_indices = { -1, 0, 1, 2, 3, 4, 5, 6, 5, 8, 3, 10, 11, 12, 13, 13, 13, 13, 3, 18, 19, 20, 21, 21, 21, 21, 0, 26, 27, 28, 28, 28, 0, 32, 33, 34, 34, 34 };

float joint_pos_scale = 0.1;

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"
#include "TrackingViewer.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
void parseArgs(int argc, char **argv, InitParameters& param);

// Quaternion multiplication helper (w,x,y,z order)
sl::float4 quatMul(const sl::float4& q1, const sl::float4& q2) {
    return sl::float4(
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3], // w
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2], // x
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1], // y
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]  // z
    );
}

void update_osc(sl::Bodies& bodies, osc::UdpTransmitSocket& pSocket, osc::OutboundPacketStream& pStream)
{
    //std::cout << "update_osc begin\n";

    //std::cout << "body count " << bodies.body_list.size() << "\n";

    for (auto& _body : bodies.body_list)
    {
        int body_id = _body.id;

        //std::cout << "body_id " << body_id << "\n";

        std::vector<sl::float2> joint_pos2d_world = _body.keypoint_2d;
        std::vector<sl::float3> joint_pos3d_world = _body.keypoint;
        std::vector<sl::float4> joint_rot_local = _body.local_orientation_per_joint;
        std::vector<sl::float3> joint_pos_local = _body.local_position_per_joint;
        sl::float4 root_rot_world = _body.global_root_orientation;

        /*
        std::cout << "joint_pos2d_world size " << joint_pos2d_world.size() << "\n";
        std::cout << "joint_pos3d_world size " << joint_pos3d_world.size() << "\n";
        std::cout << "joint_rot_local size " << joint_rot_local.size() << "\n";
        std::cout << "joint_pos_local size " << joint_pos_local.size() << "\n";
        */

        if (joint_pos3d_world.size() != joint_rot_local.size())
            continue;

        int joint_count = joint_pos2d_world.size();
        int root_joint_index;

        if (joint_count == 18) // Body18
        {
            root_joint_index = 1;
        }
        else // Body34 or Body38
        {
            root_joint_index = 0;
        }

        sl::float3 root_pos_world = joint_pos3d_world[root_joint_index];

        std::vector<sl::float2> joint_pos2d_world_osc(joint_count);
        std::vector<sl::float3> joint_pos3d_world_osc(joint_count);
        std::vector<sl::float4> joint_rot_local_osc(joint_count);
        std::vector<sl::float3> joint_pos_local_osc(joint_count);
        std::vector<sl::float4> joint_rot_world_osc(joint_count);

        for (int jI = 0; jI < joint_count; ++jI)
        {
            // joint remap

            joint_pos2d_world_osc[jI] = joint_pos2d_world[joint_map[jI]];

            //std::cout << "jI " << jI << " pos2d orig " << joint_pos2d_world[jI] << " mapped " << joint_pos2d_world[joint_map[jI]] << "\n";

            joint_pos3d_world_osc[jI] = joint_pos3d_world[joint_map[jI]];
            joint_pos3d_world_osc[jI] *= joint_pos_scale;
            joint_rot_local_osc[jI] = joint_rot_local[joint_map[jI]];
            joint_pos_local_osc[jI] = joint_pos_local[joint_map[jI]];
            joint_pos_local_osc[jI] *= joint_pos_scale;

            // quat conversion from x y z w to w x y z
            sl::float4 rot_xyzw = joint_rot_local_osc[jI];
            joint_rot_local_osc[jI] = sl::float4(rot_xyzw[3], rot_xyzw[0], rot_xyzw[1], rot_xyzw[2]);
            //joint_rot_local_osc[jI] = sl::float4(rot_xyzw[0], rot_xyzw[1], rot_xyzw[2], rot_xyzw[3]);
        }

        // quat conversion from x y z w to w x y z
        sl::float4 rot_xyzw = root_rot_world;
        sl::float4 root_rot_world_osc = sl::float4(rot_xyzw[3], rot_xyzw[0], rot_xyzw[1], rot_xyzw[2]);
        //sl::float4 root_rot_world_osc = sl::float4(rot_xyzw[0], rot_xyzw[1], rot_xyzw[2], rot_xyzw[3]);

        // ---- Compute GLOBAL ROTATIONS ----
        auto parents = joint_parent_indices;

        for (int jI = 0; jI < joint_count; ++jI) {
            int parent = parents[jI];
            if (parent == -1) {
                // Root joint gets global_root_orientation
                joint_rot_world_osc[jI] = root_rot_world_osc;
            }
            else {
                // Combine parent world rotation with this joint's local rotation
                joint_rot_world_osc[jI] = quatMul(joint_rot_world_osc[parent], joint_rot_local_osc[jI]);
            }
        }

        // send pos2d_world
        pStream.Clear();
        pStream << osc::BeginMessage((std::string("/mocap/") + std::to_string(body_id) + "/joint/pos2d_world").c_str());
        for (int jI = 0; jI < joint_count; ++jI) pStream << joint_pos2d_world_osc[jI][0] << joint_pos2d_world_osc[jI][1];
        pStream << osc::EndMessage;
        pSocket.Send(pStream.Data(), pStream.Size());

        // send pos_world
        pStream.Clear();
        pStream << osc::BeginMessage((std::string("/mocap/") + std::to_string(body_id) + "/joint/pos_world").c_str());
        for (int jI = 0; jI < joint_count; ++jI) pStream << joint_pos3d_world_osc[jI][0] << joint_pos3d_world_osc[jI][1] << joint_pos3d_world_osc[jI][2];
        pStream << osc::EndMessage;
        pSocket.Send(pStream.Data(), pStream.Size());

        // send rot_local
        pStream.Clear();
        pStream << osc::BeginMessage((std::string("/mocap/") + std::to_string(body_id) + "/joint/rot_local").c_str());
        for (int jI = 0; jI < joint_count; ++jI) pStream << joint_rot_local_osc[jI][0] << joint_rot_local_osc[jI][1] << joint_rot_local_osc[jI][2] << joint_rot_local_osc[jI][3];
        pStream << osc::EndMessage;
        pSocket.Send(pStream.Data(), pStream.Size());

        // send rot_world
        pStream.Clear();
        pStream << osc::BeginMessage(("/mocap/" + std::to_string(body_id) + "/joint/rot_world").c_str());
        for (int jI = 0; jI < joint_count; ++jI) pStream << joint_rot_world_osc[jI][0] << joint_rot_world_osc[jI][1] << joint_rot_world_osc[jI][2] << joint_rot_world_osc[jI][3];
        pStream << osc::EndMessage;
        pSocket.Send(pStream.Data(), pStream.Size());

        // send pos_local
        pStream.Clear();
        pStream << osc::BeginMessage((std::string("/mocap/") + std::to_string(body_id) + "/joint/pos_local").c_str());
        for (int jI = 0; jI < joint_count; ++jI) pStream << joint_pos_local_osc[jI][0] << joint_pos_local_osc[jI][1] << joint_pos_local_osc[jI][2];
        pStream << osc::EndMessage;
        pSocket.Send(pStream.Data(), pStream.Size());

        // send root_rot_world
        pStream.Clear();
        pStream << osc::BeginMessage((std::string("/mocap/") + std::to_string(body_id) + "/joint/root_rot_world").c_str());
        pStream << root_rot_world_osc[0] << root_rot_world_osc[1] << root_rot_world_osc[2] << root_rot_world_osc[3];
        pStream << osc::EndMessage;
        pSocket.Send(pStream.Data(), pStream.Size());

        // send root_pos_world
        pStream.Clear();
        pStream << osc::BeginMessage((std::string("/mocap/") + std::to_string(body_id) + "/joint/root_pos_world").c_str());
        pStream << root_pos_world[0] << root_pos_world[1] << root_pos_world[2];
        pStream << osc::EndMessage;
        pSocket.Send(pStream.Data(), pStream.Size());
    }


    //std::cout << "update_osc end\n";
}

int main(int argc, char **argv) {

#ifdef _SL_JETSON_
    const bool isJetson = true;
#else
    const bool isJetson = false;
#endif

    // Create ZED Bodies
    Camera zed;
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::AUTO;
    init_parameters.depth_mode = isJetson ? DEPTH_MODE::PERFORMANCE : DEPTH_MODE::ULTRA;
    //init_parameters.depth_mode = DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;

    parseArgs(argc, argv, init_parameters);

    // setup osc
    const int osc_buffer_size = 65535;

    osc::UdpSocket::SetUdpBufferSize(osc_buffer_size);
    osc::UdpTransmitSocket osc_transmit_socket(osc::IpEndpointName(osc_send_address.c_str(), osc_send_port));

    char osc_buffer[osc_buffer_size];
    osc::OutboundPacketStream osc_stream(osc_buffer, osc_buffer_size);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Open Camera", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Enable Positional tracking (mandatory for object detection)
    PositionalTrackingParameters positional_tracking_parameters;
    //If the camera is static, uncomment the following line to have better performances
    //positional_tracking_parameters.set_as_static = true;

    returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enable Positional Tracking", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Enable the Body tracking module
    BodyTrackingParameters body_tracker_params;
    body_tracker_params.enable_tracking = true; // track people across images flow
    body_tracker_params.enable_body_fitting = true; // smooth skeletons moves

    if (joint_map.size() == 34)
    {
        body_tracker_params.body_format = sl::BODY_FORMAT::BODY_34;
    }
    else if (joint_map.size() == 38)
    {
        body_tracker_params.body_format = sl::BODY_FORMAT::BODY_38;
    }

    //body_tracker_params.body_format = sl::BODY_FORMAT::BODY_34;
    //body_tracker_params.body_format = sl::BODY_FORMAT::BODY_38;
    body_tracker_params.detection_model = isJetson ? BODY_TRACKING_MODEL::HUMAN_BODY_FAST : BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;
    //body_tracker_params.allow_reduced_precision_inference = true;

    returned_state = zed.enableBodyTracking(body_tracker_params);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enable Object Detection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    auto camera_config = zed.getCameraInformation().camera_configuration;

    // For 2D GUI
    float image_aspect_ratio = camera_config.resolution.width / (1.f * camera_config.resolution.height);
    int requested_low_res_w = min(1280, (int)camera_config.resolution.width);
    sl::Resolution display_resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio);

    cv::Mat image_left_ocv(display_resolution.height, display_resolution.width, CV_8UC4, 1);
    Mat image_left(display_resolution, MAT_TYPE::U8_C4, image_left_ocv.data, image_left_ocv.step);
    sl::float2 img_scale(display_resolution.width / (float) camera_config.resolution.width, display_resolution.height / (float) camera_config.resolution.height);


    // Create OpenGL Viewer
    GLViewer viewer;
    viewer.init(argc, argv);

    Pose cam_pose;
    cam_pose.pose_data.setIdentity();

    // Configure object detection runtime parameters
    BodyTrackingRuntimeParameters body_tracker_parameters_rt;
    body_tracker_parameters_rt.detection_confidence_threshold = 40;
    body_tracker_parameters_rt.skeleton_smoothing = 0.7;
    
    // Create ZED Bodies filled in the main loop
    Bodies bodies;

    // Main Loop
    bool quit = false;
    string window_name = "ZED| 2D View";
    int key_wait = 10;
    char key = ' ';
    while (!quit) {
        // Grab images
        auto err = zed.grab();
        if (err == ERROR_CODE::SUCCESS) {
            // Retrieve Detected Human Bodies
            zed.retrieveBodies(bodies, body_tracker_parameters_rt);

            // update osc
            update_osc(bodies, osc_transmit_socket, osc_stream);

            //OCV View
            zed.retrieveImage(image_left, VIEW::LEFT, MEM::CPU, display_resolution);
            zed.getPosition(cam_pose, REFERENCE_FRAME::WORLD);

            //Update GL View
            viewer.updateData(bodies, cam_pose.pose_data);

            //printf("bodies is tracked %d \n", bodies.is_tracked);
            render_2D(image_left_ocv, img_scale, bodies.body_list, bodies.is_tracked);
            cv::imshow(window_name, image_left_ocv);

            key = cv::waitKey(key_wait);

            if (key == 'q') quit = true;
            if (key == 'm') {
                if (key_wait > 0) key_wait = 0;
                else key_wait = 10;
            }
            if (!viewer.isAvailable()) quit = true;
        } 
        else if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            zed.setSVOPosition(0);
        }
        else
            quit = true;
    }

    // Release Bodies
    viewer.exit();
    image_left.free();
    bodies.body_list.clear();

    // Disable modules
    zed.disableBodyTracking();
    zed.disablePositionalTracking();
    zed.close();

    return EXIT_SUCCESS;
}

void parseArgs(int argc, char **argv, InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        } else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        }else if (arg.find("HD1200") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1200;
            cout << "[Sample] Using Camera in resolution HD1200" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        }else if (arg.find("SVGA") != string::npos) {
            param.camera_resolution = RESOLUTION::SVGA;
            cout << "[Sample] Using Camera in resolution SVGA" << endl;
        }else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    }
    
    if (argc == 3)
    {
        osc_send_address = argv[1];
        osc_send_port = std::stoi(argv[2]);
    }
    else if (argc == 4)
    {
        osc_send_address = argv[2];
        osc_send_port = std::stoi(argv[3]);
    }
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error]";
    cout << " " << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}