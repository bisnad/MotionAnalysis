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

// OSC include
#pragma comment(lib, "ws2_32.lib")

#include "osc/OscOutboundPacketStream.h"
#include "ip/UdpSocket.h"

// ZED include
#include <sl/Camera.hpp>
#include "ClientPublisher.hpp"
#include "GLViewer.hpp"
#include "utils.hpp"

# include <array>


// joint map for body 34 to match joint numbers from live capture with those in an fbx recording
std::array<int, 34> joint_map = { 0, 1, 2, 11, 12, 13, 14, 15, 16, 17, 3, 26, 27, 28, 29, 30, 31, 4, 5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 32, 22, 23, 24, 25, 33 };
// joint map for body 38 to match joint numbers from live capture with those in an fbx recording
//std::array<int, 38> joint_map = { 0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 12, 14, 16, 30, 32, 34, 36, 11, 13, 15, 17, 31, 33, 35, 37, 18, 20, 22, 24, 26, 28, 19, 21, 23, 25, 27, 29 };

// joint parent indices for body 34
std::array<int, 34> joint_parent_indices = { -1, 0, 1, 2, 3, 4, 5, 6, 7, 6, 2, 10, 11, 11, 11, 11, 11, 2, 17, 18, 19, 20, 21, 20, 0, 24, 25, 26, 26, 0, 29, 30, 31, 31 };
// joint parent indices for body 38
//std::array<int, 38> joint_parent_indices = { -1, 0, 1, 2, 3, 4, 5, 6, 5, 8, 3, 10, 11, 12, 13, 13, 13, 13, 3, 18, 19, 20, 21, 21, 21, 21, 0, 26, 27, 28, 28, 28, 0, 32, 33, 34, 34, 34 };

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

    for(auto & _body : bodies.body_list)
    {
        int body_id = _body.id;

        //std::cout << "body_id " << body_id << "\n";

        std::vector<sl::float2> joint_pos2d_world = _body.keypoint_2d;
        std::vector<sl::float3> joint_pos3d_world = _body.keypoint;
        std::vector<sl::float4> joint_rot_local = _body.local_orientation_per_joint;
        std::vector<sl::float3> joint_pos_local = _body.local_position_per_joint;
        sl::float4 root_rot_world = _body.global_root_orientation;

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
            joint_rot_local_osc[jI] = joint_rot_local[joint_map[jI]];
            joint_pos_local_osc[jI] = joint_pos_local[joint_map[jI]];

            // quat conversion from x y z w to w x y z
            sl::float4 rot_xyzw = joint_rot_local_osc[jI];
            joint_rot_local_osc[jI] = sl::float4(rot_xyzw[3], rot_xyzw[0], rot_xyzw[1], rot_xyzw[2]);
        }

   
        // quat conversion from x y z w to w x y z
        sl::float4 rot_xyzw = root_rot_world;
        sl::float4 root_rot_world_osc = sl::float4(rot_xyzw[3], rot_xyzw[0], rot_xyzw[1], rot_xyzw[2]);

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

        // send rot_local
        pStream.Clear();
        pStream << osc::BeginMessage((std::string("/mocap/") + std::to_string(body_id) + "/joint/rot_local").c_str());
        for (int jI = 0; jI < joint_count; ++jI) pStream << joint_rot_local_osc[jI][0] << joint_rot_local_osc[jI][1] << joint_rot_local_osc[jI][2] << joint_rot_local_osc[jI][3];
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

    // setup osc
    std::string osc_send_address = "127.0.0.1";
    int osc_send_port = 9007;
    const int osc_buffer_size = 65535;

    if (argc == 4)
    {
        osc_send_address = argv[2];
        osc_send_port = std::stoi(argv[3]);
    }

    std::cout << "osc_send_address " << osc_send_address << "\n";
    std::cout << "osc_send_port " << osc_send_port << "\n";

    osc::UdpSocket::SetUdpBufferSize(osc_buffer_size);
    osc::UdpTransmitSocket osc_transmit_socket(osc::IpEndpointName(osc_send_address.c_str(), osc_send_port));

    char osc_buffer[osc_buffer_size];
    osc::OutboundPacketStream osc_stream(osc_buffer, osc_buffer_size);

    // setup zed
#ifdef _SL_JETSON_
    const bool isJetson = true;
#else
    const bool isJetson = false;
#endif

    // Defines the Coordinate system and unit used in this sample
    constexpr sl::COORDINATE_SYSTEM COORDINATE_SYSTEM = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    constexpr sl::UNIT UNIT = sl::UNIT::METER;

    std::vector<sl::FusionConfiguration> configurations;

    if (argc > 1) 
    {
        // Read json file containing the configuration of your multicamera setup.    
        configurations = sl::readFusionConfigurationFile(argv[1], COORDINATE_SYSTEM, UNIT);
	}
    else
    {
        // Read json file containing the configuration of your multicamera setup.    
        configurations = sl::readFusionConfigurationFile("data/calib/calib_file.json", COORDINATE_SYSTEM, UNIT);
    }

    if (configurations.empty()) {
        std::cout << "Empty configuration File." << std::endl;
        return EXIT_FAILURE;
    }



    // Enable the Body tracking module
    sl::BodyTrackingParameters body_tracker_params;
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
    body_tracker_params.detection_model = isJetson ? sl::BODY_TRACKING_MODEL::HUMAN_BODY_FAST : sl::BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;
    //body_tracker_params.allow_reduced_precision_inference = true;

    Trigger trigger;

    // Check if the ZED camera should run within the same process or if they are running on the edge.
    std::vector<ClientPublisher> clients(configurations.size());
    int id_ = 0;
    std::map<int, std::string> svo_files;
    for (auto conf: configurations) {
        // if the ZED camera should run locally, then start a thread to handle it
        if(conf.communication_parameters.getType() == sl::CommunicationParameters::COMM_TYPE::INTRA_PROCESS){
            std::cout << "Try to open ZED " <<conf.serial_number << ".." << std::flush;
            auto state = clients[id_].open(conf.input_type, body_tracker_params , &trigger);
            if (!state) {
                std::cerr << "Could not open ZED: " << conf.input_type.getConfiguration() << ". Skipping..." << std::endl;
                continue;
            }

            if (conf.input_type.getType() == sl::InputType::INPUT_TYPE::SVO_FILE)
                svo_files.insert(std::make_pair(id_, conf.input_type.getConfiguration()));

            std::cout << ". ready !" << std::endl;

            id_++;
        }
    }

    // Synchronize SVO files in SVO mode
    bool enable_svo_sync = (svo_files.size() > 1);
    if (enable_svo_sync) {
        std::cout << "Starting SVO sync process..." << std::endl;
        std::map<int, int> cam_idx_to_svo_frame_idx = syncDATA(svo_files);

        for (auto &it : cam_idx_to_svo_frame_idx) {
            std::cout << "Setting camera " << it.first << " to frame " << it.second << std::endl;
            clients[it.first].setStartSVOPosition(it.second);
        }
    }

    // start camera threads
    for (auto &it: clients)
        it.start();

    // Now that the ZED camera are running, we need to initialize the fusion module
    sl::InitFusionParameters init_params;
    init_params.coordinate_units = UNIT;
    init_params.coordinate_system = COORDINATE_SYSTEM;
    init_params.verbose = true;

    // create and initialize it
    sl::Fusion fusion;
    fusion.init(init_params);

    // subscribe to every cameras of the setup to internally gather their data
    std::vector<sl::CameraIdentifier> cameras;
    for (auto& it : configurations) {
        sl::CameraIdentifier uuid(it.serial_number);
        // to subscribe to a camera you must give its serial number, the way to communicate with it (shared memory or local network), and its world pose in the setup.        
        auto state = fusion.subscribe(uuid, it.communication_parameters, it.pose, it.override_gravity);
        if (state != sl::FUSION_ERROR_CODE::SUCCESS)
            std::cout << "Unable to subscribe to " << std::to_string(uuid.sn) << " . " << state << std::endl;
        else
            cameras.push_back(uuid);
    }

    // check that at least one camera is connected
    if (cameras.empty()) {
        std::cout << "no connections " << std::endl;
        return EXIT_FAILURE;
    }

    // as this sample shows how to fuse body detection from the multi camera setup
    // we enable the Body Tracking module with its options
    sl::BodyTrackingFusionParameters body_fusion_init_params;
    body_fusion_init_params.enable_tracking = true;
    body_fusion_init_params.enable_body_fitting = !isJetson; // skeletons will looks more natural but requires more computations
    fusion.enableBodyTracking(body_fusion_init_params);

    // define fusion behavior 
    sl::BodyTrackingFusionRuntimeParameters body_tracking_runtime_parameters;
    // be sure that the detection skeleton is complete enough
    body_tracking_runtime_parameters.skeleton_minimum_allowed_keypoints = 7;
    // we can also want to retrieve skeleton seen by multiple camera, in this case at least half of them
    body_tracking_runtime_parameters.skeleton_minimum_allowed_camera = cameras.size() / 2.;


    // creation of a 3D viewer
    GLViewer viewer;
    viewer.init(argc, argv);

    std::cout << "Viewer Shortcuts\n" <<
        "\t- 'q': quit the application\n" <<
        "\t- 'p': play/pause the GLViewer\n" <<
        "\t- 'r': switch on/off for raw skeleton display\n" <<
        "\t- 's': switch on/off for live point cloud display\n" <<
        "\t- 'c': switch on/off point cloud display with raw color\n" << std::endl;

    // fusion outputs
    sl::Bodies fused_bodies;
    std::map<sl::CameraIdentifier, sl::Bodies> camera_raw_data;
    sl::FusionMetrics metrics;
    std::map<sl::CameraIdentifier, sl::Mat> views;
    std::map<sl::CameraIdentifier, sl::Mat> pointClouds;
    sl::Resolution low_res(512,360);
    sl::CameraIdentifier fused_camera(0);

    // run the fusion as long as the viewer is available.
    while (viewer.isAvailable()) {
        trigger.notifyZED();

        // run the fusion process (which gather data from all camera, sync them and process them)
        if (fusion.process() == sl::FUSION_ERROR_CODE::SUCCESS) {
            // Retrieve fused body
            fusion.retrieveBodies(fused_bodies, body_tracking_runtime_parameters);

            // for debug, you can retrieve the data sent by each camera
            for (auto& id : cameras) { 
                fusion.retrieveBodies(camera_raw_data[id], body_tracking_runtime_parameters, id);
                sl::Pose pose;
                if(fusion.getPosition(pose, sl::REFERENCE_FRAME::WORLD, id, sl::POSITION_TYPE::RAW) == sl::POSITIONAL_TRACKING_STATE::OK)
                    viewer.setCameraPose(id.sn, pose.pose_data);

                auto state_view = fusion.retrieveImage(views[id], id, low_res);
                auto state_pc = fusion.retrieveMeasure(pointClouds[id], id, sl::MEASURE::XYZBGRA, low_res);

                if (state_view == sl::FUSION_ERROR_CODE::SUCCESS && state_pc == sl::FUSION_ERROR_CODE::SUCCESS)
                    viewer.updateCamera(id.sn, views[id], pointClouds[id]);
            }

            // get metrics about the fusion process for monitoring purposes
            fusion.getProcessMetrics(metrics);

            // update osc
            update_osc(fused_bodies, osc_transmit_socket, osc_stream);
        }
        // update the 3D view
        viewer.updateBodies(fused_bodies, camera_raw_data, metrics);

        while (!viewer.isPlaying() && viewer.isAvailable()) {
            sl::sleep_ms(10);
        }
    }

    viewer.exit();
    
    trigger.running = false;
    trigger.notifyZED();

    for (auto &it: clients)
        it.stop();

    fusion.close();

    return EXIT_SUCCESS;
}
