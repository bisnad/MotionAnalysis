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

// ZED include
#include "ClientPublisher.hpp"
#include "GLViewer.hpp"
#include "utils.hpp"

// FBX includes
#include <fbxsdk.h>
#include "Common.h"

// FBX settings and tools
float const FbxAnimCurveDef::sDEFAULT_WEIGHT = 1.0;
float const FbxAnimCurveDef::sDEFAULT_VELOCITY = 1.0;
std::vector<std::string> vec_str_component = { FBXSDK_CURVENODE_COMPONENT_X, FBXSDK_CURVENODE_COMPONENT_Y, FBXSDK_CURVENODE_COMPONENT_Z };
int key_index = 0;

struct SkeletonHandler {
    FbxNode* root;
    std::vector<FbxNode*> joints;
};

SkeletonHandler CreateSkeleton(FbxScene* pScene);

bool fbx_has_started = false;
sl::Timestamp ts_start = sl::Timestamp(0);
sl::RuntimeParameters rt_params;

// Create Skeleton node hierarchy based on sl::BODY_FORMAT::POSE_34 body format.
SkeletonHandler CreateSkeleton(FbxScene* pScene) {
    FbxNode* reference_node = FbxNode::Create(pScene, ("Skeleton"));
    SkeletonHandler skeleton;
    for (int i = 0; i < static_cast<int>(sl::BODY_34_PARTS::LAST); i++) {
        FbxString joint_name;
        joint_name = sl::toString(static_cast<sl::BODY_34_PARTS>(i)).c_str();
        FbxSkeleton* skeleton_node_attribute = FbxSkeleton::Create(pScene, joint_name);
        skeleton_node_attribute->SetSkeletonType(FbxSkeleton::eLimbNode);
        skeleton_node_attribute->Size.Set(1.0);
        FbxNode* skeleton_node = FbxNode::Create(pScene, joint_name.Buffer());
        skeleton_node->SetNodeAttribute(skeleton_node_attribute);

        FbxDouble3 tr(local_joints_translations[i].x, local_joints_translations[i].y, local_joints_translations[i].z);
        skeleton_node->LclTranslation.Set(tr);

        skeleton.joints.push_back(skeleton_node);
    }

    reference_node->AddChild(skeleton.joints[0]);

    // Build skeleton node hierarchy. 
    for (int i = 0; i < skeleton.joints.size(); i++) {
        for (int j = 0; j < childenIdx[i].size(); j++)
            skeleton.joints[i]->AddChild(skeleton.joints[childenIdx[i][j]]);
    }

    skeleton.root = reference_node;

    return skeleton;
}

void updateFBX(sl::Bodies& bodies, FbxTime& time, FbxScene* fbx_scene, SkeletonHandler& skeleton, std::map<int, FbxAnimLayer*>& animLayers)
{
    if (!fbx_has_started) {
        ts_start = bodies.timestamp;
        fbx_has_started = true;
    }

    // Compute animation timestamp
    sl::Timestamp ts_ms = (bodies.timestamp - ts_start).getMilliseconds();

    //std::cout << "ts_ms " << ts_ms << "\n";

    time.SetMilliSeconds(ts_ms);

    // For each detection
    for (auto& it : bodies.body_list) {
        // Create a new animLayer if it is a new detection
        if (animLayers.find(it.id) == animLayers.end())
        {
            FbxAnimStack* anim_stack = FbxAnimStack::Create(fbx_scene, ("Anim Stack  ID " + std::to_string(it.id)).c_str());
            // Create the base layer (this is mandatory)
            FbxAnimLayer* anim_base_layer = FbxAnimLayer::Create(fbx_scene, ("Base Layer " + std::to_string(it.id)).c_str());
            anim_stack->AddMember(anim_base_layer);
            animLayers[it.id] = anim_base_layer;
        }

        auto anim_id = animLayers[it.id];

        // For each keypoint
        for (int j = 0; j < skeleton.joints.size(); j++) {
            auto joint = skeleton.joints[j];
            sl::Orientation lcl_rotation = sl::Orientation(it.local_orientation_per_joint[j]);

            // Set translation of the root (first joint)
            if (j == 0) {
                sl::float3 rootPosition = it.keypoint[0];
                joint->LclTranslation.GetCurveNode(anim_id, true);

                for (int t = 0; t < 3; t++) {
                    FbxAnimCurve* lCurve = joint->LclTranslation.GetCurve(anim_id, vec_str_component[t].c_str(), true);
                    if (lCurve) {
                        lCurve->KeyModifyBegin();
                        key_index = lCurve->KeyAdd(time);
                        lCurve->KeySet(key_index,
                            time,
                            rootPosition.v[t],
                            FbxAnimCurveDef::eInterpolationConstant);
                        lCurve->KeyModifyEnd();
                    }
                }
                // Use global rotation for the root
                lcl_rotation = sl::Orientation(it.global_root_orientation);
            }

            // Convert rotation to euler angles
            FbxQuaternion quat = FbxQuaternion(lcl_rotation.x, lcl_rotation.y, lcl_rotation.z, lcl_rotation.w);
            FbxVector4 rota_euler;
            rota_euler.SetXYZ(quat);

            // Set local rotation of the joint
            for (int r = 0; r < 3; r++) {
                FbxAnimCurve* lCurve = joint->LclRotation.GetCurve(anim_id, vec_str_component[r].c_str(), true);
                if (lCurve) {
                    lCurve->KeyModifyBegin();
                    key_index = lCurve->KeyAdd(time);
                    lCurve->KeySet(key_index,
                        time,
                        rota_euler[r],
                        FbxAnimCurveDef::eInterpolationConstant);
                    lCurve->KeyModifyEnd();
                }
            }
        }
    }
}

int exportFBX(FbxManager* fbx_manager, FbxScene* fbx_scene)
{
    // Save the scene.    
    std::string sampleFileName = "ZedSkeletons.fbx";
    auto result = SaveScene(fbx_manager, fbx_scene, sampleFileName.c_str(), 0 /*save as binary*/);
    DestroySdkObjects(fbx_manager, result);

    if (result == false) {
        FBXSDK_printf("\n\nAn error occurred while saving the scene...\n");
        return EXIT_FAILURE;
}
    else
        return EXIT_SUCCESS;
}

int main(int argc, char **argv) {

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
    //body_tracker_params.body_format = sl::BODY_FORMAT::BODY_34;
    body_tracker_params.body_format = sl::BODY_FORMAT::BODY_38;

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

    // initialise FBX
    FbxManager* fbx_manager = nullptr;
    FbxScene* fbx_scene = nullptr;
    // Prepare the FBX SDK.
    InitializeSdkObjects(fbx_manager, fbx_scene);
    FbxTime time;

    // Create FBX skeleton hierarchy
    SkeletonHandler skeleton = CreateSkeleton(fbx_scene);
    FbxNode* root_node = fbx_scene->GetRootNode();
    // Add skeleton in the Scene
    root_node->AddChild(skeleton.root);

    // List of all FBX anim layers, in case there are multiple skeletons in the scene
    std::map<int, FbxAnimLayer*> animLayers;

 


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

            // update FBX
            updateFBX(fused_bodies, time, fbx_scene, skeleton, animLayers);
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

    return exportFBX(fbx_manager, fbx_scene);
}
