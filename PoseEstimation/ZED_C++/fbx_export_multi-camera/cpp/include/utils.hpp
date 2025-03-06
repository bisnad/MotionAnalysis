#pragma once

#include <sl/Camera.hpp>

/**
* @brief Compute the start frame of each SVO for playback to be synced
*
* @param svo_files Map camera index to SVO file path
* @return Map camera index to starting SVO frame for synced playback
*/
std::map<int, int> syncDATA(std::map<int, std::string> svo_files) {
    std::map<int, int> output; // map of camera index and frame index of the starting point for each

    // Open all SVO
    std::map<int, std::shared_ptr<sl::Camera>> p_zeds;

    for (auto &it : svo_files) {
        auto p_zed = std::make_shared<sl::Camera>();

        sl::InitParameters init_param;
        init_param.depth_mode = sl::DEPTH_MODE::NONE;
        init_param.camera_disable_self_calib = true;
        init_param.input.setFromSVOFile(it.second.c_str());

        auto error = p_zed->open(init_param);
        if (error == sl::ERROR_CODE::SUCCESS)
            p_zeds.insert(std::make_pair(it.first, p_zed));
        else {
            std::cerr << "Could not open file " << it.second.c_str() << ": " << sl::toString(error) << ". Skipping" << std::endl;
        }
    }

    // Compute the starting point, we have to take the latest one
    sl::Timestamp start_ts = 0;
    for (auto &it : p_zeds) {
        it.second->grab();
        auto ts = it.second->getTimestamp(sl::TIME_REFERENCE::IMAGE);

        if (ts > start_ts)
            start_ts = ts;
    }

    std::cout << "Found SVOs common starting time: " << start_ts << std::endl;

    // The starting point is now known, let's find the frame idx for all corresponding
    for (auto &it : p_zeds) {
        auto frame_position_at_ts = it.second->getSVOPositionAtTimestamp(start_ts);

        if (frame_position_at_ts != -1)
            output.insert(std::make_pair(it.first, frame_position_at_ts));
    }

    for (auto &it : p_zeds) it.second->close();

    return output;
}

// Array containing keypoint indexes of each joint's parent. Used to build skeleton node hierarchy.
const int parentsIdx[] = {
	-1,
	0,
	1,
	2,
	2,
	4,
	5,
	6,
	7,
	8,
	7,
	2,
	11,
	12,
	13,
	14,
	15,
	14,
	0,
	18,
	19,
	20,
	0,
	22,
	23,
	24,
	3,
	26,
	26,
	26,
	26,
	26,
	20,
	24
};

// List of children of each joint. Used to build skeleton node hierarchy.
std::vector<std::vector<int>> childenIdx{
	{1,18,22},
	{2},
	{11,3,4},
	{26},
	{5},
	{6},
	{7},
	{8,10},
	{9},
	{},
	{},
	{12},
	{13},
	{14},
	{15,17},
	{16},
	{},
	{},
	{19},
	{20},
	{21,32},
	{},
	{23},
	{24},
	{25,33},
	{},
	{27,28,29,30,31},
	{},
	{},
	{},
	{},
	{},
	{},
	{}
};

// Local joint position of each joint. Used to build skeleton rest pose.
std::vector<sl::float3> local_joints_translations{
	sl::float3(0,0,0),              // 0
	sl::float3(0,20,0),				// 1
	sl::float3(0,20,0),				// 2
	sl::float3(0,20,0),				// 3
	sl::float3(-5,20,0),			// 4
	sl::float3(-15,0,0),			// 5
	sl::float3(-26,0,0),			// 6
	sl::float3(-25,0,0),			// 7
	sl::float3(-5,0,0),				// 8
	sl::float3(-10,0,0),			// 9
	sl::float3(-10,-6,0),			// 10
	sl::float3(5,20,0),				// 11
	sl::float3(15,0,0),				// 12
	sl::float3(26,0,0),				// 13
	sl::float3(25,0,0),				// 14
	sl::float3(5, 0, 0),			// 15
	sl::float3(10,0,0),				// 16
	sl::float3(10,-6,0),			// 17
	sl::float3(-10,0,0),			// 18
	sl::float3(0,-45,0),			// 19
	sl::float3(0,-40,0),			// 20
	sl::float3(0,-10,12),			// 21
	sl::float3(10,0,0),				// 22
	sl::float3(0,-45,0),			// 23
	sl::float3(0,-40,0),			// 24
	sl::float3(0,-10,12),			// 25
	sl::float3(0,15,0),				// 26
	sl::float3(0,10,0),				// 27
	sl::float3(-3,13,0),			// 28
	sl::float3(-8.5,10,-10),		// 29
	sl::float3(3,13,0),				// 30
	sl::float3(8.5,10,-10),			// 31
	sl::float3(0,-10,-4),			// 32
	sl::float3(0,-10,-4),			// 33
};