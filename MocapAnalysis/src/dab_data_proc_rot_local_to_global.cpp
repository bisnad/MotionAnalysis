/** \file dab_data_proc_rot_local_to_global.cpp
*/

#include "dab_data_proc_rot_local_to_global.h"
#include "dab_data.h"
#include <math.h> 

using namespace dab;

DataProcRotLocalToGlobal::DataProcRotLocalToGlobal()
	: DataProc()
{}

DataProcRotLocalToGlobal::DataProcRotLocalToGlobal(const std::string& pName, const std::string& pOutputDataName, const std::vector< std::vector<int> >& pJointConnectivity)
	: DataProc(mName)
	, mOutputDataName(pOutputDataName)
	, mJointConnectivity(pJointConnectivity)
{
	//// skeleton joint connectivity (Qualisys Version with all joints)
	//mSkeletonEdgeConnectivity.push_back({ 1, 56, 60 }); // Hips(0) -> Spine(1), LeftUpLeg(56), RightUpLeg(60)
	//mSkeletonEdgeConnectivity.push_back({ 2 }); // Spine(1) -> Spine1(2)
	//mSkeletonEdgeConnectivity.push_back({3}); // Spine1(2) -> Spine2(3)
	//mSkeletonEdgeConnectivity.push_back({4, 6, 31}); // Spine2(3) -> Neck(4), LeftShoulder(6), RightShoulder(31)
	//mSkeletonEdgeConnectivity.push_back({5}); // Neck(4) -> Head(5)
	//mSkeletonEdgeConnectivity.push_back({}); // Head(5) ->
	//mSkeletonEdgeConnectivity.push_back({7}); // LeftShoulder(6) -> LeftArm(7)
	//mSkeletonEdgeConnectivity.push_back({8}); // LeftArm(7) -> LeftForeArm(8)
	//mSkeletonEdgeConnectivity.push_back({9}); // LeftForeArm(8) -> LeftForeArmRoll(9)
	//mSkeletonEdgeConnectivity.push_back({10}); // LeftForeArmRoll(9) -> LeftHand(10)
	//mSkeletonEdgeConnectivity.push_back({11, 15, 19, 23, 27}); // LeftHand(10) -> LeftInHandThumb(11), LeftInHandIndex(15), LeftInHandMiddle(19), LeftInHandRing(23), LeftInHandPinky(27)
	//mSkeletonEdgeConnectivity.push_back({12}); // LeftInHandThumb(11) -> LeftHandThumb1(12)
	//mSkeletonEdgeConnectivity.push_back({13}); // LeftHandThumb1(12) -> LeftHandThumb2(13)
	//mSkeletonEdgeConnectivity.push_back({14}); // LeftHandThumb2(13) -> LeftHandThumb3(14)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftHandThumb3(14) ->
	//mSkeletonEdgeConnectivity.push_back({16}); // LeftInHandIndex(15) -> LeftHandIndex1(16)
	//mSkeletonEdgeConnectivity.push_back({17}); // LeftHandIndex1(16) -> LeftHandIndex2(17)
	//mSkeletonEdgeConnectivity.push_back({18}); // LeftHandIndex2(17) -> LeftHandIndex3(18)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftHandIndex3(18) ->
	//mSkeletonEdgeConnectivity.push_back({20}); // LeftInHandMiddle(19) -> LeftHandMiddle1(20)
	//mSkeletonEdgeConnectivity.push_back({21}); // LeftHandMiddle1(20) -> LeftHandMiddle2(21)
	//mSkeletonEdgeConnectivity.push_back({22}); // LeftHandMiddle2(21) -> LeftHandMiddle3(22)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftHandMiddle3(22) ->
	//mSkeletonEdgeConnectivity.push_back({24}); // LeftInHandRing(23) -> LeftHandRing1(24)
	//mSkeletonEdgeConnectivity.push_back({25}); // LeftHandRing1(24) -> LeftHandRing2(25)
	//mSkeletonEdgeConnectivity.push_back({26}); // LeftHandRing2(25) -> LeftHandRing3(26)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftHandRing3(26) ->
	//mSkeletonEdgeConnectivity.push_back({28}); // LeftInHandPinky(27) -> LeftHandPinky1(28)
	//mSkeletonEdgeConnectivity.push_back({29}); // LeftHandPinky1(28) -> LeftHandPinky2(29)
	//mSkeletonEdgeConnectivity.push_back({30}); // LeftHandPinky2(29) -> LeftHandPinky3(30)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftHandPinky3(30) ->
	//mSkeletonEdgeConnectivity.push_back({32}); // RightShoulder(31) -> RightArm(32)
	//mSkeletonEdgeConnectivity.push_back({33}); // RightArm(32) -> RightForeArm(33)
	//mSkeletonEdgeConnectivity.push_back({34}); // RightForeArm(33) -> RightForeArmRoll(34)
	//mSkeletonEdgeConnectivity.push_back({35}); // RightForeArmRoll(34) -> RightHand(35)
	//mSkeletonEdgeConnectivity.push_back({36, 40, 44, 48, 52}); // RightHand(35) -> RightInHandThumb(36), RightInHandIndex(40), RightInHandMiddle(44), RightInHandRing(48), LeftInHandPinky(52)
	//mSkeletonEdgeConnectivity.push_back({37}); // RightInHandThumb(36) -> RightHandThumb1(37)
	//mSkeletonEdgeConnectivity.push_back({38}); // RightHandThumb1(37) -> RightHandThumb2(38)
	//mSkeletonEdgeConnectivity.push_back({39}); // RightHandThumb2(38) -> RightHandThumb3(39)
	//mSkeletonEdgeConnectivity.push_back({}); // RightHandThumb3(39) ->
	//mSkeletonEdgeConnectivity.push_back({41}); // RightInHandIndex(40) -> RightHandIndex1(41)
	//mSkeletonEdgeConnectivity.push_back({42}); // RightHandIndex1(41) -> RightHandIndex2(42)
	//mSkeletonEdgeConnectivity.push_back({43}); // RightHandIndex2(42) -> RightHandIndex3(43)
	//mSkeletonEdgeConnectivity.push_back({}); // RightHandIndex3(43) ->
	//mSkeletonEdgeConnectivity.push_back({45}); // RightInHandMiddle(44) -> RightHandMiddle1(45)
	//mSkeletonEdgeConnectivity.push_back({46}); // RightHandMiddle1(45) -> RightHandMiddle2(46)
	//mSkeletonEdgeConnectivity.push_back({47}); // RightHandMiddle2(46) -> RightHandMiddle3(47)
	//mSkeletonEdgeConnectivity.push_back({}); // RightHandMiddle3(47) ->
	//mSkeletonEdgeConnectivity.push_back({49}); // RightInHandRing(48) -> RightHandRing1(49)
	//mSkeletonEdgeConnectivity.push_back({50}); // RightHandRing1(49) -> RightHandRing2(50)
	//mSkeletonEdgeConnectivity.push_back({51}); // RightHandRing2(50) -> RightHandRing3(51)
	//mSkeletonEdgeConnectivity.push_back({}); // RightHandRing3(51) ->
	//mSkeletonEdgeConnectivity.push_back({53}); // RightInHandPinky(52) -> RightHandPinky1(53)
	//mSkeletonEdgeConnectivity.push_back({54}); // RightHandPinky1(53) -> RightHandPinky2(54)
	//mSkeletonEdgeConnectivity.push_back({55}); // RightHandPinky2(54) -> RightHandPinky3(55)
	//mSkeletonEdgeConnectivity.push_back({}); // RightHandPinky3(55) ->
	//mSkeletonEdgeConnectivity.push_back({57}); // LeftUpLeg(56) -> LeftLeg(57)
	//mSkeletonEdgeConnectivity.push_back({58}); // LeftLeg(57) -> LeftFoot(58)
	//mSkeletonEdgeConnectivity.push_back({59}); // LeftFoot(58) -> LeftToeBase(59)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftToeBase(59) -> 
	//mSkeletonEdgeConnectivity.push_back({61}); // RightUpLeg(60) -> RightLeg(61)
	//mSkeletonEdgeConnectivity.push_back({62}); // RightLeg(61) -> RightFoot(62)
	//mSkeletonEdgeConnectivity.push_back({63}); // RightFoot(62) -> RightToeBase(63)
	//mSkeletonEdgeConnectivity.push_back({}); // RightToeBase(63) -> 

	//// skeleton joint connectivity (motion builder Version with all joints)
	//mSkeletonEdgeConnectivity.push_back({ 1, 67, 72 }); // Hips(0) -> Spine(1), LeftUpLeg(67), RightUpLeg(72)
	//mSkeletonEdgeConnectivity.push_back({ 2 }); // Spine(1) -> Spine1(2)
	//mSkeletonEdgeConnectivity.push_back({ 3 }); // Spine1(2) -> Spine2(3)
	//mSkeletonEdgeConnectivity.push_back({ 4, 7, 37 }); // Spine2(3) -> Neck(4), LeftShoulder(7), RightShoulder(37)
	//mSkeletonEdgeConnectivity.push_back({ 5 }); // Neck(4) -> Head(5)
	//mSkeletonEdgeConnectivity.push_back({ 6 }); // Head(5) -> Head_Nub(6)
	//mSkeletonEdgeConnectivity.push_back({}); // Head_Nub(6) ->
	//mSkeletonEdgeConnectivity.push_back({ 8 }); // LeftShoulder(7) -> LeftArm(8)
	//mSkeletonEdgeConnectivity.push_back({ 9 }); // LeftArm(8) -> LeftForeArm(9)
	//mSkeletonEdgeConnectivity.push_back({ 10 }); // LeftForeArm(9) -> LeftForeArmRoll(10)
	//mSkeletonEdgeConnectivity.push_back({ 11 }); // LeftForeArmRoll(10) -> LeftHand(11)
	//mSkeletonEdgeConnectivity.push_back({ 12, 17, 22, 27, 32 }); // LeftHand(11) -> LeftInHandThumb(12), LeftInHandIndex(17), LeftInHandMiddle(22), LeftInHandRing(27), LeftInHandPinky(32)
	//mSkeletonEdgeConnectivity.push_back({ 13 }); // LeftInHandThumb(12) -> LeftHandThumb1(13)
	//mSkeletonEdgeConnectivity.push_back({ 14 }); // LeftHandThumb1(13) -> LeftHandThumb2(14)
	//mSkeletonEdgeConnectivity.push_back({ 15 }); // LeftHandThumb2(14) -> LeftHandThumb3(15)
	//mSkeletonEdgeConnectivity.push_back({ 16 }); // LeftHandThumb3(15) -> LeftHandThumb3_Nub(16)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftHandThumb3_Nub(16) -> 
	//mSkeletonEdgeConnectivity.push_back({ 18 }); // LeftInHandIndex(17) -> LeftHandIndex1(18)
	//mSkeletonEdgeConnectivity.push_back({ 19 }); // LeftHandIndex1(18) -> LeftHandIndex2(19)
	//mSkeletonEdgeConnectivity.push_back({ 20 }); // LeftHandIndex2(19) -> LeftHandIndex3(20)
	//mSkeletonEdgeConnectivity.push_back({ 21 }); // LeftHandIndex3(20) -> LeftHandIndex3_Nub(21)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftHandIndex3_Nub(21)->
	//mSkeletonEdgeConnectivity.push_back({ 23 }); // LeftInHandMiddle(22) -> LeftHandMiddle1(23)
	//mSkeletonEdgeConnectivity.push_back({ 24 }); // LeftHandMiddle1(23) -> LeftHandMiddle2(24)
	//mSkeletonEdgeConnectivity.push_back({ 25 }); // LeftHandMiddle2(24) -> LeftHandMiddle3(25)
	//mSkeletonEdgeConnectivity.push_back({ 26 }); // LeftHandMiddle3(25) -> LeftHandMiddle3_Nub(26)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftHandMiddle3_Nub(26) ->
	//mSkeletonEdgeConnectivity.push_back({ 28 }); // LeftInHandRing(27) -> LeftHandRing1(28)
	//mSkeletonEdgeConnectivity.push_back({ 29 }); // LeftHandRing1(28) -> LeftHandRing2(29)
	//mSkeletonEdgeConnectivity.push_back({ 30 }); // LeftHandRing2(29) -> LeftHandRing3(30)
	//mSkeletonEdgeConnectivity.push_back({ 31 }); // LeftHandRing3(30) -> LeftHandRing3_Nub(31)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftHandRing3_Nub(31) ->
	//mSkeletonEdgeConnectivity.push_back({ 33 }); // LeftInHandPinky(32) -> LeftHandPinky1(33)
	//mSkeletonEdgeConnectivity.push_back({ 34 }); // LeftHandPinky1(33) -> LeftHandPinky2(34)
	//mSkeletonEdgeConnectivity.push_back({ 35 }); // LeftHandPinky2(34) -> LeftHandPinky3(35)
	//mSkeletonEdgeConnectivity.push_back({ 36 }); // LeftHandPinky3(35) -> LeftHandPinky3_Nub(36)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftHandPinky3_Nub(36) ->
	//mSkeletonEdgeConnectivity.push_back({ 38 }); // RightShoulder(37) -> RightArm(38)
	//mSkeletonEdgeConnectivity.push_back({ 39 }); // RightArm(38) -> RightForeArm(39)
	//mSkeletonEdgeConnectivity.push_back({ 40 }); // RightForeArm(39) -> RightForeArmRoll(40)
	//mSkeletonEdgeConnectivity.push_back({ 41 }); // RRightForeArmRoll(40) -> RightHand(41)
	//mSkeletonEdgeConnectivity.push_back({ 42, 47, 52, 57, 62 }); // RightHand(41) -> RightInHandThumb(42), RightInHandIndex(47), RightInHandMiddle(52), RightInHandRing(57), LeftInHandPinky(62)
	//mSkeletonEdgeConnectivity.push_back({ 43 }); // RightInHandThumb(42) -> RightHandThumb1(43)
	//mSkeletonEdgeConnectivity.push_back({ 44 }); // RightHandThumb1(43) -> RightHandThumb2(44)
	//mSkeletonEdgeConnectivity.push_back({ 45 }); // RightHandThumb2(44) -> RightHandThumb3(45)
	//mSkeletonEdgeConnectivity.push_back({ 46 }); // RightHandThumb3(45) -> RightHandThumb3_Nub(46)
	//mSkeletonEdgeConnectivity.push_back({}); // RightHandThumb3_Nub(46) -> 
	//mSkeletonEdgeConnectivity.push_back({ 48 }); // RightInHandIndex(47) -> RightHandIndex1(48)
	//mSkeletonEdgeConnectivity.push_back({ 49 }); // RightHandIndex1(48) -> RightHandIndex2(49)
	//mSkeletonEdgeConnectivity.push_back({ 50 }); // RightHandIndex2(49) -> RightHandIndex3(50)
	//mSkeletonEdgeConnectivity.push_back({ 51 }); // RightHandIndex3(50) -> RightHandIndex3_Nub(51)
	//mSkeletonEdgeConnectivity.push_back({}); // RightHandIndex3_Nub(51)->
	//mSkeletonEdgeConnectivity.push_back({ 53 }); // RightInHandMiddle(52) -> RightHandMiddle1(53)
	//mSkeletonEdgeConnectivity.push_back({ 54 }); // RightHandMiddle1(53) -> RightHandMiddle2(54)
	//mSkeletonEdgeConnectivity.push_back({ 55 }); // RightHandMiddle2(54) -> RightHandMiddle3(55)
	//mSkeletonEdgeConnectivity.push_back({ 56 }); // RightHandMiddle3(55) -> RightHandMiddle3_Nub(56)
	//mSkeletonEdgeConnectivity.push_back({}); // RightHandMiddle3_Nub(56) ->
	//mSkeletonEdgeConnectivity.push_back({ 58 }); // RightInHandRing(57) -> RightHandRing1(58)
	//mSkeletonEdgeConnectivity.push_back({ 59 }); // RightHandRing1(58) -> RightHandRing2(59)
	//mSkeletonEdgeConnectivity.push_back({ 60 }); // RightHandRing2(59) -> RightHandRing3(60)
	//mSkeletonEdgeConnectivity.push_back({ 61 }); // RightHandRing3(60) -> RightHandRing3_Nub(61)
	//mSkeletonEdgeConnectivity.push_back({}); // RightHandRing3_Nub(61) ->
	//mSkeletonEdgeConnectivity.push_back({ 63 }); // RightInHandPinky(62) -> RightHandPinky1(63)
	//mSkeletonEdgeConnectivity.push_back({ 64 }); // RightHandPinky1(63) -> RightHandPinky2(64)
	//mSkeletonEdgeConnectivity.push_back({ 65 }); // RightHandPinky2(64) -> RightHandPinky3(65)
	//mSkeletonEdgeConnectivity.push_back({ 66 }); // RightHandPinky3(65) -> RightHandPinky3_Nub(66)
	//mSkeletonEdgeConnectivity.push_back({}); // RightHandPinky3_Nub(66) ->
	//mSkeletonEdgeConnectivity.push_back({ 68 }); // LeftUpLeg(67) -> LeftLeg(68)
	//mSkeletonEdgeConnectivity.push_back({ 69 }); // LeftLeg(68) -> LeftFoot(69)
	//mSkeletonEdgeConnectivity.push_back({ 70 }); // LeftFoot(69) -> LeftToeBase(70)
	//mSkeletonEdgeConnectivity.push_back({ 71 }); // LeftToeBase(70) -> LeftToeBase_Nub(71)
	//mSkeletonEdgeConnectivity.push_back({}); // LeftToeBase_Nub(71) -> 
	//mSkeletonEdgeConnectivity.push_back({ 73 }); // RightUpLeg(72) -> RightLeg(73)
	//mSkeletonEdgeConnectivity.push_back({ 74 }); // RightLeg(73) -> RightFoot(74)
	//mSkeletonEdgeConnectivity.push_back({ 75 }); // RightFoot(74) -> RightToeBase(75)
	//mSkeletonEdgeConnectivity.push_back({ 76 }); // RightToeBase(75) -> RightToeBase_Nub(76)
	//mSkeletonEdgeConnectivity.push_back({}); // RightToeBase_Nub(76) -> 

	int jointCount = mJointConnectivity.size();

	mJointRotationsLocal = std::vector<glm::quat>(jointCount, glm::quat(1, 0, 0, 0));
	mJointRotationsGlobal = std::vector<glm::quat>(jointCount, glm::quat(1, 0, 0, 0));

	mDataRotationsGlobal = std::shared_ptr<Data>(new Data(mOutputDataName, 4, jointCount));
	mData.push_back(mDataRotationsGlobal);
}

void
DataProcRotLocalToGlobal::process() throw (dab::Exception)
{
	//std::cout << "DataProcQuatLocalToGlobal::process()\n";

	if (mInputProcs.size() == 0) return;

	const std::vector< std::shared_ptr<Data> >& inputDataVector = mInputProcs[0]->data();

	if (inputDataVector.size() == 0) return;

	// take only the first data from potentially longer list of data
	// ignore the remaining data
	std::shared_ptr<Data> inputData = inputDataVector[0];

	const std::string& inputDataName = inputData->name();
	int valueDim = inputData->valueDim();
	int valueCount = inputData->valueCount();
	int arraySize = valueDim * valueCount;

	int jointDim = mDataRotationsGlobal->valueDim();
	int jointCount = mDataRotationsGlobal->valueCount();

	//std::cout << "valueCount " << valueCount << " jointCount " << jointCount << "\n";
	//std::cout << "valueDim " << valueDim << " jointDim " << jointDim << "\n";

	if (valueCount != jointCount) throw dab::Exception("DATA ERROR: Data " + inputData->name() + " does not contain expected value count", __FILE__, __FUNCTION__, __LINE__);
	if (valueDim != jointDim) throw dab::Exception("DATA ERROR: Data " + inputData->name() + " does not contain expected value dim", __FILE__, __FUNCTION__, __LINE__);

	const std::vector<float>& quatRotationsLocal = inputData->values();
	std::vector<float>& quatRotationsGlobal = mDataRotationsGlobal->values();

	for (int jI = 0, dI = 0; jI < jointCount; ++jI, dI += 4)
	{
		//	mJointRotationsLocal[jI] = glm::quat(quatRotationsLocal[dI], quatRotationsLocal[dI+1], quatRotationsLocal[dI+2], quatRotationsLocal[dI+3]);

		mJointRotationsLocal[jI][0] = quatRotationsLocal[dI];
		mJointRotationsLocal[jI][1] = quatRotationsLocal[dI + 1];
		mJointRotationsLocal[jI][2] = quatRotationsLocal[dI + 2];
		mJointRotationsLocal[jI][3] = quatRotationsLocal[dI + 3];
	}

	// assumes that joint 0 is root
	mJointRotationsGlobal[0] = mJointRotationsLocal[0];
	calcGlobalRotation(mJointConnectivity[0], mJointRotationsGlobal[0]);

	//std::cout << "jointCount " << jointCount << "\n";

	//// debug
	//std::cout << "calc joint global rot\n";
	//for (int jI = 0; jI < jointCount; ++jI)
	//{
	//	std::cout << "jI " << jI << " loc rot " << mJointRotationsLocal[jI][0] << " " << mJointRotationsLocal[jI][1] << " " << mJointRotationsLocal[jI][2] << " " << mJointRotationsLocal[jI][3] << "\n";
	//	std::cout << "jI " << jI << " glob rot " << mJointRotationsGlobal[jI][0] << " " << mJointRotationsGlobal[jI][1] << " " << mJointRotationsGlobal[jI][2] << " " << mJointRotationsGlobal[jI][3] << "\n";
	//}
	//// debug done

	for (int jI = 0, dI = 0; jI < jointCount; ++jI, dI += 4)
	{
		quatRotationsGlobal[dI] = mJointRotationsGlobal[jI][0];
		quatRotationsGlobal[dI+1] = mJointRotationsGlobal[jI][1];
		quatRotationsGlobal[dI+2] = mJointRotationsGlobal[jI][2];
		quatRotationsGlobal[dI+3] = mJointRotationsGlobal[jI][3];
	}
}

void 
DataProcRotLocalToGlobal::calcGlobalRotation(const std::vector<int>& pChildJointIndices, const glm::quat& pParentRotation)
{
	for (int childJointIndex : pChildJointIndices)
	{
		// TODO: verify that this is the correct sequence of multiplication
		mJointRotationsGlobal[childJointIndex] = pParentRotation * mJointRotationsLocal[childJointIndex]; 
		calcGlobalRotation(mJointConnectivity[childJointIndex], mJointRotationsGlobal[childJointIndex]);
	}
}