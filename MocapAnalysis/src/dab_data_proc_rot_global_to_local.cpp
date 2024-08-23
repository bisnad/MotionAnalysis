/** \file dab_data_proc_rot_global_to_local.cpp
*/

#include "dab_data_proc_rot_global_to_local.h"
#include "dab_data.h"
#include <math.h> 

using namespace dab;

DataProcRotGlobalToLocal::DataProcRotGlobalToLocal()
	: DataProc()
{}

DataProcRotGlobalToLocal::DataProcRotGlobalToLocal(const std::string& pName, const std::string& pOutputDataName, const std::vector< std::vector<int> >& pJointConnectivity)
	: DataProc(mName)
	, mOutputDataName(pOutputDataName)
	, mJointConnectivity(pJointConnectivity)
{
	int jointCount = mJointConnectivity.size();

	mJointRotationsGlobal = std::vector<glm::quat>(jointCount, glm::quat(1, 0, 0, 0));
	mJointRotationsLocal = std::vector<glm::quat>(jointCount, glm::quat(1, 0, 0, 0));

	mDataRotationsLocal= std::shared_ptr<Data>(new Data(mOutputDataName, 4, jointCount));
	mData.push_back(mDataRotationsLocal);
}

void
DataProcRotGlobalToLocal::process() throw (dab::Exception)
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

	int jointDim = mDataRotationsLocal->valueDim();
	int jointCount = mDataRotationsLocal->valueCount();

	//std::cout << "valueCount " << valueCount << " jointCount " << jointCount << "\n";
	//std::cout << "valueDim " << valueDim << " jointDim " << jointDim << "\n";

	if (valueCount != jointCount) throw dab::Exception("DATA ERROR: Data " + inputData->name() + " does not contain expected value count", __FILE__, __FUNCTION__, __LINE__);
	if (valueDim != jointDim) throw dab::Exception("DATA ERROR: Data " + inputData->name() + " does not contain expected value dim", __FILE__, __FUNCTION__, __LINE__);

	const std::vector<float>& quatRotationsGlobal = inputData->values();
	std::vector<float>& quatRotationsLocal = mDataRotationsLocal->values();

	for (int jI = 0, dI = 0; jI < jointCount; ++jI, dI += 4)
	{
		mJointRotationsGlobal[jI][0] = quatRotationsGlobal[dI];
		mJointRotationsGlobal[jI][1] = quatRotationsGlobal[dI + 1];
		mJointRotationsGlobal[jI][2] = quatRotationsGlobal[dI + 2];
		mJointRotationsGlobal[jI][3] = quatRotationsGlobal[dI + 3];
	}

	//// flipped
	//for (int jI = 0, dI = 0; jI < jointCount; ++jI, dI += 4)
	//{
	//	mJointRotationsGlobal[jI][3] = quatRotationsGlobal[dI];
	//	mJointRotationsGlobal[jI][0] = quatRotationsGlobal[dI + 1];
	//	mJointRotationsGlobal[jI][1] = quatRotationsGlobal[dI + 2];
	//	mJointRotationsGlobal[jI][2] = quatRotationsGlobal[dI + 3];
	//}

	// assumes that joint 0 is root
	//mJointRotationsLocal[0] = mJointRotationsGlobal[0];

	mJointRotationsLocal[0][0] = mJointRotationsGlobal[0][0];
	mJointRotationsLocal[0][1] = mJointRotationsGlobal[0][1];
	mJointRotationsLocal[0][2] = mJointRotationsGlobal[0][2];
	mJointRotationsLocal[0][3] = mJointRotationsGlobal[0][3];

	calcLocalRotation(mJointConnectivity[0], mJointRotationsGlobal[0]);

	//std::cout << "jointCount " << jointCount << "\n";

	//// debug
	//std::cout << "calc joint local rot\n";
	//for (int jI = 0; jI < jointCount; ++jI)
	//{
	//	std::cout << "jI " << jI << " glob rot " << mJointRotationsGlobal[jI][0] << " " << mJointRotationsGlobal[jI][1] << " " << mJointRotationsGlobal[jI][2] << " " << mJointRotationsGlobal[jI][3] << "\n";
	//	std::cout << "jI " << jI << " loc rot " << mJointRotationsLocal[jI][0] << " " << mJointRotationsLocal[jI][1] << " " << mJointRotationsLocal[jI][2] << " " << mJointRotationsLocal[jI][3] << "\n";
	//}
	//// debug done

	for (int jI = 0, dI = 0; jI < jointCount; ++jI, dI += 4)
	{
		quatRotationsLocal[dI] = mJointRotationsLocal[jI][0];
		quatRotationsLocal[dI + 1] = mJointRotationsLocal[jI][1];
		quatRotationsLocal[dI + 2] = mJointRotationsLocal[jI][2];
		quatRotationsLocal[dI + 3] = mJointRotationsLocal[jI][3];
	}
}

void
DataProcRotGlobalToLocal::calcLocalRotation(const std::vector<int>& pChildJointIndices, const glm::quat& pParentRotation)
{
	glm::quat invParentRot = glm::inverse(pParentRotation);

	for (int childJointIndex : pChildJointIndices)
	{
		mJointRotationsLocal[childJointIndex] = invParentRot * mJointRotationsGlobal[childJointIndex];
		//mJointRotationsLocal[childJointIndex] = mJointRotationsGlobal[childJointIndex] * invParentRot;
		calcLocalRotation(mJointConnectivity[childJointIndex], mJointRotationsGlobal[childJointIndex]);
	}
}