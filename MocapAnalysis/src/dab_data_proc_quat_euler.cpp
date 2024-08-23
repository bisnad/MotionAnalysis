/** \file dab_data_proc_quat_euler.cpp
*/

#include "dab_data_proc_quat_euler.h"
#include "dab_data.h"
#include "ofVectorMath.h"
#include <math.h> 

using namespace dab;

DataProcQuatEuler::DataProcQuatEuler()
	: DataProc()
{}
	
DataProcQuatEuler::DataProcQuatEuler(const std::string& pName, const std::string& pOutputDataName)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
{
	mDataEuler = std::shared_ptr<Data>(new Data(mOutputDataName, 3, 1));
	mData.push_back(mDataEuler);
}

void 
DataProcQuatEuler::process() throw (dab::Exception)
{
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

	if (valueDim != 4) throw dab::Exception("DATA ERROR: Data " + inputData->name() + " does not contain values with dimension 4", __FILE__, __FUNCTION__, __LINE__);

	if (mDataEuler->valueCount() != valueCount)
	{
		mDataEuler->resize(3, valueCount);
	}

	const std::vector<float>& quatRotations = inputData->values();
	std::vector<float>& eulerRotations = mDataEuler->values();

	glm::quat quatRotation;
	glm::vec3 eulerRotation;

	for (int vI = 0, qI=0, eI=0; vI < valueCount; ++vI, qI +=4, eI += 3)
	{
		// get current quaternion
		//quatRotation.w = quatRotations[qI+3];
		//quatRotation.x = quatRotations[qI];
		//quatRotation.y = quatRotations[qI+1];
		//quatRotation.z = quatRotations[qI+2];

		quatRotation[0] = quatRotations[qI];
		quatRotation[1] = quatRotations[qI+1];
		quatRotation[2] = quatRotations[qI+2];
		quatRotation[3] = quatRotations[qI+3];

		//if (vI == 0) std::cout << "quat " << quatRotation << "\n";

		// ensure its a unit quaternion
		quatRotation = glm::normalize(quatRotation);

		//if (vI == 0) std::cout << "unit quat " << quatRotation << "\n";

		// convert quaternion to euler angles
		eulerRotation = glm::eulerAngles(quatRotation);

		//if (vI == 0) std::cout << "euler " << eulerRotation << "\n";

		// make sure all euler rotations are in the range -pi to +pi
		eulerRotation.x = std::fmod(eulerRotation.x, glm::pi<float>());
		eulerRotation.y = std::fmod(eulerRotation.y, glm::pi<float>());
		eulerRotation.z = std::fmod(eulerRotation.z, glm::pi<float>());

		//if (vI == 0) std::cout << "euler mod " << eulerRotation << "\n";

		eulerRotations[eI] = eulerRotation.x;
		eulerRotations[eI+1] = eulerRotation.y;
		eulerRotations[eI+2] = eulerRotation.z;
	}

	//// debug begin
	//if (mName == "joint rot euler calc")
	//{
	//	std::cout << "DataProcQuatEuler " << mName << "\n";

	//	std::cout << "rec joint global rot\n";
	//	for (int jI = 0, dI = 0; jI < valueCount; ++jI, dI += 4)
	//	{
	//		//std::cout << "jI " << jI << " rot " << quatRotations[dI + 3] << " " << quatRotations[dI] << " " << quatRotations[dI + 1] << " " << quatRotations[dI + 2] << "\n";
	//		std::cout << "jI " << jI << " rot " << quatRotations[dI] << " " << quatRotations[dI+1] << " " << quatRotations[dI+2] << " " << quatRotations[dI+3] << "\n";

	//	}
	//}
	//// debug done
}
