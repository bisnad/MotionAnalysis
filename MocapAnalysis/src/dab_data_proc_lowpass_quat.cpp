/** \file dab_data_proc_lowpass_quat.cpp
*/

#include "dab_data_proc_lowpass_quat.h"
#include "dab_data.h"
#include "ofVectorMath.h"

using namespace dab;

DataProcLowPassQuat::DataProcLowPassQuat()
	: DataProc()
{}

DataProcLowPassQuat::DataProcLowPassQuat(const std::string& pName, const std::string& pOutputDataName, float pFactor)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
	, mFactor(pFactor)
{
	mDataLowPass = std::shared_ptr<Data>(new Data(mOutputDataName, 4, 1));
	mData.push_back(mDataLowPass);
}

void
DataProcLowPassQuat::process() throw (dab::Exception)
{
	//std::cout << "DataProcLowPass " << mName << " process\n";

	if (mInputProcs.size() == 0) return;

	const std::vector< std::shared_ptr<Data> >& inputDataVector = mInputProcs[0]->data();

	if (inputDataVector.size() == 0) return;

	// take only the first data from potentially longer list of data
	// ignore the remaining data
	std::shared_ptr<Data> inputData = inputDataVector[0];

	const std::string& inputDataName = inputData->name();
	int valueDim = inputData->valueDim();

	if (valueDim != 4) throw dab::Exception("DATA ERROR: value dim mismatch: expected 4 received " + std::to_string(valueDim), __FILE__, __FUNCTION__, __LINE__);

	int valueCount = inputData->valueCount();
	int arraySize = valueDim * valueCount;

	if (mDataLowPass->valueDim() != valueDim || mDataLowPass->valueCount() != valueCount)
	{
		mDataLowPass->resize(valueDim, valueCount);
	}

	const std::vector<float>& inputValues = inputData->values();
	std::vector<float>& lowPassValues = mDataLowPass->values();

	glm::quat currentQuat;
	glm::quat newQuat;

	for (int vI = 0, aI = 0; vI < valueCount; ++vI, aI += 4)
	{
		//glm::quat currentQuat(lowPassValues[aI], lowPassValues[aI+1], lowPassValues[aI+2], lowPassValues[aI+3]);
		//glm::quat newQuat(inputValues[aI], inputValues[aI + 1], inputValues[aI + 2], inputValues[aI + 3]);

		currentQuat[0] = lowPassValues[aI];
		currentQuat[1] = lowPassValues[aI+1];
		currentQuat[2] = lowPassValues[aI+2];
		currentQuat[3] = lowPassValues[aI+3];
		
		newQuat[0] = inputValues[aI];
		newQuat[1] = inputValues[aI+1];
		newQuat[2] = inputValues[aI+2];
		newQuat[3] = inputValues[aI+3];
		
		currentQuat = glm::normalize(currentQuat);
		newQuat = glm::normalize(newQuat);
		
		glm::quat lowPassQuat = glm::mix(currentQuat, newQuat, static_cast<float>(1.0 - mFactor));

		lowPassQuat = glm::normalize(lowPassQuat);

		//lowPassValues[aI] = lowPassQuat.w;
		//lowPassValues[aI+1] = lowPassQuat.x;
		//lowPassValues[aI+2] = lowPassQuat.y;
		//lowPassValues[aI+3] = lowPassQuat.z;

		lowPassValues[aI] = lowPassQuat[0];
		lowPassValues[aI + 1] = lowPassQuat[1];
		lowPassValues[aI + 2] = lowPassQuat[2];
		lowPassValues[aI + 3] = lowPassQuat[3];
	}
}
