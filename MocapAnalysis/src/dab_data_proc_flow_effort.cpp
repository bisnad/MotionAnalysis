/** \file dab_data_proc_flow_effort.cpp
*/

#include "dab_data_proc_flow_effort.h"
#include "dab_data.h"

using namespace dab;

DataProcFlowEffort::DataProcFlowEffort()
	: DataProc()
	, mWindowSize(10)
	, mJerks({ 0.0 }, 10)
{}

DataProcFlowEffort::DataProcFlowEffort(const std::string& pName, const std::string& pOutputDataName, unsigned int pWindowSize)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
	, mWindowSize(10)
	, mJerks({ 0.0 }, 10)
{
	mEffort = std::shared_ptr<Data>(new Data(mOutputDataName, 1, 1));
	mData.push_back(mEffort);
}

void
DataProcFlowEffort::setWeights(const std::vector<float>& pWeights)
{
	mWeights = pWeights;
}

void
DataProcFlowEffort::process() throw (dab::Exception)
{
	//std::cout << "DataProcFlowEffort " << mName << " process\n";

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

	if (valueDim != 1) return;

	if (mWeights.size() != valueCount)
	{
		mWeights.resize(valueCount, 1.0 / static_cast<float>(valueCount));
	}

	if (mJerks[0].size() != valueCount)
	{
		int ringSize = mJerks.size();

		for (int i = 0; i < ringSize; ++i)
		{
			mJerks[i].resize(valueCount, 0.0);
		}
	}

	const std::vector<float>& scalarJerks = inputData->values();

	mJerks.update(scalarJerks);

	std::vector<float> flows(valueCount, 0.0);
	for (int wI = 0; wI < mWindowSize; ++wI)
	{
		for (int vI = 0; vI < valueCount; ++vI)
		{
			flows[vI] += mJerks[wI][vI];
		}
	}

	float invWindowSize = 1.0 / static_cast<float>(mWindowSize);
	float _effort = 0.0;
	for (int vI = 0; vI < valueCount; ++vI)
	{
		flows[vI] *= invWindowSize;
		_effort += flows[vI] * mWeights[vI];
	}

	mEffort->values()[0] = _effort;

	//std::cout << "DataProcFlowEffort " << mName << " effort " << _effort << "\n";
}
