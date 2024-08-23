/** \file dab_data_proc_weight_effort.cpp
*/

#include "dab_data_proc_weight_effort.h"
#include "dab_data.h"
#include <algorithm>

using namespace dab;

DataProcWeightEffort::DataProcWeightEffort()
	: DataProc()
	, mEfforts(0.0, 10)
{}

DataProcWeightEffort::DataProcWeightEffort(const std::string& pName, const std::string& pOutputDataName, unsigned int pWindowSize)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
	, mWindowSize(pWindowSize)
	, mEfforts(0.0, pWindowSize)
{
	mEffort = std::shared_ptr<Data>(new Data(mOutputDataName, 1, 1));
	mData.push_back(mEffort);
}

void 
DataProcWeightEffort::setWeights(const std::vector<float>& pWeights)
{
	mWeights = pWeights;
}

void 
DataProcWeightEffort::process() throw (dab::Exception)
{
	//std::cout << "DataProcWeightEffort " << mName << " process\n";

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

	if(valueDim != 1) return;

	if (mWeights.size() != valueCount)
	{
		std::cout << "Proc " << mName << " resize weights\n";
		mWeights.resize(valueCount, 1.0 / static_cast<float>(valueCount));
	}

	const std::vector<float>& scalarVelocities = inputData->values();

	float _effort = 0.0;
	for (int vI = 0; vI < valueCount; ++vI) _effort += mWeights[vI] * scalarVelocities[vI] * scalarVelocities[vI];

	mEfforts.update({ _effort });


	float _maxEffort = 0.0;
	_effort = 0.0;

	for (int i = 0; i < mWindowSize; ++i)
	{
		_effort = mEfforts[i];
		_maxEffort = std::max(_maxEffort, _effort);
	}

	mEffort->values()[0] = _maxEffort;

	//std::cout << "DataProcWeightEffort " << mName << " effort " << _maxEffort << "\n";
}
