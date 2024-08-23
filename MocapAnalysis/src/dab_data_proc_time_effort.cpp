/** \file dab_data_proc_time_effort.cpp
*/

#include "dab_data_proc_time_effort.h"
#include "dab_data.h"

using namespace dab;

DataProcTimeEffort::DataProcTimeEffort()
	: DataProc()
	, mWindowSize(10)
	, mAccelerations({0.0}, 10)
{}

DataProcTimeEffort::DataProcTimeEffort(const std::string& pName, const std::string& pOutputDataName, unsigned int pWindowSize)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
	, mWindowSize(10)
	, mAccelerations({ 0.0 }, 10)
{
	mEffort = std::shared_ptr<Data>(new Data(mOutputDataName, 1, 1));
	mData.push_back(mEffort);
}

void
DataProcTimeEffort::setWeights(const std::vector<float>& pWeights)
{
	mWeights = pWeights;
}

void 
DataProcTimeEffort::process() throw (dab::Exception)
{
	//std::cout << "DataProcTimeEffort " << mName << " process\n";

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
		std::cout << "Proc " << mName << " resize weights\n";
		mWeights.resize(valueCount, 1.0 / static_cast<float>(valueCount));
	}

	if (mAccelerations[0].size() != valueCount)
	{
		int ringSize = mAccelerations.size();

		for (int i = 0; i < ringSize; ++i)
		{
			mAccelerations[i].resize(valueCount, 0.0);
		}
	}

	const std::vector<float>& scalarAccelerations = inputData->values();

	mAccelerations.update(scalarAccelerations);

	std::vector<float> times( valueCount, 0.0 );
	for (int wI = 0; wI < mWindowSize; ++wI)
	{
		for (int vI = 0; vI < valueCount; ++vI)
		{
			times[vI] += mAccelerations[wI][vI];
		}
	}

	float invWindowSize = 1.0 / static_cast<float>(mWindowSize);
	float _effort = 0.0;
	for (int vI = 0; vI < valueCount; ++vI)
	{
		times[vI] *= invWindowSize;
		_effort += times[vI] * mWeights[vI];
	}

	mEffort->values()[0] = _effort;

	//std::cout << "DataProcTimeEffort " << mName << " effort " << _effort << "\n";
}
