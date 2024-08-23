/** \file dab_data_proc_space_effort.cpp
*/

#include "dab_data_proc_space_effort.h"
#include "dab_data.h"
#include <math.h>

using namespace dab;

DataProcSpaceEffort::DataProcSpaceEffort()
	: DataProc()
	, mWindowSize(10)
	, mPositions({ 0.0 }, 10)
{}

DataProcSpaceEffort::DataProcSpaceEffort(const std::string& pName, const std::string& pOutputDataName, unsigned int pWindowSize)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
	, mWindowSize(10)
	, mPositions({ 0.0 }, 10)
{
	mEffort = std::shared_ptr<Data>(new Data(mOutputDataName, 1, 1));
	mData.push_back(mEffort);
}

void
DataProcSpaceEffort::setWeights(const std::vector<float>& pWeights)
{
	mWeights = pWeights;
}

void
DataProcSpaceEffort::process() throw (dab::Exception)
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

	if (mWeights.size() != valueCount)
	{
		std::cout << "Proc " << mName << " resize weights\n";
		mWeights.resize(valueCount, 1.0 / static_cast<float>(valueCount));
	}

	if (mPositions[0].size() != arraySize)
	{
		int ringSize = mPositions.size();

		for (int i = 0; i < ringSize; ++i)
		{
			mPositions[i].resize(arraySize, 0.0);
		}
	}

	const std::vector<float>& positions = inputData->values();

	mPositions.update(positions);

	std::vector<float> spaces(valueCount, 0.0);
	std::vector<float> positions_now(arraySize);
	std::vector<float> positions_start(arraySize);
	std::vector<float> positions_ti(arraySize);
	std::vector<float> positions_ti_1(arraySize);
	std::vector<float> pos1(valueDim);
	std::vector<float> pos2(valueDim);
	float dist1;
	float dist2;

	positions_now = mPositions[0];
	positions_start = mPositions[mWindowSize - 1];

	for (int wI = 1; wI < mWindowSize; ++wI)
	{
		positions_ti = mPositions[wI];
		positions_ti_1 = mPositions[wI - 1];

		for (int vI = 0, aI = 0; vI < valueCount; ++vI)
		{
			for (int dI = 0; dI < valueDim; ++dI, ++aI)
			{
				pos1[dI] = positions_ti[aI];
				pos2[dI] = positions_ti_1[aI];
			}

			dist1 = 0.0;

			for (int dI = 0; dI < valueDim; ++dI)
			{
				dist1 += (pos1[dI] - pos2[dI])*(pos1[dI] - pos2[dI]);
			}

			dist1 = sqrt(dist1);
			spaces[vI] += dist1;
		}
	}

	for (int vI = 0, aI = 0; vI < valueCount; ++vI)
	{
		dist2 = 0.0;

		for (int dI = 0; dI < valueDim; ++dI, ++aI)
		{
			pos1[dI] = positions_now[aI];
			pos2[dI] = positions_start[aI];

			dist2 += (pos1[dI] - pos2[dI])*(pos1[dI] - pos2[dI]);
		}

		dist2 = sqrt(dist2);
		if (dist2 > 0.0001) spaces[vI] /= dist2;
		else spaces[vI] = 0.0;
	}

	float _effort = 0.0;

	for (int vI = 0; vI < valueCount; ++vI)
	{
		_effort += spaces[vI] * mWeights[vI];
	}


	mEffort->values()[0] = _effort;
}
