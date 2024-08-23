/** \file dab_data_proc_scale.cpp
*/

#include "dab_data_proc_scale.h"
#include "dab_data.h"

using namespace dab;

DataProcScale::DataProcScale()
	: DataProc()
{}

DataProcScale::DataProcScale(const std::string& pName, const std::string& pOutputDataName, float pValueScale)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
	, mValueScale(pValueScale)
{
	mDataScaled = std::shared_ptr<Data>(new Data(mOutputDataName, 1, 1));
	mData.push_back(mDataScaled);
}

void
DataProcScale::process() throw (dab::Exception)
{
	//std::cout << "DataProcScale " << mName << " process\n";

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

	if (mDataScaled->valueDim() != valueDim || mDataScaled->valueCount() != valueCount)
	{
		mDataScaled->resize(valueDim, valueCount);
	}

	const std::vector<float>& inputValues = inputData->values();
	std::vector<float>& scaledValues = mDataScaled->values();

	for (int i = 0; i < arraySize; ++i)
	{
		scaledValues[i] = inputValues[i] * mValueScale;
	}

	//std::cout << "in " << inputValues[0] << " out " << scaledValues[0] << "\n";
}