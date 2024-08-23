/** \file dab_data_proc_lowpass.cpp
*/

#include "dab_data_proc_lowpass.h"
#include "dab_data.h"

using namespace dab;

DataProcLowPass::DataProcLowPass()
	: DataProc()
{}
	
DataProcLowPass::DataProcLowPass(const std::string& pName, const std::string& pOutputDataName, float pFactor)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
	, mFactor(pFactor)
{
	mDataLowPass = std::shared_ptr<Data>(new Data(mOutputDataName, 1, 1));
	mData.push_back(mDataLowPass);
}

void 
DataProcLowPass::process() throw (dab::Exception)
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
	int valueCount = inputData->valueCount();
	int arraySize = valueDim * valueCount;

	if (mDataLowPass->valueDim() != valueDim || mDataLowPass->valueCount() != valueCount)
	{
		mDataLowPass->resize(valueDim, valueCount);
	}

	const std::vector<float>& inputValues = inputData->values();
	std::vector<float>& lowPassValues = mDataLowPass->values();

	for (int i = 0; i < arraySize; ++i)
	{
		lowPassValues[i] = lowPassValues[i] * mFactor + inputValues[i] * (1.0 - mFactor);
	}

	//std::cout << "in " << inputValues[0] << " out " << lowPassValues[0] << "\n";
}
