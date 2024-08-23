/** \file dab_data_proc_combine.cpp
*/

#include "dab_data_proc_combine.h"
#include "dab_data.h"

using namespace dab;

DataProcCombine::DataProcCombine()
	: DataProc()
{}

DataProcCombine::DataProcCombine(const std::string& pName, const std::string& pOutputDataName)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
{
	mDataCombined = std::shared_ptr<Data>(new Data(mOutputDataName, 1, 1));
	mData.push_back(mDataCombined);
}

void 
DataProcCombine::process() throw (dab::Exception)
{
	//std::cout << "DataProcCombine " << mName << " process\n";

	int inputProcCount = mInputProcs.size();

	if (inputProcCount == 0) return;

	if (mInputProcs[0]->data().size() == 0) return;

	int valueDim = mInputProcs[0]->data()[0]->valueDim();

	int totalValueCount = 0;

	for (int pI = 0; pI < inputProcCount; ++pI)
	{
		if (mInputProcs[pI]->data().size() == 0) throw dab::Exception("DATA ERROR: input proc " + mInputProcs[pI]->name() + " produces no outputs", __FILE__, __FUNCTION__, __LINE__);
		if (mInputProcs[pI]->data()[0]->valueDim() != valueDim) throw dab::Exception("DATA ERROR: input proc " + mInputProcs[pI]->name() + " first output has wrong value dimension ", __FILE__, __FUNCTION__, __LINE__);
	
		totalValueCount += mInputProcs[pI]->data()[0]->valueCount();
	}

	if (mDataCombined->valueDim() != valueDim || mDataCombined->valueCount() != totalValueCount)
	{
		mDataCombined->resize(valueDim, totalValueCount);
	}

	std::vector<float>& combinedValues = mDataCombined->values();

	for (int pI = 0, cI = 0; pI < inputProcCount; ++pI)
	{
		std::shared_ptr<Data> inputData = mInputProcs[pI]->data()[0];
		const std::vector<float>& inputValues = inputData->values();
		int valueCount = inputData->valueCount();

		for (int vI = 0, aI=0; vI < valueCount; ++vI)
		{
			for (int d = 0; d < valueDim; ++d, ++aI, ++cI)
			{
				combinedValues[cI] = inputValues[aI];
			}
		}
	}

	//std::cout << "in " << inputValues[0] << " out " << combinedValues[0] << "\n";
}
