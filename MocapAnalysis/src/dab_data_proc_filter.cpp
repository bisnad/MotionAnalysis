/** \file dab_data_proc_filter.cpp
*/

#include "dab_data_proc_filter.h"
#include "dab_data.h"
#include <algorithm>

using namespace dab;

DataProcFilter::DataProcFilter()
	: DataProc()
{}

DataProcFilter::DataProcFilter(const std::string& pName, const std::string& pOutputDataName, const std::vector<int>& pFilterIndices)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
	, mFilterIndices(pFilterIndices)
{
	std::sort(mFilterIndices.begin(), mFilterIndices.end());

	mDataFiltered = std::shared_ptr<Data>(new Data(mOutputDataName, 1, mFilterIndices.size()));
	mData.push_back(mDataFiltered);
}

void 
DataProcFilter::process() throw (dab::Exception)
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
	int filterCount = mFilterIndices.size();

	if (mDataFiltered->valueDim() != valueDim || mDataFiltered->valueCount() != filterCount)
	{
		mDataFiltered->resize(valueDim, filterCount);
	}

	const std::vector<float>& inputValues = inputData->values();
	std::vector<float>& filteredValues = mDataFiltered->values();

	int filterNr = 0;
	for (int vI = 0, aI=0, aI2=0; vI < valueCount; ++vI)
	{
		if (vI == mFilterIndices[filterNr])
		{
			for (int d = 0; d < valueDim; ++d, ++aI, ++aI2)
			{

				filteredValues[aI2] = inputValues[aI];
			}

			filterNr++;
		}
		else
		{
			aI += valueDim;
		}

	}

	//std::cout << "in " << inputValues[0] << " out " << filteredValues[0] << "\n";
}
