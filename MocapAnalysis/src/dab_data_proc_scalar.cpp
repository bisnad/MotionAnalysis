/** \file dab_data_proc_scalar.cpp
*/

#include "dab_data_proc_scalar.h"
#include "dab_data.h"
#include <algorithm>
#include <math.h>
#include <float.h>

using namespace dab;

DataProcScalar::DataProcScalar()
	: DataProc()
{}
		
DataProcScalar::DataProcScalar(const std::string& pName, const std::string& pOutputDataName, ScalarMode pScalarMode)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
	, mScalarMode(pScalarMode)
{
	mDataScalar = std::shared_ptr<Data>(new Data(mOutputDataName, 1, 1));
	mData.push_back(mDataScalar);
}

void 
DataProcScalar::process() throw (dab::Exception)
{
	//std::cout << "DataProcScalar " << mName << " process\n";

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

	if (mDataScalar->valueCount() != valueCount)
	{
		mDataScalar->resize(1, valueCount);
	}

	const std::vector<float>& inputValues = inputData->values();
	std::vector<float>& scalarValues = mDataScalar->values();
	float _scalarValue = 0.0;

	if (mScalarMode == Average)
	{
		float _invDim = 1.0 / static_cast<float>(valueDim);

		for (int vI = 0, iI=0, oI=0; vI < valueCount; ++vI, ++oI)
		{
			_scalarValue = 0.0;

			for (int dI = 0; dI < valueDim; ++dI, iI++)
			{
				_scalarValue += inputValues[iI];
			}

			_scalarValue *= _invDim;

			scalarValues[oI] = _scalarValue;
		}
	}
	else if(mScalarMode == Max)
	{ 
		for (int vI = 0, iI = 0, oI = 0; vI < valueCount; ++vI, ++oI)
		{
			_scalarValue = 0.0;

			for (int dI = 0; dI < valueDim; ++dI, iI++)
			{
				_scalarValue = std::max(_scalarValue, fabs(inputValues[iI]));
			}

			scalarValues[oI] = _scalarValue;
		}
	}
	else if(mScalarMode == Min)
	{ 
		for (int vI = 0, iI = 0, oI = 0; vI < valueCount; ++vI, ++oI)
		{
			_scalarValue = FLT_MAX;

			for (int dI = 0; dI < valueDim; ++dI, iI++)
			{
				_scalarValue = std::min(_scalarValue, fabs(inputValues[iI]));
			}

			scalarValues[oI] = _scalarValue;
		}
	}

	//std::cout << "in " << inputValues[0] << " out " << scalarValues[0] << "\n";
}
