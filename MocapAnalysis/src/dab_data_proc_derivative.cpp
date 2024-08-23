/** \file dab_data_proc_derivative.cpp
*/

#include "dab_data_proc_derivative.h"
#include "dab_data.h"
#include "ofUtils.h"

using namespace dab;

DataProcDerivative::DataProcDerivative()
	: DataProc()
{}

DataProcDerivative::DataProcDerivative(const std::string& pName, const std::string& pDerivDataName)
	: DataProc(pName)
	, mDerivDataName(pDerivDataName)
{
	mDataDeriv = std::shared_ptr<Data>(new Data(mDerivDataName, 1, 1));
	mData.push_back(mDataDeriv);
}

void 
DataProcDerivative::process() throw (dab::Exception)
{
	//std::cout << "DataProcDerivative " << mName << " process\n";

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

	if (mDataT1 == nullptr) // create data 
	{
		mDataT1 = std::shared_ptr<Data>(new Data(inputDataName + "_T1", valueDim, valueCount));
		mDataT2 = std::shared_ptr<Data>(new Data(inputDataName + "_T2", valueDim, valueCount));
		mDataDeriv->resize(valueDim, valueCount);

		mDataT1->copyFrom(*inputData);
		mDataT2->copyFrom(*inputData);

		mProcTime = ofGetElapsedTimef();
		return;
	}

	if (mDataT1->valueDim() != valueDim) throw dab::Exception("DATA ERROR: value dim mismatch: expected " + std::to_string(mDataT1->valueDim()) + " received " + std::to_string(valueDim), __FILE__, __FUNCTION__, __LINE__);
	if (mDataT1->valueCount() != valueCount) throw dab::Exception("DATA ERROR: value count mismatch: expected " + std::to_string(mDataT1->valueCount()) + " received " + std::to_string(valueCount), __FILE__, __FUNCTION__, __LINE__);

	double currentTime = ofGetElapsedTimef();
	double timeDiff = currentTime - mProcTime;
	mProcTime = currentTime;

	mDataT2->copyFrom(*inputData);

	const std::vector<float>& valuesT1 = mDataT1->values();
	const std::vector<float>& valuesT2 = mDataT2->values();
	std::vector<float>& valuesDeriv = mDataDeriv->values();

	for (int i = 0; i < arraySize; ++i)
	{
		valuesDeriv[i] = valuesT2[i] - valuesT1[i];
		valuesDeriv[i] /= timeDiff;
	}

	//std::cout << "timeDiff " << timeDiff << "\n";
	//std::cout << "inputData " << *inputData << "\n";
	//std::cout << "data t1\n" << *mDataT1 << "\n";
	//std::cout << "data t2\n" << *mDataT2 << "\n";

	//std::cout << "DataProcDerivative " << mName << " derivative " << valuesDeriv[0] << "\n";

	//std::cout << "in1 " << valuesT1[0] << " in2 " << valuesT2[0] << " out " << valuesDeriv[0] << "\n";

	mDataT1->copyFrom(*mDataT2);
}