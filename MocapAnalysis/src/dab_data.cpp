/** \file dab_mocap_data.cpp
*/

#include "dab_data.h"
#include "ofVectorMath.h"
#include <sstream>

using namespace dab;

# pragma mark Data implementation

Data::Data()
	: mName("")
{}

Data::Data(const std::string& pName, int pValueDim, int pValueCount)
	: mName(pName)
	, mValueDim(pValueDim)
	, mValueCount(pValueCount)
{
	mValues.resize(mValueDim * mValueCount, 0.0);

	//std::cout << "Data vdim " << mValueDim << " vcount " << mValueCount << " tcount " << mValues.size() << "\n";
}

Data::Data(const Data& pData)
	: mName(pData.mName)
	, mValueDim(pData.mValueDim)
	, mValueCount(pData.mValueCount)
	, mValues(pData.mValues)
{}

Data::~Data()
{}

Data&
Data::operator=(const Data& pData)
{
	mName = pData.mName;
	mValueDim = pData.mValueDim;
	mValueCount = pData.mValueCount;
	mValues = pData.mValues;

	return *this;
}

void 
Data::copyFrom(const Data& pData) throw (dab::Exception)
{
	if (pData.valueDim() != mValueDim) throw dab::Exception("DATA ERROR: value dim mismatch: expected " + std::to_string(mValueDim) + " received " + std::to_string(pData.valueDim()), __FILE__, __FUNCTION__, __LINE__);
	if (pData.valueCount() != mValueCount) throw dab::Exception("DATA ERROR: value count mismatch: expected " + std::to_string(mValueCount) + " received " + std::to_string(pData.valueCount()), __FILE__, __FUNCTION__, __LINE__);

	mValues = pData.values();
}

void 
Data::copyTo(Data& pData) const throw (dab::Exception)
{
	if (pData.valueDim() != mValueDim) throw dab::Exception("DATA ERROR: value dim mismatch: expected " + std::to_string(mValueDim) + " received " + std::to_string(pData.valueDim()), __FILE__, __FUNCTION__, __LINE__);
	if (pData.valueCount() != mValueCount) throw dab::Exception("DATA ERROR: value count mismatch: expected " + std::to_string(mValueCount) + " received " + std::to_string(pData.valueCount()), __FILE__, __FUNCTION__, __LINE__);

	pData.values() = mValues;
}

void 
Data::resize(int pValueDim, int pValueCount)
{
	mValueDim = pValueDim;
	mValueCount = pValueCount;
	mValues.resize(mValueDim * mValueCount, 0.0);
}

const std::string&
Data::name() const
{
	return mName;
}

int 
Data::valueDim() const
{
	return mValueDim;
}

int 
Data::valueCount() const
{
	return mValueCount;
}

std::vector<float>& 
Data::values()
{
	return mValues;
}

const std::vector<float>& 
Data::values() const
{
	return mValues;
}

Data::operator std::string() const
{
	std::stringstream ss;

	ss << "Data:\n";
	ss << "name: " << mName << "\n";
	ss << "valueDim: " << mValueDim << "\n";
	ss << "valueCount: " << mValueCount << "\n";

	ss << "values: [ ";
	for (int vI=0, i = 0; vI < mValueCount; ++vI)
	{
		ss << "[ ";
		for (int dI = 0; dI < mValueDim; ++dI, ++i)
		{ 
			ss << mValues[i] << " ";
		}
		ss << "] ";
	}
	ss << " ]\n";

	return ss.str();
}