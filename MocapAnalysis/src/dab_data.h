/** \file dab_data.h
*/

#pragma once

#include <vector>
#include "dab_exception.h"

namespace dab
{
# pragma mark Data definition

	class Data
	{
	public:
		Data();
		Data(const std::string& pName, int pValueDim, int pValueCount);
		Data(const Data& pData);
		~Data();

		Data& operator=(const Data& pData);
		void copyFrom(const Data& pData) throw (dab::Exception);
		void copyTo(Data& pData) const throw (dab::Exception);

		void resize(int pValueDim, int pValueCount);

		const std::string& name() const;
		int valueDim() const;
		int valueCount() const;
		std::vector<float>& values();
		const std::vector<float>& values() const;

		operator std::string() const;

		friend std::ostream& operator << (std::ostream& pOstream, const Data& pData)
		{
			std::string info = pData;
			pOstream << info;
			return pOstream;
		};

	protected:
		std::string mName;
		int mValueDim;
		int mValueCount;
		std::vector<float> mValues;
	};

};