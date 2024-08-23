/** \file dab_data_proc_filter.h
*/

#pragma once

#include "dab_data_proc.h"

namespace dab
{

	class DataProcFilter : public DataProc
	{
	public:
		DataProcFilter();
		DataProcFilter(const std::string& pName, const std::string& pOutputDataName, const std::vector<int>& pFilterIndices);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		std::vector<int> mFilterIndices;
		std::shared_ptr<Data> mDataFiltered = nullptr;
	};

};