/** \file dab_data_proc_combine.h
*/

#pragma once

#include "dab_data_proc.h"

namespace dab
{
	class DataProcCombine : public DataProc
	{
	public:
		DataProcCombine();
		DataProcCombine(const std::string& pName, const std::string& pOutputDataName);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		std::shared_ptr<Data> mDataCombined = nullptr;
	};
};
