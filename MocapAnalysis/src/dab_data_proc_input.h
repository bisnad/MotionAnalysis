/** \file data_proc_input.h
*/

#pragma once

#include "dab_data_proc.h"

namespace dab
{
	class Data;

# pragma mark DataProcInput definition

	class DataProcInput : public DataProc
	{
	public:
		DataProcInput();
		DataProcInput(const std::string& pName, const std::vector<std::shared_ptr<Data>>& pData);
		DataProcInput(const DataProcInput& pDataProcInput);
		~DataProcInput();

		DataProcInput& operator=(const DataProcInput& pDataProcInput);
	};

};