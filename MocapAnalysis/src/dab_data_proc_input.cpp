/** \file data_proc_pipeline.cpp
*/

#include "dab_data_proc_input.h"
#include "dab_data.h"

using namespace dab;

# pragma mark DataProcInput implementation

DataProcInput::DataProcInput()
	: DataProc()
{}

DataProcInput::DataProcInput(const std::string& pName, const std::vector<std::shared_ptr<Data>>& pData)
	: DataProc(pName)
{
	mData = pData;
}

DataProcInput::DataProcInput(const DataProcInput& pDataProcInput)
	: DataProc(pDataProcInput.mName)
{
	mData = pDataProcInput.mData;
}

DataProcInput::~DataProcInput()
{}


DataProcInput& 
DataProcInput::operator=(const DataProcInput& pDataProcInput)
{
	DataProc::operator=(pDataProcInput);

	return *this;
}