/** \file dab_data_proc_quat_euler.h
*/

#pragma once

#include "dab_data_proc.h"

namespace dab
{

	class DataProcQuatEuler: public DataProc
	{
	public:
		DataProcQuatEuler();
		DataProcQuatEuler(const std::string& pName, const std::string& pOutputDataName);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		std::shared_ptr<Data> mDataEuler = nullptr;
	};

};