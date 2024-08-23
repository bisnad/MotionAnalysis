/** \file dab_data_proc_lowpass.h
*/

#pragma once

#include "dab_data_proc.h"

namespace dab
{
	class DataProcLowPass : public DataProc
	{
	public:
		DataProcLowPass();
		DataProcLowPass(const std::string& pName, const std::string& pOutputDataName, float pFactor);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		float mFactor;
		std::shared_ptr<Data> mDataLowPass = nullptr;
	};
};
