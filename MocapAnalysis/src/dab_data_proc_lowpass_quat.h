/** \file dab_data_proc_lowpass_quat.h
*/

#pragma once

#include "dab_data_proc.h"

namespace dab
{
	class DataProcLowPassQuat : public DataProc
	{
	public:
		DataProcLowPassQuat();
		DataProcLowPassQuat(const std::string& pName, const std::string& pOutputDataName, float pFactor);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		float mFactor;
		std::shared_ptr<Data> mDataLowPass = nullptr;
	};

};
