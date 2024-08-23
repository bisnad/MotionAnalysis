/** \file dab_data_proc_derivative.h
*/

#pragma once

#include "dab_data_proc.h"

namespace dab
{
	class DataProcDerivative : public DataProc
	{
	public:
		DataProcDerivative();
		DataProcDerivative(const std::string& pName, const std::string& pDerivDataName);

		void process() throw (dab::Exception);

	protected:
		std::string mDerivDataName;
		std::shared_ptr<Data> mDataT1 = nullptr;
		std::shared_ptr<Data> mDataT2 = nullptr;
		std::shared_ptr<Data> mDataDeriv = nullptr;
		double mProcTime;
	};

};
