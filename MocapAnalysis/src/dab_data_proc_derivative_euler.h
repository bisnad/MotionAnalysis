/** \file dab_data_proc_derivative_euler.h
This is class is mostly identical to dab_data_proc_derivative with the exception
of applying calculating modulo pi on the difference between previous and current rotations
*/

#pragma once

#include "dab_data_proc.h"

namespace dab
{
	class DataProcDerivativeEuler : public DataProc
	{
	public:
		DataProcDerivativeEuler();
		DataProcDerivativeEuler(const std::string& pName, const std::string& pDerivDataName);

		void process() throw (dab::Exception);

	protected:
		std::string mDerivDataName;
		std::shared_ptr<Data> mDataT1 = nullptr;
		std::shared_ptr<Data> mDataT2 = nullptr;
		std::shared_ptr<Data> mDataDeriv = nullptr;
		double mProcTime;
	};

};
