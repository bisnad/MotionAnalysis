/** \file dab_data_proc_scalar.h
*/

#pragma once

#include "dab_data_proc.h"

namespace dab
{

	class DataProcScalar : public DataProc
	{
	public:

		enum ScalarMode
		{
			Average,
			Max,
			Min
		};

	public:
		DataProcScalar();
		DataProcScalar(const std::string& pName, const std::string& pOutputDataName, ScalarMode pScalarMode);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		ScalarMode mScalarMode;
		std::shared_ptr<Data> mDataScalar = nullptr;
	};

};
