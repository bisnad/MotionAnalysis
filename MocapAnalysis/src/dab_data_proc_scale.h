/** \file dab_data_proc_scale.h
*/

#pragma once

#include "dab_data_proc.h"

namespace dab
{
	class DataProcScale: public DataProc
	{
	public:
		DataProcScale();
		DataProcScale(const std::string& pName, const std::string& pOutputDataName, float pValueScale);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		float mValueScale;
		std::shared_ptr<Data> mDataScaled = nullptr;
	};

};
