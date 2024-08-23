/** \file dab_data_proc_flow_effort.h
*/

#pragma once

#include "dab_data_proc.h"
#include "dab_ringbuffer.h"

namespace dab
{

	class DataProcFlowEffort : public DataProc
	{
	public:
		DataProcFlowEffort();
		DataProcFlowEffort(const std::string& pName, const std::string& pOutputDataName, unsigned int pWindowSize);

		void setWeights(const std::vector<float>& pWeights);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		unsigned int mWindowSize;
		std::vector<float> mWeights;
		RingBuffer< std::vector<float> > mJerks;
		std::shared_ptr<Data> mEffort = nullptr;
	};

};