/** \file data_proc_time_effort.h
*/

#pragma once

#include "dab_data_proc.h"
#include "dab_ringbuffer.h"

namespace dab
{

	class DataProcTimeEffort : public DataProc
	{
	public:
		DataProcTimeEffort();
		DataProcTimeEffort(const std::string& pName, const std::string& pOutputDataName, unsigned int pWindowSize);

		void setWeights(const std::vector<float>& pWeights);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		unsigned int mWindowSize;
		std::vector<float> mWeights;
		RingBuffer< std::vector<float> > mAccelerations;
		std::shared_ptr<Data> mEffort = nullptr;
	};

};