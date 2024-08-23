/** \file dab_data_proc_space_effort_2.h
*/

#pragma once

#include "dab_data_proc.h"
#include "dab_ringbuffer.h"

namespace dab
{

	class DataProcSpaceEffort2 : public DataProc
	{
	public:
		DataProcSpaceEffort2();
		DataProcSpaceEffort2(const std::string& pName, const std::string& pOutputDataName, unsigned int pWindowSize);

		void setWeights(const std::vector<float>& pWeights);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		unsigned int mWindowSize;
		std::vector<float> mWeights;
		RingBuffer< std::vector<float> > mPositions;
		std::shared_ptr<Data> mEffort = nullptr;

		std::vector<float> process_vec3() throw (dab::Exception);
		std::vector<float> process_quat() throw (dab::Exception);
	};

};
