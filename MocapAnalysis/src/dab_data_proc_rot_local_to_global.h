/** \file dab_data_proc_rot_local_to_global.h
*/

#pragma once

#include "dab_data_proc.h"
#include "ofVectorMath.h"

namespace dab
{

	class DataProcRotLocalToGlobal : public DataProc
	{
	public:
		DataProcRotLocalToGlobal();
		DataProcRotLocalToGlobal(const std::string& pName, const std::string& pOutputDataName, const std::vector< std::vector<int> >& pJointConnectivity);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		std::shared_ptr<Data> mDataRotationsGlobal = nullptr;

		std::vector< std::vector<int> > mJointConnectivity;
		std::vector<glm::quat> mJointRotationsLocal;
		std::vector<glm::quat> mJointRotationsGlobal;

		void calcGlobalRotation(const std::vector<int>& pChildJointIndices, const glm::quat& pParentRotation);
	};

};