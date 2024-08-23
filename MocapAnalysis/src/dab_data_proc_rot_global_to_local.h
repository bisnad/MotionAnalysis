/** \file dab_data_proc_rot_global_to_local.h
*/

#pragma once

#include "dab_data_proc.h"
#include "ofVectorMath.h"

namespace dab
{

	class DataProcRotGlobalToLocal : public DataProc
	{
	public:
		DataProcRotGlobalToLocal();
		DataProcRotGlobalToLocal(const std::string& pName, const std::string& pOutputDataName, const std::vector< std::vector<int> >& pJointConnectivity);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		std::shared_ptr<Data> mDataRotationsLocal = nullptr;

		std::vector< std::vector<int> > mJointConnectivity;
		std::vector<glm::quat> mJointRotationsGlobal;
		std::vector<glm::quat> mJointRotationsLocal;

		void calcLocalRotation(const std::vector<int>& pChildJointIndices, const glm::quat& pParentRotation);
	};

};