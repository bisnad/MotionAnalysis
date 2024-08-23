/** \file dab_data_proc_pos_global_to_local.h
*/

#pragma once

#include "dab_data_proc.h"
#include "ofVectorMath.h"

namespace dab
{

	class DataProcPosGlobalToLocal : public DataProc
	{
	public:
		DataProcPosGlobalToLocal();
		DataProcPosGlobalToLocal(const std::string& pName, const std::string& pOutputDataName, const std::vector< std::vector<int> >& pJointConnectivity);

		void process() throw (dab::Exception);

	protected:
		std::string mOutputDataName;
		std::shared_ptr<Data> mDataPositionsLocal = nullptr;

		std::vector< std::vector<int> > mJointConnectivity;
		std::vector<glm::vec3> mJointPositionsGlobal;
		std::vector<glm::vec3> mJointPositionsLocal;

		void calcLocalPosition(const std::vector<int>& pChildJointIndices, const glm::vec3& pParentPosition);
	};

};