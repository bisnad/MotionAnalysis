/** \file dab_data_proc_space_effort_2.cpp
*/

#include "dab_data_proc_space_effort_2.h"
#include "dab_data.h"
#include "ofVectorMath.h"
#include <math.h>

using namespace dab;

DataProcSpaceEffort2::DataProcSpaceEffort2()
	: DataProc()
	, mWindowSize(10)
	, mPositions({ 0.0 }, 10)
{}

DataProcSpaceEffort2::DataProcSpaceEffort2(const std::string& pName, const std::string& pOutputDataName, unsigned int pWindowSize)
	: DataProc(pName)
	, mOutputDataName(pOutputDataName)
	, mWindowSize(10)
	, mPositions({ 0.0 }, 10)
{
	mEffort = std::shared_ptr<Data>(new Data(mOutputDataName, 1, 1));
	mData.push_back(mEffort);
}

void
DataProcSpaceEffort2::setWeights(const std::vector<float>& pWeights)
{
	mWeights = pWeights;
}

void
DataProcSpaceEffort2::process() throw (dab::Exception)
{
	if (mInputProcs.size() == 0) return;

	const std::vector< std::shared_ptr<Data> >& inputDataVector = mInputProcs[0]->data();

	if (inputDataVector.size() == 0) return;

	// take only the first data from potentially longer list of data
	// ignore the remaining data
	std::shared_ptr<Data> inputData = inputDataVector[0];

	const std::string& inputDataName = inputData->name();
	int valueDim = inputData->valueDim();
	int valueCount = inputData->valueCount();
	int arraySize = valueDim * valueCount;

	//std::cout << "DataProcSpaceEffort2::process arraySize " << arraySize << " valueDim " << valueDim << " valueCount " << valueCount << "\n";

	if (mWeights.size() != valueCount)
	{
		std::cout << "Proc " << mName << " resize weights\n";
		mWeights.resize(valueCount, 1.0 / static_cast<float>(valueCount));
	}

	if (mPositions[0].size() != arraySize)
	{
		int ringSize = mPositions.size();

		for (int i = 0; i < ringSize; ++i)
		{
			mPositions[i].resize(arraySize, 0.0);
		}
	}

	const std::vector<float>& positions = inputData->values();
	mPositions.update(positions);

	std::vector<float> spaces;  

	// works only with vec3 and quat (could also work in principle with vec2, but is not yet implemented)
	if (valueDim == 3)
	{
		try
		{
			spaces = process_vec3();
		}
		catch (dab::Exception& e)
		{
			throw e;
		}
	}
	else if (valueDim == 4)
	{
		try
		{
			spaces = process_quat();
		}
		catch (dab::Exception& e)
		{
			throw e;
		}
	}
	else
	{
		throw dab::Exception("Data Error: expected value dim of 3 or 4 but receive " + std::to_string(valueDim), __FILE__, __FUNCTION__, __LINE__);
	}

	float _effort = 0.0;

	for (int vI = 0; vI < valueCount; ++vI)
	{
		_effort += spaces[vI] * mWeights[vI];
	}


	mEffort->values()[0] = _effort;
}

//std::vector<float>
//DataProcSpaceEffort2::process_vec3() throw (dab::Exception)
//{
//	int arraySize = mPositions[0].size();
//	int valueDim = 3;
//	int valueCount = arraySize / valueDim;
//
//	//std::cout << "DataProcSpaceEffort2::process_vec3 arraySize " << arraySize << " valueDim " << valueDim << " valueCount " << valueCount << "\n";
//
//	std::vector<float> positions_now(arraySize);
//	std::vector<float> positions_start(arraySize);
//	std::vector<float> positions_t1(arraySize);
//	std::vector<float> positions_t2(arraySize);
//	std::vector<float> spaces(valueCount, 0.0);
//
//	glm::vec3 pos_now;
//	glm::vec3 pos_start;
//	glm::vec3 pos_start_now_dir;
//	glm::vec3 pos_t1;
//	glm::vec3 pos_t2;
//	glm::vec3 pos_t1_t2_dir;
//
//	float epsilon = 0.0001;
//
//	positions_now = mPositions[0];
//	positions_start = mPositions[mWindowSize - 1];
//
//	for (int wI = 1; wI < mWindowSize; ++wI)
//	{
//		positions_t2 = mPositions[wI];
//		positions_t1 = mPositions[wI - 1];
//
//		for (int vI = 0, aI=0; vI < valueCount; ++vI, aI += valueDim)
//		{
//			// get pos dir from start to end of window
//			for (int dI = 0; dI < valueDim; ++dI)
//			{
//				pos_now[dI] = positions_now[aI + dI];
//				pos_start[dI] = positions_start[aI + dI];
//			}
//
//			pos_start_now_dir = pos_now - pos_start;
//			if (glm::length(pos_start_now_dir) < epsilon) continue;
//			pos_start_now_dir = glm::normalize(pos_start_now_dir);
//
//			// get pos dir from current to next time step
//			for (int dI = 0; dI < valueDim; ++dI)
//			{
//				pos_t1[dI] = positions_t1[aI + dI];
//				pos_t2[dI] = positions_t2[aI + dI];
//			}
//
//			pos_t1_t2_dir = pos_t2 - pos_t1;
//			if (glm::length(pos_t1_t2_dir) < epsilon) continue;
//			pos_t1_t2_dir = glm::normalize(pos_t1_t2_dir);
//
//			float dir_dot = glm::dot(pos_start_now_dir, pos_t1_t2_dir);
//			float dir_deviation = ((dir_dot * -1.0) + 1.0) * 0.5;
//
//			//std::cout << "wI " << wI << " vI " << vI;
//			//std::cout << " sndir " << pos_start_now_dir[0] << " " << pos_start_now_dir[1] << " " << pos_start_now_dir[0];
//			//std::cout << " t12dir " << pos_t1_t2_dir[0] << " " << pos_t1_t2_dir[1] << " " << pos_t1_t2_dir[0];
//			//std::cout << " dot " << dir_dot << " dev " << dir_deviation << "\n";
//
//			spaces[vI] += dir_deviation;
//		}
//	}
//
//	float _invWindowSize = 1.0 / static_cast<float>(mWindowSize);
//
//	for (int vI = 0; vI < valueCount; ++vI)
//	{
//		spaces[vI] *= _invWindowSize;
//		//std::cout << "vI " << vI << " sp " << spaces[vI] << "\n";
//	}
//
//	return spaces;
//}
std::vector<float>
DataProcSpaceEffort2::process_vec3() throw (dab::Exception)
{
	int arraySize = mPositions[0].size();
	int valueDim = 3;
	int valueCount = arraySize / valueDim;

	//std::cout << "DataProcSpaceEffort2::process_vec3 arraySize " << arraySize << " valueDim " << valueDim << " valueCount " << valueCount << "\n";

	std::vector<float> positions_now(arraySize);
	std::vector<float> positions_start(arraySize);
	std::vector<float> positions_t1(arraySize);
	std::vector<float> positions_t2(arraySize);
	std::vector<float> spaces(valueCount, 0.0);

	glm::vec3 pos_now;
	glm::vec3 pos_start;
	std::vector<glm::vec3> pos_start_now_dir(valueCount);
	std::vector<float> pos_start_now_length(valueCount);
	glm::vec3 pos_t1;
	glm::vec3 pos_t2;
	glm::vec3 pos_t1_t2_dir;

	float epsilon = 0.0001;

	// get pos dir from start to end of window
	positions_now = mPositions[0];
	positions_start = mPositions[mWindowSize - 1];

	for (int vI = 0, aI = 0; vI < valueCount; ++vI)
	{
		// get pos dir from start to end of window
		for (int dI = 0; dI < valueDim; ++dI, ++aI)
		{
			pos_now[dI] = positions_now[aI];
			pos_start[dI] = positions_start[aI];
		}

		pos_start_now_dir[vI] = pos_now - pos_start;
		pos_start_now_length[vI] = glm::length(pos_start_now_dir[vI]);
		if (pos_start_now_length[vI] > epsilon)
		{
			pos_start_now_dir[vI] = glm::normalize(pos_start_now_dir[vI]);
		}
	}

	// get pos dir from current to next time step
	for (int wI = 1; wI < mWindowSize; ++wI)
	{
		positions_t2 = mPositions[wI];
		positions_t1 = mPositions[wI - 1];

		for (int vI = 0, aI = 0; vI < valueCount; ++vI, aI += valueDim)
		{
			// get pos dir from start to end of window
			if (pos_start_now_length[vI] < epsilon) continue;

			for (int dI = 0; dI < valueDim; ++dI)
			{
				pos_t1[dI] = positions_t1[aI + dI];
				pos_t2[dI] = positions_t2[aI + dI];
			}

			pos_t1_t2_dir = pos_t2 - pos_t1;
			float pos_t1_t2_length = glm::length(pos_t1_t2_dir);
			if (pos_t1_t2_length < epsilon) continue;
			pos_t1_t2_dir = glm::normalize(pos_t1_t2_dir);

			float dir_dot = glm::dot(pos_start_now_dir[vI], pos_t1_t2_dir);
			float dir_deviation = ((dir_dot * -1.0) + 1.0) * 0.5;

			//std::cout << "wI " << wI << " vI " << vI;
			//std::cout << " sndir " << pos_start_now_dir[0] << " " << pos_start_now_dir[1] << " " << pos_start_now_dir[0];
			//std::cout << " t12dir " << pos_t1_t2_dir[0] << " " << pos_t1_t2_dir[1] << " " << pos_t1_t2_dir[0];
			//std::cout << " dot " << dir_dot << " dev " << dir_deviation << "\n";

			//spaces[vI] += dir_deviation * pos_t1_t2_length;
			spaces[vI] += dir_deviation;
		}
	}

	float _invWindowSize = 1.0 / static_cast<float>(mWindowSize);

	for (int vI = 0; vI < valueCount; ++vI)
	{
		spaces[vI] *= pos_start_now_length[vI];
		spaces[vI] *= _invWindowSize;
		//std::cout << "vI " << vI << " sp " << spaces[vI] << "\n";
	}

	return spaces;
}


std::vector<float>
DataProcSpaceEffort2::process_quat() throw (dab::Exception)
{
	int arraySize = mPositions[0].size();
	int valueDim = 4;
	int valueCount = arraySize / valueDim;

	//std::cout << "DataProcSpaceEffort2::process_vec3 arraySize " << arraySize << " valueDim " << valueDim << " valueCount " << valueCount << "\n";

	std::vector<float> positions_now(arraySize);
	std::vector<float> positions_start(arraySize);
	std::vector<float> positions_t1(arraySize);
	std::vector<float> positions_t2(arraySize);
	std::vector<float> spaces(valueCount, 0.0);

	glm::quat pos_now;
	glm::quat pos_start;
	std::vector<glm::quat> pos_start_now_dir(valueCount);
	std::vector<float> pos_start_now_length(valueCount);
	glm::quat pos_t1;
	glm::quat pos_t2;
	glm::quat pos_t1_t2_dir;

	float epsilon = 0.0001;

	// get pos dir from start to end of window
	positions_now = mPositions[0];
	positions_start = mPositions[mWindowSize - 1];

	for (int vI = 0, aI = 0; vI < valueCount; ++vI)
	{
		// get pos dir from start to end of window
		for (int dI = 0; dI < valueDim; ++dI, ++aI)
		{
			pos_now[dI] = positions_now[aI];
			pos_start[dI] = positions_start[aI];
		}

		pos_start_now_dir[vI] = pos_now - pos_start;
		pos_start_now_length[vI] = glm::length(pos_start_now_dir[vI]);
		if (pos_start_now_length[vI] > epsilon)
		{
			pos_start_now_dir[vI] = glm::normalize(pos_start_now_dir[vI]);
		}
	}

	// get pos dir from current to next time step
	for (int wI = 1; wI < mWindowSize; ++wI)
	{
		positions_t2 = mPositions[wI];
		positions_t1 = mPositions[wI - 1];

		for (int vI = 0, aI = 0; vI < valueCount; ++vI, aI += valueDim)
		{
			// get pos dir from start to end of window
			if (pos_start_now_length[vI] < epsilon) continue;

			for (int dI = 0; dI < valueDim; ++dI)
			{
				pos_t1[dI] = positions_t1[aI + dI];
				pos_t2[dI] = positions_t2[aI + dI];
			}

			pos_t1_t2_dir = pos_t2 - pos_t1;
			float pos_t1_t2_length = glm::length(pos_t1_t2_dir);
			if (pos_t1_t2_length < epsilon) continue;
			pos_t1_t2_dir = glm::normalize(pos_t1_t2_dir);

			float dir_dot = glm::dot(pos_start_now_dir[vI], pos_t1_t2_dir);
			float dir_deviation = ((dir_dot * -1.0) + 1.0) * 0.5;

			//std::cout << "wI " << wI << " vI " << vI;
			//std::cout << " sndir " << pos_start_now_dir[0] << " " << pos_start_now_dir[1] << " " << pos_start_now_dir[0];
			//std::cout << " t12dir " << pos_t1_t2_dir[0] << " " << pos_t1_t2_dir[1] << " " << pos_t1_t2_dir[0];
			//std::cout << " dot " << dir_dot << " dev " << dir_deviation << "\n";

			spaces[vI] += dir_deviation * pos_t1_t2_length;
		}
	}

	float _invWindowSize = 1.0 / static_cast<float>(mWindowSize);

	for (int vI = 0; vI < valueCount; ++vI)
	{
		spaces[vI] *= pos_start_now_length[vI];
		spaces[vI] *= _invWindowSize;
		//std::cout << "vI " << vI << " sp " << spaces[vI] << "\n";
	}

	return spaces;
}