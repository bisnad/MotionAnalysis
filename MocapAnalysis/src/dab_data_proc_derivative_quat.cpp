/** \file dab_data_proc_derivative_quat.cpp
*/

#include "dab_data_proc_derivative_quat.h"
#include "dab_data.h"
#include "ofUtils.h"
#include "ofVectorMath.h"

using namespace dab;

DataProcDerivativeQuat::DataProcDerivativeQuat()
	: DataProc()
{}

DataProcDerivativeQuat::DataProcDerivativeQuat(const std::string& pName, const std::string& pDerivDataName)
	: DataProc(pName)
	, mDerivDataName(pDerivDataName)
{
	mDataDeriv = std::shared_ptr<Data>(new Data(mDerivDataName, 1, 1));
	mData.push_back(mDataDeriv);
}

void
DataProcDerivativeQuat::process() throw (dab::Exception)
{
	//std::cout << "DataProcDerivative " << mName << " process\n";

	if (mInputProcs.size() == 0) return;

	const std::vector< std::shared_ptr<Data> >& inputDataVector = mInputProcs[0]->data();

	if (inputDataVector.size() == 0) return;

	// take only the first data from potentially longer list of data
	// ignore the remaining data
	std::shared_ptr<Data> inputData = inputDataVector[0];

	const std::string& inputDataName = inputData->name();
	int valueDim = inputData->valueDim();

	if(valueDim != 4) throw dab::Exception("DATA ERROR: value dim mismatch: expected 4 received " + std::to_string(valueDim), __FILE__, __FUNCTION__, __LINE__);

	int valueCount = inputData->valueCount();
	int arraySize = valueDim * valueCount;

	if (mDataT1 == nullptr) // create data 
	{
		mDataT1 = std::shared_ptr<Data>(new Data(inputDataName + "_T1", valueDim, valueCount));
		mDataT2 = std::shared_ptr<Data>(new Data(inputDataName + "_T2", valueDim, valueCount));
		mDataDeriv->resize(valueDim, valueCount);

		mDataT1->copyFrom(*inputData);
		mDataT2->copyFrom(*inputData);

		mProcTime = ofGetElapsedTimef();
		return;
	}

	if (mDataT1->valueDim() != valueDim) throw dab::Exception("DATA ERROR: value dim mismatch: expected " + std::to_string(mDataT1->valueDim()) + " received " + std::to_string(valueDim), __FILE__, __FUNCTION__, __LINE__);
	if (mDataT1->valueCount() != valueCount) throw dab::Exception("DATA ERROR: value count mismatch: expected " + std::to_string(mDataT1->valueCount()) + " received " + std::to_string(valueCount), __FILE__, __FUNCTION__, __LINE__);

	double currentTime = ofGetElapsedTimef();
	double timeDiff = currentTime - mProcTime;

	mProcTime = currentTime;

	mDataT2->copyFrom(*inputData);

	const std::vector<float>& valuesT1 = mDataT1->values();
	const std::vector<float>& valuesT2 = mDataT2->values();
	std::vector<float>& valuesDeriv = mDataDeriv->values();

	glm::quat quat1;
	glm::quat quat2;

	for (int vI = 0, aI = 0; vI < valueCount; ++vI, aI += valueDim)
	{
		//glm::quat quat1(valuesT1[aI], valuesT1[aI+1], valuesT1[aI+2], valuesT1[aI+3]);
		//glm::quat quat2(valuesT2[aI], valuesT2[aI + 1], valuesT2[aI + 2], valuesT2[aI + 3]);

		quat1[0] = valuesT1[aI];
		quat1[1] = valuesT1[aI + 1];
		quat1[2] = valuesT1[aI + 2];
		quat1[3] = valuesT1[aI + 3];

		quat2[0] = valuesT2[aI];
		quat2[1] = valuesT2[aI + 1];
		quat2[2] = valuesT2[aI + 2];
		quat2[3] = valuesT2[aI + 3];

		quat1 = glm::normalize(quat1);
		quat2 = glm::normalize(quat2);

		//Rot Diff based on slerp begin
		float scaledTimeDiff = timeDiff;
		scaledTimeDiff *= 10.0; // hacky: scale time diff to avoid diff rotations covering more than 360 degrees (which can't be handled by quaternions)
		glm::quat timeQuat2 = glm::normalize(glm::mix(quat1, quat2, 1.0f / (float)scaledTimeDiff));
		glm::quat quat1inv = glm::inverse(quat1);
		//glm::quat quatdiff = timeQuat2 * quat1inv;
		glm::quat quatdiff = glm::normalize(quat1inv * timeQuat2);
		//Rot Diff based on slerp end

		////Rot Diff based on conjugate begin (https://math.stackexchange.com/questions/160908/how-to-get-angular-velocity-from-difference-orientation-quaternion-and-time)
		//glm::quat quatdiff = quat2 - quat1;
		//glm::quat quat1conj = glm::conjugate(quat1);
		//quatdiff[0] = quatdiff[0] * 2.0 / (float)timeDiff * quat1conj[0];
		//quatdiff[1] = quatdiff[1] * 2.0 / (float)timeDiff * quat1conj[1];
		//quatdiff[2] = quatdiff[2] * 2.0 / (float)timeDiff * quat1conj[2];
		//quatdiff[3] = quatdiff[3] * 2.0 / (float)timeDiff * quat1conj[3];
		////Rot Diff based on conjugate end

		////Rot Diff based on angle/axis begin  (https://answers.unity.com/questions/49082/rotation-quaternion-to-angular-velocity.html)
		//glm::quat quatdiff = glm::normalize(quat2 * glm::inverse(quat1));
		//float rotAngle = glm::angle(quatdiff);
		//glm::vec3 rotAxis = glm::axis(quatdiff);
		//rotAngle /= (float)timeDiff;
		//quatdiff = angleAxis(rotAngle, rotAxis);
		//quatdiff = glm::normalize(quatdiff);
		////Rot Diff based on angle/axis end  

		//if (mName == "joint rot quat velocity calc")
		//{
		//	//std::cout << "rotAngle " << rotAngle << "\n";
		//	//std::cout << "rotAxis " << rotAxis[0] << " " << rotAxis[1] << " " << rotAxis[2] << "\n";
		//	std::cout << "quatdiff " << quatdiff[0] << " " << quatdiff[1] << " " << quatdiff[2] << " " << quatdiff[3] << "\n";
		//}
		//if (mName == "joint rot quat jerk calc")
		//{
		//	//std::cout << "rotAngle " << rotAngle << "\n";
		//	//std::cout << "rotAxis " << rotAxis[0] << " " << rotAxis[1] << " " << rotAxis[2] << "\n";
		//	std::cout << "quatdiff " << quatdiff[0] << " " << quatdiff[1] << " " << quatdiff[2] << " " << quatdiff[3] << "\n";
		//}


		valuesDeriv[aI] = quatdiff[0];
		valuesDeriv[aI + 1] = quatdiff[1];
		valuesDeriv[aI + 2] = quatdiff[2];
		valuesDeriv[aI + 3] = quatdiff[3];

		//if (vI == 0)
		//{
		//	std::cout << "quat1 " << quat1.w << " " << quat1.x << " " << quat1.y << " " << quat1.z << "\n";
		//	std::cout << "quat2 " << quat2.w << " " << quat2.x << " " << quat2.y << " " << quat2.z << "\n";
		//	std::cout << "timeDiff " << timeDiff << "\n";
		//	std::cout << "timeQuat2 " << timeQuat2.w << " " << timeQuat2.x << " " << timeQuat2.y << " " << timeQuat2.z << "\n";
		//	std::cout << "quat1inv " << quat1inv.w << " " << quat1inv.x << " " << quat1inv.y << " " << quat1inv.z << "\n";
		//	std::cout << "quatdiff " << quatdiff.w << " " << quatdiff.x << " " << quatdiff.y << " " << quatdiff.z << "\n";
		//}
	}

	//std::cout << "timeDiff " << timeDiff << "\n";
	//std::cout << "inputData " << *inputData << "\n";
	//std::cout << "data t1\n" << *mDataT1 << "\n";
	//std::cout << "data t2\n" << *mDataT2 << "\n";

	//std::cout << "DataProcDerivative " << mName << " derivative " << valuesDeriv[0] << "\n";

	//std::cout << "in1 " << valuesT1[0] << " in2 " << valuesT2[0] << " out " << valuesDeriv[0] << "\n";

	mDataT1->copyFrom(*mDataT2);
}