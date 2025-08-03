#include "ofApp.h"
#include "dab_data_sender.h"
#include "dab_file_io.h"
#include "dab_json_helper.h"
#include "ofVectorMath.h"

//--------------------------------------------------------------
void ofApp::setup()
{
	try
	{
		loadConfig(ofToDataPath("config.json"));
        
		setupData();
		setupDataProc();
		setupOsc();
		setupGraphics();
		setupGui();

		ofSetWindowTitle("Mocap Analysis");
	}
	catch (dab::Exception& e)
	{
		std::cout << e << "\n";
	}
}

//--------------------------------------------------------------
void ofApp::update()
{
	try
	{
		updateDataProc();

		mPlotMenu->update();
		mOSCMenu->update();
	}
	catch (dab::Exception& e)
	{
		std::cout << e << "\n";
	}
}

//--------------------------------------------------------------
void ofApp::draw()
{
	ofClear(255.0, 255.0, 255.0);
	if(mActivePlotIndex != -1 && mDataPlots[mActivePlotIndex]->isSetup() == true) mDataPlots[mActivePlotIndex]->draw();

	mGui->draw();
	//mPlotMenu->draw();
	//mOSCMenu->draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
	mActivePlotIndex = (mActivePlotIndex + 1) % mDataPlots.size();
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}

void
ofApp::loadConfig(const std::string& pFileName) throw (dab::Exception)
{
	try
	{
		std::string restoreString;
		dab::FileIO::get().read(pFileName, restoreString);

		Json::Reader reader;
		Json::Value restoreData;
		dab::JsonHelper& jsonHelper = dab::JsonHelper::get();

		bool parsingSuccessful = reader.parse(restoreString, restoreData);

		if (parsingSuccessful == false) throw dab::Exception("FILE ERROR: failed to parse config data file " + pFileName, __FILE__, __FUNCTION__, __LINE__);

		mOscReceivePort = jsonHelper.getInt(restoreData, "oscReceivePort");
		mOscSendAddress = jsonHelper.getString(restoreData, "oscSendAddress");
		mOscSendPort = jsonHelper.getInt(restoreData, "oscSendPort");

		std::string jointBodyPartFiltersFileName = jsonHelper.getString(restoreData, "jointFilters");
		std::string jointWeightsFileName = jsonHelper.getString(restoreData, "jointWeights");
		std::string jointConnectivityFileName = jsonHelper.getString(restoreData, "jointConnectivity");

		loadJointBodyPartFilters(ofToDataPath(jointBodyPartFiltersFileName));
		loadJointWeights(ofToDataPath(jointWeightsFileName));
		loadJointConnectivity(ofToDataPath(jointConnectivityFileName));
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("JSON ERROR: failed to restore config from file " + pFileName, __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void
ofApp::loadJointBodyPartFilters(const std::string& pFileName) throw (dab::Exception)
{
	try
	{
		std::string restoreString;
		dab::FileIO::get().read(pFileName, restoreString);

		Json::Reader reader;
		Json::Value restoreData;
		dab::JsonHelper& jsonHelper = dab::JsonHelper::get();

		bool parsingSuccessful = reader.parse(restoreString, restoreData);

		if (parsingSuccessful == false) throw dab::Exception("FILE ERROR: failed to parse config data file " + pFileName, __FILE__, __FUNCTION__, __LINE__);

		Json::Value torsoJointIndexData = jsonHelper.getValue(restoreData, "torsoJointIndices");
		Json::Value leftArmJointIndexData = jsonHelper.getValue(restoreData, "leftArmJointIndices");
		Json::Value rightArmJointIndexData = jsonHelper.getValue(restoreData, "rightArmJointIndices");
		Json::Value leftLegJointIndexData = jsonHelper.getValue(restoreData, "leftLegJointIndices");
		Json::Value rightLegJointIndexData = jsonHelper.getValue(restoreData, "rightLegJointIndices");

		jsonHelper.getInts(torsoJointIndexData, mTorsoJointIndices);
		jsonHelper.getInts(leftArmJointIndexData, mLeftArmJointIndices);
		jsonHelper.getInts(rightArmJointIndexData, mRightArmJointIndices);
		jsonHelper.getInts(leftLegJointIndexData, mLeftLegJointIndices);
		jsonHelper.getInts(rightLegJointIndexData, mRightLegJointIndices);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("JSON ERROR: failed to restore config from file " + pFileName, __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void
ofApp::loadJointWeights(const std::string& pFileName) throw (dab::Exception)
{
	try
	{
		std::string restoreString;
		dab::FileIO::get().read(pFileName, restoreString);

		Json::Reader reader;
		Json::Value restoreData;
		dab::JsonHelper& jsonHelper = dab::JsonHelper::get();

		bool parsingSuccessful = reader.parse(restoreString, restoreData);

		if (parsingSuccessful == false) throw dab::Exception("FILE ERROR: failed to parse config data file " + pFileName, __FILE__, __FUNCTION__, __LINE__);

		Json::Value jointWeightsData = jsonHelper.getValue(restoreData, "jointWeights");

		jsonHelper.getFloats(jointWeightsData, mJointWeights);

		// apply body part filters
		for (int i = 0; i < mTorsoJointIndices.size(); ++i) mTorsoJointWeights.push_back(mJointWeights[mTorsoJointIndices[i]]);
		for (int i = 0; i < mLeftArmJointIndices.size(); ++i) mLeftArmJointWeights.push_back(mJointWeights[mLeftArmJointIndices[i]]);
		for (int i = 0; i < mRightArmJointIndices.size(); ++i) mRightArmJointWeights.push_back(mJointWeights[mRightArmJointIndices[i]]);
		for (int i = 0; i < mLeftLegJointIndices.size(); ++i) mLeftLegJointWeights.push_back(mJointWeights[mLeftLegJointIndices[i]]);
		for (int i = 0; i < mRightLegJointIndices.size(); ++i) mRightLegJointWeights.push_back(mJointWeights[mRightLegJointIndices[i]]);

		// normalize joint weights
		// option 1: normalise weights for each body part individually
		// option 2: normalise weights for fully body only and then use this normlisation factor for all body parts
		// currently option 2 is used
		float weightSum = 0;
		for (int i = 0; i < mJointWeights.size(); ++i) weightSum += mJointWeights[i];
		for (int i = 0; i < mJointWeights.size(); ++i) mJointWeights[i] /= weightSum;
		//weightSum = 0;
		//for (int i = 0; i < mTorsoJointWeights.size(); ++i) weightSum += mTorsoJointWeights[i];
		for (int i = 0; i < mTorsoJointWeights.size(); ++i) mTorsoJointWeights[i] /= weightSum;
		//weightSum = 0;
		//for (int i = 0; i < mLeftArmJointWeights.size(); ++i) weightSum += mLeftArmJointWeights[i];
		for (int i = 0; i < mLeftArmJointWeights.size(); ++i) mLeftArmJointWeights[i] /= weightSum;
		//weightSum = 0;
		//for (int i = 0; i < mRightArmJointWeights.size(); ++i) weightSum += mRightArmJointWeights[i];
		for (int i = 0; i < mRightArmJointWeights.size(); ++i) mRightArmJointWeights[i] /= weightSum;
		//weightSum = 0;
		//for (int i = 0; i < mLeftLegJointWeights.size(); ++i) weightSum += mLeftLegJointWeights[i];
		for (int i = 0; i < mLeftLegJointWeights.size(); ++i) mLeftLegJointWeights[i] /= weightSum;
		//weightSum = 0;
		//for (int i = 0; i < mRightLegJointWeights.size(); ++i) weightSum += mRightLegJointWeights[i];
		for (int i = 0; i < mRightLegJointWeights.size(); ++i) mRightLegJointWeights[i] /= weightSum;
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("JSON ERROR: failed to restore config from file " + pFileName, __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void
ofApp::loadJointConnectivity(const std::string& pFileName) throw (dab::Exception)
{
	try
	{
		std::string restoreString;
		dab::FileIO::get().read(pFileName, restoreString);

		Json::Reader reader;
		Json::Value restoreData;
		dab::JsonHelper& jsonHelper = dab::JsonHelper::get();

		bool parsingSuccessful = reader.parse(restoreString, restoreData);

		if (parsingSuccessful == false) throw dab::Exception("FILE ERROR: failed to parse config data file " + pFileName, __FILE__, __FUNCTION__, __LINE__);

		std::vector<Json::Value> jointsConnectivityData;
		jsonHelper.getValues(restoreData, "jointConnectivity", jointsConnectivityData);

		int jointCount = jointsConnectivityData.size();
		for (int jI = 0; jI < jointCount; ++jI)
		{
			std::vector<int> jointConnectivity;
			jsonHelper.getInts(jointsConnectivityData[jI], jointConnectivity);

			mJointConnectivity.push_back(jointConnectivity);
		}

		//// debug begin
		//for (int jI = 0; jI < jointCount; ++jI)
		//{
		//	std::cout << "joint parent " << jI << " children [ ";
		//	for (int cI = 0; cI < mJointConnectivity[jI].size(); ++cI)
		//	{
		//		std::cout << mJointConnectivity[jI][cI] << " ";
		//	}
		//	std::cout << "]\n";
		//}
		//// debug done

	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("JSON ERROR: failed to restore config from file " + pFileName, __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void
ofApp::setupData()
{
	//mMarkerPosData = std::shared_ptr<dab::Data>( new dab::Data("markerPositions", 3, 62));

	int jointCount = mJointWeights.size();

	mJointPosData = std::shared_ptr<dab::Data>(new dab::Data("jointPositions", 3, jointCount));
	mJointRotData = std::shared_ptr<dab::Data>(new dab::Data("jointRotations", 4, jointCount));

	// new procs for calculating linear and angular jerk from xsens kinematics data
	mJointLinAccelData = std::shared_ptr<dab::Data>(new dab::Data("jointLinAccel", 3, jointCount));
	mJointAngAccelData = std::shared_ptr<dab::Data>(new dab::Data("jointAngAccel", 3, jointCount));

}

void
ofApp::setupDataProc() throw (dab::Exception)
{
	try
	{
		// data processing pipeline
		mDataProcPipeline = std::shared_ptr<dab::DataProcPipeline>(new dab::DataProcPipeline());

		// data inputs
		mJointPosInputProc = std::shared_ptr<dab::DataProcInput>(new dab::DataProcInput("joint pos input", { mJointPosData }));
		mJointRotInputProc = std::shared_ptr<dab::DataProcInput>(new dab::DataProcInput("joint rot input", { mJointRotData }));

		//mJointLocRotInputProc = std::shared_ptr<dab::DataProcInput>(new dab::DataProcInput("joint locrot input", { mJointLocRotData }));
		
		mDataProcPipeline->addDataProc(mJointPosInputProc);
		mDataProcPipeline->addDataProc(mJointRotInputProc);

		//mDataProcPipeline->addDataProc(mJointLocRotInputProc);

		// joint position global to local conversion
		mJointPosGlobalToLocalProc = std::shared_ptr<dab::DataProcPosGlobalToLocal>(new dab::DataProcPosGlobalToLocal("joint pos global local calc", "jointPosLocal", mJointConnectivity));
	
		// joint position data processing
		mJointPosScaleProc = std::shared_ptr<dab::DataProcScale>(new dab::DataProcScale("joint pos scale", "jointPosition", 1.0));

		// derivative procs
		mJointVelocityProc = std::shared_ptr<dab::DataProcDerivative>(new dab::DataProcDerivative("joint velocity calc", "jointVelocity"));
		mJointAccelerationProc = std::shared_ptr<dab::DataProcDerivative>(new dab::DataProcDerivative("joint acceleration calc", "jointAcceleration"));
		mJointJerkProc = std::shared_ptr<dab::DataProcDerivative>(new dab::DataProcDerivative("joint jerk calc", "jointJerk"));

		// smooth procs
		mJointPosSmoothProc = std::shared_ptr<dab::DataProcLowPass>(new dab::DataProcLowPass("joint pos smooth", "jointPosition", 0.0));
		mJointVelocitySmoothProc = std::shared_ptr<dab::DataProcLowPass>(new dab::DataProcLowPass("joint velocity smooth", "jointVelocity", 0.9));
		mJointAccelerationSmoothProc = std::shared_ptr<dab::DataProcLowPass>(new dab::DataProcLowPass("joint acceleration smooth", "jointAcceleration", 0.9));
		mJointJerkSmoothProc = std::shared_ptr<dab::DataProcLowPass>(new dab::DataProcLowPass("joint jerk smooth", "jointJerk", 0.9));

		// scalar procs
		mJointVelocityScalarProc = std::shared_ptr<dab::DataProcScalar>(new dab::DataProcScalar("joint velocity scalar calc", "jointVelocityScalar", dab::DataProcScalar::Max));
		mJointAccelerationScalarProc = std::shared_ptr<dab::DataProcScalar>(new dab::DataProcScalar("joint acceleration scalar calc", "jointAccelerationScalar", dab::DataProcScalar::Max));
		mJointJerkScalarProc = std::shared_ptr<dab::DataProcScalar>(new dab::DataProcScalar("joint jerk scalar calc", "jointJerkScalar", dab::DataProcScalar::Max));

		// filter procs
		mJointPosTorsoFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint position torso filter", "JointPosTorso", mTorsoJointIndices));
		mJointPosLeftArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint position left arm filter", "JointPosLeftArm", mLeftArmJointIndices));
		mJointPosRightArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint position right arm filter", "JointPosRightArm", mRightArmJointIndices));
		mJointPosLeftLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint position left leg filter", "JointPosLeftLeg", mLeftLegJointIndices));
		mJointPosRightLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint position right leg filter", "JointPosRightLeg", mRightLegJointIndices));

		mJointVelocityTorsoFilterProc = std::shared_ptr<dab::DataProcFilter>( new dab::DataProcFilter("joint velocity torso filter", "JointVelocityTorso", mTorsoJointIndices) );
		mJointVelocityLeftArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint velocity left arm filter", "JointVelocityLeftArm", mLeftArmJointIndices));
		mJointVelocityRightArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint velocity right arm filter", "JointVelocityRightArm", mRightArmJointIndices));
		mJointVelocityLeftLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint velocity left leg filter", "JointVelocityLeftLeg", mLeftLegJointIndices));
		mJointVelocityRightLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint velocity right leg filter", "JointVelocityRightLeg", mRightLegJointIndices));

		mJointAccelerationTorsoFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint acceleration torso filter", "JointAccelerationTorso", mTorsoJointIndices));
		mJointAccelerationLeftArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint acceleration left arm filter", "JointAccelerationLeftArm", mLeftArmJointIndices));
		mJointAccelerationRightArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint acceleration right arm filter", "JointAccelerationRightArm", mRightArmJointIndices));
		mJointAccelerationLeftLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint acceleration left leg filter", "JointAccelerationLeftLeg", mLeftLegJointIndices));
		mJointAccelerationRightLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint acceleration right leg filter", "JointAccelerationRightLeg", mRightLegJointIndices));

		mJointJerkTorsoFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint jerk torso filter", "JointJerkTorso", mTorsoJointIndices));
		mJointJerkLeftArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint jerk left arm filter", "JointJerkLeftArm", mLeftArmJointIndices));
		mJointJerkRightArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint jerk right arm filter", "JointJerkRightArm", mRightArmJointIndices));
		mJointJerkLeftLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint jerk left leg filter", "JointJerkLeftLeg", mLeftLegJointIndices));
		mJointJerkRightLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint jerk right leg filter", "JointJerkRightLeg", mRightLegJointIndices));

		// analysis procs
		mJointWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint weight effort calc", "jointWeightEffort", 10));
		mJointTorsoWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint torso weight effort calc", "jointTorsoWeightEffort", 10));
		mJointLeftArmWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint left arm weight effort calc", "jointLeftArmWeightEffort", 10));
		mJointRightArmWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint right arm weight effort calc", "jointRightArmWeightEffort", 10));
		mJointLeftLegWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint left leg weight effort calc", "jointLeftLegWeightEffort", 10));
		mJointRightLegWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint right leg weight effort calc", "jointRightLegWeightEffort", 10));

		mJointWeightEffortProc->setWeights(mJointWeights);
		mJointTorsoWeightEffortProc->setWeights(mTorsoJointWeights);
		mJointLeftArmWeightEffortProc->setWeights(mLeftArmJointWeights);
		mJointRightArmWeightEffortProc->setWeights(mRightArmJointWeights);
		mJointLeftLegWeightEffortProc->setWeights(mLeftLegJointWeights);
		mJointRightLegWeightEffortProc->setWeights(mRightLegJointWeights);

		mJointTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint time effort calc", "jointTimeEffort", 10));
		mJointTorsoTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint torso time effort calc", "jointTorsoTimeEffort", 10));
		mJointLeftArmTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint left arm time effort calc", "jointLeftArmTimeEffort", 10));
		mJointRightArmTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint right arm time effort calc", "jointRightArmTimeEffort", 10));
		mJointLeftLegTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint left leg time effort calc", "jointLeftLegTimeEffort", 10));
		mJointRightLegTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint right leg time effort calc", "jointRightLegTimeEffort", 10));

		mJointTimeEffortProc->setWeights(mJointWeights);
		mJointTorsoTimeEffortProc->setWeights(mTorsoJointWeights);
		mJointLeftArmTimeEffortProc->setWeights(mLeftArmJointWeights);
		mJointRightArmTimeEffortProc->setWeights(mRightArmJointWeights);
		mJointLeftLegTimeEffortProc->setWeights(mLeftLegJointWeights);
		mJointRightLegTimeEffortProc->setWeights(mRightLegJointWeights);
		
		mJointFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint flow effort calc", "jointFlowEffort", 10));
		mJointTorsoFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint torso flow effort calc", "jointTorsoFlowEffort", 10));
		mJointLeftArmFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint left arm flow effort calc", "jointLeftArmFlowEffort", 10));
		mJointRightArmFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint right arm flow effort calc", "jointRightArmFlowEffort", 10));
		mJointLeftLegFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint left leg flow effort calc", "jointLeftLegFlowEffort", 10));
		mJointRightLegFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint right leg flow effort calc", "jointRightLegFlowEffort", 10));
		
		mJointFlowEffortProc->setWeights(mJointWeights);
		mJointTorsoFlowEffortProc->setWeights(mTorsoJointWeights);
		mJointLeftArmFlowEffortProc->setWeights(mLeftArmJointWeights);
		mJointRightArmFlowEffortProc->setWeights(mRightArmJointWeights);
		mJointLeftLegFlowEffortProc->setWeights(mLeftLegJointWeights);
		mJointRightLegFlowEffortProc->setWeights(mRightLegJointWeights);

		mJointSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint space effort calc", "jointSpaceEffort", 10));
		mJointTorsoSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint torso space effort calc", "jointTorsoSpaceEffort", 10));
		mJointLeftArmSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint left arm space effort calc", "jointLeftArmSpaceEffort", 10));
		mJointRightArmSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint right arm space effort calc", "jointRightArmSpaceEffort", 10));
		mJointLeftLegSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint left leg space effort calc", "jointLeftLegSpaceEffort", 10));
		mJointRightLegSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint right leg space effort calc", "jointRightLegSpaceEffort", 10));

		mJointSpaceEffortProc->setWeights(mJointWeights);
		mJointTorsoSpaceEffortProc->setWeights(mTorsoJointWeights);
		mJointLeftArmSpaceEffortProc->setWeights(mLeftArmJointWeights);
		mJointRightArmSpaceEffortProc->setWeights(mRightArmJointWeights);
		mJointLeftLegSpaceEffortProc->setWeights(mLeftLegJointWeights);
		mJointRightLegSpaceEffortProc->setWeights(mRightLegJointWeights);

		// util procs
		mWeightEffortCombineProc = std::shared_ptr<dab::DataProcCombine>(new dab::DataProcCombine("weight effort combined", "WeightEffortCombined"));
		mTimeEffortCombineProc = std::shared_ptr<dab::DataProcCombine>(new dab::DataProcCombine("time effort combined", "TimeEffortCombined"));
		mFlowEffortCombineProc = std::shared_ptr<dab::DataProcCombine>(new dab::DataProcCombine("flow effort combined", "FlowEffortCombined"));
		mSpaceEffortCombineProc = std::shared_ptr<dab::DataProcCombine>(new dab::DataProcCombine("space effort combined", "SpaceEffortCombined"));

		// add procs to pipeline
		mDataProcPipeline->addDataProc(mJointPosGlobalToLocalProc);
		mDataProcPipeline->addDataProc(mJointPosScaleProc);
		mDataProcPipeline->addDataProc(mJointVelocityProc);
		mDataProcPipeline->addDataProc(mJointAccelerationProc);
		mDataProcPipeline->addDataProc(mJointJerkProc);
		mDataProcPipeline->addDataProc(mJointPosSmoothProc);
		mDataProcPipeline->addDataProc(mJointVelocitySmoothProc);
		mDataProcPipeline->addDataProc(mJointAccelerationSmoothProc);
		mDataProcPipeline->addDataProc(mJointJerkSmoothProc);
		mDataProcPipeline->addDataProc(mJointVelocityScalarProc);
		mDataProcPipeline->addDataProc(mJointAccelerationScalarProc);
		mDataProcPipeline->addDataProc(mJointJerkScalarProc);
		mDataProcPipeline->addDataProc(mJointPosTorsoFilterProc);
		mDataProcPipeline->addDataProc(mJointPosLeftArmFilterProc);
		mDataProcPipeline->addDataProc(mJointPosRightArmFilterProc);
		mDataProcPipeline->addDataProc(mJointPosLeftLegFilterProc);
		mDataProcPipeline->addDataProc(mJointPosRightLegFilterProc);
		mDataProcPipeline->addDataProc(mJointVelocityTorsoFilterProc);
		mDataProcPipeline->addDataProc(mJointVelocityLeftArmFilterProc);
		mDataProcPipeline->addDataProc(mJointVelocityRightArmFilterProc);
		mDataProcPipeline->addDataProc(mJointVelocityLeftLegFilterProc);
		mDataProcPipeline->addDataProc(mJointVelocityRightLegFilterProc);
		mDataProcPipeline->addDataProc(mJointAccelerationTorsoFilterProc);
		mDataProcPipeline->addDataProc(mJointAccelerationLeftArmFilterProc);
		mDataProcPipeline->addDataProc(mJointAccelerationRightArmFilterProc);
		mDataProcPipeline->addDataProc(mJointAccelerationLeftLegFilterProc);
		mDataProcPipeline->addDataProc(mJointAccelerationRightLegFilterProc);
		mDataProcPipeline->addDataProc(mJointJerkTorsoFilterProc);
		mDataProcPipeline->addDataProc(mJointJerkLeftArmFilterProc);
		mDataProcPipeline->addDataProc(mJointJerkRightArmFilterProc);
		mDataProcPipeline->addDataProc(mJointJerkLeftLegFilterProc);
		mDataProcPipeline->addDataProc(mJointJerkRightLegFilterProc);
		mDataProcPipeline->addDataProc(mJointWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointTorsoWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointLeftArmWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointRightArmWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointLeftLegWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointRightLegWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointTorsoTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointLeftArmTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointRightArmTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointLeftLegTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointRightLegTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointTorsoFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointLeftArmFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointRightArmFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointLeftLegFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointRightLegFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointSpaceEffortProc);
		mDataProcPipeline->addDataProc(mJointTorsoSpaceEffortProc);
		mDataProcPipeline->addDataProc(mJointLeftArmSpaceEffortProc);
		mDataProcPipeline->addDataProc(mJointRightArmSpaceEffortProc);
		mDataProcPipeline->addDataProc(mJointLeftLegSpaceEffortProc);
		mDataProcPipeline->addDataProc(mJointRightLegSpaceEffortProc);
		mDataProcPipeline->addDataProc(mWeightEffortCombineProc);
		mDataProcPipeline->addDataProc(mTimeEffortCombineProc);
		mDataProcPipeline->addDataProc(mFlowEffortCombineProc);
		mDataProcPipeline->addDataProc(mSpaceEffortCombineProc);

		// connect procs

		mDataProcPipeline->connect(mJointPosInputProc, mJointPosGlobalToLocalProc);

		mDataProcPipeline->connect(mJointPosInputProc, mJointPosScaleProc);
		mDataProcPipeline->connect(mJointPosScaleProc, mJointPosSmoothProc);
		mDataProcPipeline->connect(mJointPosSmoothProc, mJointVelocityProc);
		mDataProcPipeline->connect(mJointVelocityProc, mJointVelocitySmoothProc);
		mDataProcPipeline->connect(mJointVelocitySmoothProc, mJointAccelerationProc);
		mDataProcPipeline->connect(mJointAccelerationProc, mJointAccelerationSmoothProc);
		mDataProcPipeline->connect(mJointAccelerationSmoothProc, mJointJerkProc);
		mDataProcPipeline->connect(mJointJerkProc, mJointJerkSmoothProc);
		mDataProcPipeline->connect(mJointVelocitySmoothProc, mJointVelocityScalarProc);
		mDataProcPipeline->connect(mJointAccelerationSmoothProc, mJointAccelerationScalarProc);
		mDataProcPipeline->connect(mJointJerkSmoothProc, mJointJerkScalarProc);
		mDataProcPipeline->connect(mJointPosSmoothProc, mJointPosTorsoFilterProc);
		mDataProcPipeline->connect(mJointPosSmoothProc, mJointPosLeftArmFilterProc);
		mDataProcPipeline->connect(mJointPosSmoothProc, mJointPosRightArmFilterProc);
		mDataProcPipeline->connect(mJointPosSmoothProc, mJointPosLeftLegFilterProc);
		mDataProcPipeline->connect(mJointPosSmoothProc, mJointPosRightLegFilterProc);
		mDataProcPipeline->connect(mJointVelocityScalarProc, mJointVelocityTorsoFilterProc);
		mDataProcPipeline->connect(mJointVelocityScalarProc, mJointVelocityLeftArmFilterProc);
		mDataProcPipeline->connect(mJointVelocityScalarProc, mJointVelocityRightArmFilterProc);
		mDataProcPipeline->connect(mJointVelocityScalarProc, mJointVelocityLeftLegFilterProc);
		mDataProcPipeline->connect(mJointVelocityScalarProc, mJointVelocityRightLegFilterProc);
		mDataProcPipeline->connect(mJointAccelerationScalarProc, mJointAccelerationTorsoFilterProc);
		mDataProcPipeline->connect(mJointAccelerationScalarProc, mJointAccelerationLeftArmFilterProc);
		mDataProcPipeline->connect(mJointAccelerationScalarProc, mJointAccelerationRightArmFilterProc);
		mDataProcPipeline->connect(mJointAccelerationScalarProc, mJointAccelerationLeftLegFilterProc);
		mDataProcPipeline->connect(mJointAccelerationScalarProc, mJointAccelerationRightLegFilterProc);
		mDataProcPipeline->connect(mJointJerkScalarProc, mJointJerkTorsoFilterProc);
		mDataProcPipeline->connect(mJointJerkScalarProc, mJointJerkLeftArmFilterProc);
		mDataProcPipeline->connect(mJointJerkScalarProc, mJointJerkRightArmFilterProc);
		mDataProcPipeline->connect(mJointJerkScalarProc, mJointJerkLeftLegFilterProc);
		mDataProcPipeline->connect(mJointJerkScalarProc, mJointJerkRightLegFilterProc);
		mDataProcPipeline->connect(mJointVelocityScalarProc, mJointWeightEffortProc);
		mDataProcPipeline->connect(mJointVelocityTorsoFilterProc, mJointTorsoWeightEffortProc);
		mDataProcPipeline->connect(mJointVelocityLeftArmFilterProc, mJointLeftArmWeightEffortProc);
		mDataProcPipeline->connect(mJointVelocityRightArmFilterProc, mJointRightArmWeightEffortProc);
		mDataProcPipeline->connect(mJointVelocityLeftLegFilterProc, mJointLeftLegWeightEffortProc);
		mDataProcPipeline->connect(mJointVelocityRightLegFilterProc, mJointRightLegWeightEffortProc);
		mDataProcPipeline->connect(mJointAccelerationScalarProc, mJointTimeEffortProc);
		mDataProcPipeline->connect(mJointAccelerationTorsoFilterProc, mJointTorsoTimeEffortProc);
		mDataProcPipeline->connect(mJointAccelerationLeftArmFilterProc, mJointLeftArmTimeEffortProc);
		mDataProcPipeline->connect(mJointAccelerationRightArmFilterProc, mJointRightArmTimeEffortProc);
		mDataProcPipeline->connect(mJointAccelerationLeftLegFilterProc, mJointLeftLegTimeEffortProc);
		mDataProcPipeline->connect(mJointAccelerationRightLegFilterProc, mJointRightLegTimeEffortProc);
		mDataProcPipeline->connect(mJointJerkScalarProc, mJointFlowEffortProc);
		mDataProcPipeline->connect(mJointJerkTorsoFilterProc, mJointTorsoFlowEffortProc);
		mDataProcPipeline->connect(mJointJerkLeftArmFilterProc, mJointLeftArmFlowEffortProc);
		mDataProcPipeline->connect(mJointJerkRightArmFilterProc, mJointRightArmFlowEffortProc);
		mDataProcPipeline->connect(mJointJerkLeftLegFilterProc, mJointLeftLegFlowEffortProc);
		mDataProcPipeline->connect(mJointJerkRightLegFilterProc, mJointRightLegFlowEffortProc);
		mDataProcPipeline->connect(mJointPosSmoothProc, mJointSpaceEffortProc);
		mDataProcPipeline->connect(mJointPosTorsoFilterProc, mJointTorsoSpaceEffortProc);
		mDataProcPipeline->connect(mJointPosLeftArmFilterProc, mJointLeftArmSpaceEffortProc);
		mDataProcPipeline->connect(mJointPosRightArmFilterProc, mJointRightArmSpaceEffortProc);
		mDataProcPipeline->connect(mJointPosLeftLegFilterProc, mJointLeftLegSpaceEffortProc);
		mDataProcPipeline->connect(mJointPosRightLegFilterProc, mJointRightLegSpaceEffortProc);

		mDataProcPipeline->connect(mJointWeightEffortProc, mWeightEffortCombineProc);
		mDataProcPipeline->connect(mJointTorsoWeightEffortProc, mWeightEffortCombineProc);
		mDataProcPipeline->connect(mJointLeftArmWeightEffortProc, mWeightEffortCombineProc);
		mDataProcPipeline->connect(mJointRightArmWeightEffortProc, mWeightEffortCombineProc);
		mDataProcPipeline->connect(mJointLeftLegWeightEffortProc, mWeightEffortCombineProc);
		mDataProcPipeline->connect(mJointRightLegWeightEffortProc, mWeightEffortCombineProc);

		mDataProcPipeline->connect(mJointTimeEffortProc, mTimeEffortCombineProc);
		mDataProcPipeline->connect(mJointTorsoTimeEffortProc, mTimeEffortCombineProc);
		mDataProcPipeline->connect(mJointLeftArmTimeEffortProc, mTimeEffortCombineProc);
		mDataProcPipeline->connect(mJointRightArmTimeEffortProc, mTimeEffortCombineProc);
		mDataProcPipeline->connect(mJointLeftLegTimeEffortProc, mTimeEffortCombineProc);
		mDataProcPipeline->connect(mJointRightLegTimeEffortProc, mTimeEffortCombineProc);

		mDataProcPipeline->connect(mJointFlowEffortProc, mFlowEffortCombineProc);
		mDataProcPipeline->connect(mJointTorsoFlowEffortProc, mFlowEffortCombineProc);
		mDataProcPipeline->connect(mJointLeftArmFlowEffortProc, mFlowEffortCombineProc);
		mDataProcPipeline->connect(mJointRightArmFlowEffortProc, mFlowEffortCombineProc);
		mDataProcPipeline->connect(mJointLeftLegFlowEffortProc, mFlowEffortCombineProc);
		mDataProcPipeline->connect(mJointRightLegFlowEffortProc, mFlowEffortCombineProc);

		mDataProcPipeline->connect(mJointSpaceEffortProc, mSpaceEffortCombineProc);
		mDataProcPipeline->connect(mJointTorsoSpaceEffortProc, mSpaceEffortCombineProc);
		mDataProcPipeline->connect(mJointLeftArmSpaceEffortProc, mSpaceEffortCombineProc);
		mDataProcPipeline->connect(mJointRightArmSpaceEffortProc, mSpaceEffortCombineProc);
		mDataProcPipeline->connect(mJointLeftLegSpaceEffortProc, mSpaceEffortCombineProc);
		mDataProcPipeline->connect(mJointRightLegSpaceEffortProc, mSpaceEffortCombineProc);


		// joint rotation global to local conversion
		mJointRotGlobalToLocalProc = std::shared_ptr<dab::DataProcRotGlobalToLocal>(new dab::DataProcRotGlobalToLocal("joint rot global local calc", "jointRotLocal", mJointConnectivity));

		// joint rotation data quaternion to euler conversion
		mJointRotQuatEulerProc = std::shared_ptr<dab::DataProcQuatEuler>(new dab::DataProcQuatEuler("joint rot euler calc", "jointRotEuler"));
		mJointRotQuatVelocityEulerProc = std::shared_ptr<dab::DataProcQuatEuler>(new dab::DataProcQuatEuler("joint rot velocity euler calc", "jointRotVelocityEuler"));
		mJointRotQuatAccelerationEulerProc = std::shared_ptr<dab::DataProcQuatEuler>(new dab::DataProcQuatEuler("joint rot acceleration euler calc", "jointRotAccelerationEuler"));
		mJointRotQuatJerkEulerProc = std::shared_ptr<dab::DataProcQuatEuler>(new dab::DataProcQuatEuler("joint rot jerk euler calc", "jointRotJerkEuler"));

		// derivative procs
		mJointRotQuatVelocityProc = std::shared_ptr<dab::DataProcDerivativeQuat>(new dab::DataProcDerivativeQuat("joint rot quat velocity calc", "jointRotQuatVelocity"));
		mJointRotQuatAccelerationProc = std::shared_ptr<dab::DataProcDerivativeQuat>(new dab::DataProcDerivativeQuat("joint rot quat acceleration calc", "jointRotQuatAcceleration"));
		mJointRotQuatJerkProc = std::shared_ptr<dab::DataProcDerivativeQuat>(new dab::DataProcDerivativeQuat("joint rot quat jerk calc", "jointRotQuatJerk"));

		// smooth procs
		mJointRotQuatSmoothProc = std::shared_ptr<dab::DataProcLowPassQuat>(new dab::DataProcLowPassQuat("joint rot quat smooth", "jointRotQuat", 0.9));
		mJointRotQuatVelocitySmoothProc = std::shared_ptr<dab::DataProcLowPassQuat>(new dab::DataProcLowPassQuat("joint rot quat velocity smooth", "jointRotQuatVelocity", 0.9));
		mJointRotQuatAccelerationSmoothProc = std::shared_ptr<dab::DataProcLowPassQuat>(new dab::DataProcLowPassQuat("joint rot quat acceleration smooth", "jointRotQuatAcceleration", 0.9));
		mJointRotQuatJerkSmoothProc = std::shared_ptr<dab::DataProcLowPassQuat>(new dab::DataProcLowPassQuat("joint rot quat jerk smooth", "jointRotQuatJerk", 0.9));


		//// smooth procs
		//mJointRotQuatSmoothProc = std::shared_ptr<dab::DataProcLowPassQuat>(new dab::DataProcLowPassQuat("joint rot quat smooth", "jointRotQuat", 0.9));
		//mJointRotQuatVelocitySmoothProc = std::shared_ptr<dab::DataProcLowPassQuat>(new dab::DataProcLowPassQuat("joint rot quat velocity smooth", "jointRotQuatVelocity", 0.9));
		//mJointRotQuatAccelerationSmoothProc = std::shared_ptr<dab::DataProcLowPassQuat>(new dab::DataProcLowPassQuat("joint rot quat acceleration smooth", "jointRotQuatAcceleration", 0.9));
		//mJointRotQuatJerkSmoothProc = std::shared_ptr<dab::DataProcLowPassQuat>(new dab::DataProcLowPassQuat("joint rot quat jerk smooth", "jointRotQuatJerk", 0.9));

		// scalar procs
		mJointRotVelocityScalarProc = std::shared_ptr<dab::DataProcScalar>(new dab::DataProcScalar("joint rot velocity scalar calc", "jointRotVelocityScalar", dab::DataProcScalar::Max));
		mJointRotAccelerationScalarProc = std::shared_ptr<dab::DataProcScalar>(new dab::DataProcScalar("joint rot acceleration scalar calc", "jointRotAccelerationScalar", dab::DataProcScalar::Max));
		mJointRotJerkScalarProc = std::shared_ptr<dab::DataProcScalar>(new dab::DataProcScalar("joint rot jerk scalar calc", "jointRotJerkScalar", dab::DataProcScalar::Max));

		// filter procs
		mJointRotTorsoFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation torso filter", "JointRotTorso", mTorsoJointIndices));
		mJointRotLeftArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation left arm filter", "JointRotLeftArm", mLeftArmJointIndices));
		mJointRotRightArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation right arm filter", "JointRotRightArm", mRightArmJointIndices));
		mJointRotLeftLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation left leg filter", "JointRotLeftLeg", mLeftLegJointIndices));
		mJointRotRightLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation right leg filter", "JointRotRightLeg", mRightLegJointIndices));

		mJointRotVelocityTorsoFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation velocity torso filter", "JointRotVelocityTorso", mTorsoJointIndices));
		mJointRotVelocityLeftArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation velocity left arm filter", "JointRotVelocityLeftArm", mLeftArmJointIndices));
		mJointRotVelocityRightArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation velocity right arm filter", "JointRotVelocityRightArm", mRightArmJointIndices));
		mJointRotVelocityLeftLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation velocity left leg filter", "JointRotVelocityLeftLeg", mLeftLegJointIndices));
		mJointRotVelocityRightLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation velocity right leg filter", "JointRotVelocityRightLeg", mRightLegJointIndices));

		mJointRotAccelerationTorsoFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation acceleration torso filter", "JointRotAccelerationTorso", mTorsoJointIndices));
		mJointRotAccelerationLeftArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation acceleration left arm filter", "JointRotAccelerationLeftArm", mLeftArmJointIndices));
		mJointRotAccelerationRightArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation acceleration right arm filter", "JointRotAccelerationRightArm", mRightArmJointIndices));
		mJointRotAccelerationLeftLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation acceleration left leg filter", "JointRotAccelerationLeftLeg", mLeftLegJointIndices));
		mJointRotAccelerationRightLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation acceleration right leg filter", "JointRotAccelerationRightLeg", mRightLegJointIndices));

		mJointRotJerkTorsoFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation jerk torso filter", "JointRotJerkTorso", mTorsoJointIndices));
		mJointRotJerkLeftArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation jerk left arm filter", "JointRotJerkLeftArm", mLeftArmJointIndices));
		mJointRotJerkRightArmFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation jerk right arm filter", "JointRotJerkRightArm", mRightArmJointIndices));
		mJointRotJerkLeftLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation jerk left leg filter", "JointRotJerkLeftLeg", mLeftLegJointIndices));
		mJointRotJerkRightLegFilterProc = std::shared_ptr<dab::DataProcFilter>(new dab::DataProcFilter("joint rotation jerk right leg filter", "JointRotJerkRightLeg", mRightLegJointIndices));

		// analysis procs
		mJointRotWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint rot effort calc", "jointRotWeightEffort", 10));
		mJointRotTorsoWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint rot torso weight effort calc", "jointRotTorsoWeightEffort", 10));
		mJointRotLeftArmWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint rot left arm weight effort calc", "jointRotLeftArmWeightEffort", 10));
		mJointRotRightArmWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint rot right arm weight effort calc", "jointRotRightArmWeightEffort", 10));
		mJointRotLeftLegWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint rot left leg weight effort calc", "jointRotLeftLegWeightEffort", 10));
		mJointRotRightLegWeightEffortProc = std::shared_ptr<dab::DataProcWeightEffort>(new dab::DataProcWeightEffort("joint rot right leg weight effort calc", "jointRotRightLegWeightEffort", 10));

		mJointRotWeightEffortProc->setWeights(mJointWeights);
		mJointRotTorsoWeightEffortProc->setWeights(mTorsoJointWeights);
		mJointRotLeftArmWeightEffortProc->setWeights(mLeftArmJointWeights);
		mJointRotRightArmWeightEffortProc->setWeights(mRightArmJointWeights);
		mJointRotLeftLegWeightEffortProc->setWeights(mLeftLegJointWeights);
		mJointRotRightLegWeightEffortProc->setWeights(mRightLegJointWeights);

		mJointRotTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint rot time effort calc", "jointRotTimeEffort", 10));
		mJointRotTorsoTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint rot torso time effort calc", "jointRotTorsoTimeEffort", 10));
		mJointRotLeftArmTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint rot left arm time effort calc", "jointRotLeftArmTimeEffort", 10));
		mJointRotRightArmTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint rot right arm time effort calc", "jointRotRightArmTimeEffort", 10));
		mJointRotLeftLegTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint rot left leg time effort calc", "jointRotLeftLegTimeEffort", 10));
		mJointRotRightLegTimeEffortProc = std::shared_ptr<dab::DataProcTimeEffort>(new dab::DataProcTimeEffort("joint rot right leg time effort calc", "jointRotRightLegTimeEffort", 10));

		mJointRotTimeEffortProc->setWeights(mJointWeights);
		mJointRotTorsoTimeEffortProc->setWeights(mTorsoJointWeights);
		mJointRotLeftArmTimeEffortProc->setWeights(mLeftArmJointWeights);
		mJointRotRightArmTimeEffortProc->setWeights(mRightArmJointWeights);
		mJointRotLeftLegTimeEffortProc->setWeights(mLeftLegJointWeights);
		mJointRotRightLegTimeEffortProc->setWeights(mRightLegJointWeights);

		mJointRotFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint rot flow effort calc", "jointRotFlowEffort", 10));
		mJointRotTorsoFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint rot torso flow effort calc", "jointRotTorsoFlowEffort", 10));
		mJointRotLeftArmFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint rot left arm flow effort calc", "jointRotLeftArmFlowEffort", 10));
		mJointRotRightArmFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint rot right arm flow effort calc", "jointRotRightArmFlowEffort", 10));
		mJointRotLeftLegFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint rot left leg flow effort calc", "jointRotLeftLegFlowEffort", 10));
		mJointRotRightLegFlowEffortProc = std::shared_ptr<dab::DataProcFlowEffort>(new dab::DataProcFlowEffort("joint rot right leg flow effort calc", "jointRotRightLegFlowEffort", 10));

		mJointRotFlowEffortProc->setWeights(mJointWeights);
		mJointRotTorsoFlowEffortProc->setWeights(mTorsoJointWeights);
		mJointRotLeftArmFlowEffortProc->setWeights(mLeftArmJointWeights);
		mJointRotRightArmFlowEffortProc->setWeights(mRightArmJointWeights);
		mJointRotLeftLegFlowEffortProc->setWeights(mLeftLegJointWeights);
		mJointRotRightLegFlowEffortProc->setWeights(mRightLegJointWeights);

		mJointRotSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint rot space effort calc", "jointRotSpaceEffort", 10));
		mJointRotTorsoSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint rot torso space effort calc", "jointRotTorsoSpaceEffort", 10));
		mJointRotLeftArmSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint rot left arm space effort calc", "jointRotLeftArmSpaceEffort", 10));
		mJointRotRightArmSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint rot right arm space effort calc", "jointRotRightArmSpaceEffort", 10));
		mJointRotLeftLegSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint rot left leg space effort calc", "jointRotLeftLegSpaceEffort", 10));
		mJointRotRightLegSpaceEffortProc = std::shared_ptr<dab::DataProcSpaceEffort2>(new dab::DataProcSpaceEffort2("joint rot right leg space effort calc", "jointRotRightLegSpaceEffort", 10));

		mJointRotSpaceEffortProc->setWeights(mJointWeights);
		mJointRotTorsoSpaceEffortProc->setWeights(mTorsoJointWeights);
		mJointRotLeftArmSpaceEffortProc->setWeights(mLeftArmJointWeights);
		mJointRotRightArmSpaceEffortProc->setWeights(mRightArmJointWeights);
		mJointRotLeftLegSpaceEffortProc->setWeights(mLeftLegJointWeights);
		mJointRotRightLegSpaceEffortProc->setWeights(mRightLegJointWeights);

		// util procs
		mRotWeightEffortCombineProc = std::shared_ptr<dab::DataProcCombine>(new dab::DataProcCombine("rotation weight effort combined", "RotWeightEffortCombined"));
		mRotTimeEffortCombineProc = std::shared_ptr<dab::DataProcCombine>(new dab::DataProcCombine("rotation time effort combined", "RotTimeEffortCombined"));
		mRotFlowEffortCombineProc = std::shared_ptr<dab::DataProcCombine>(new dab::DataProcCombine("rotation flow effort combined", "RotFlowEffortCombined"));
		mRotSpaceEffortCombineProc = std::shared_ptr<dab::DataProcCombine>(new dab::DataProcCombine("rotation space effort combined", "RotSpaceEffortCombined"));

		mDataProcPipeline->addDataProc(mJointRotGlobalToLocalProc);

		mDataProcPipeline->addDataProc(mJointRotQuatEulerProc);
		mDataProcPipeline->addDataProc(mJointRotQuatSmoothProc);
		mDataProcPipeline->addDataProc(mJointRotQuatVelocityProc);
		mDataProcPipeline->addDataProc(mJointRotQuatVelocitySmoothProc);
		mDataProcPipeline->addDataProc(mJointRotQuatVelocityEulerProc);
		mDataProcPipeline->addDataProc(mJointRotVelocityScalarProc);
		mDataProcPipeline->addDataProc(mJointRotQuatAccelerationProc);
		mDataProcPipeline->addDataProc(mJointRotQuatAccelerationSmoothProc);
		mDataProcPipeline->addDataProc(mJointRotQuatAccelerationEulerProc);
		mDataProcPipeline->addDataProc(mJointRotAccelerationScalarProc);
		mDataProcPipeline->addDataProc(mJointRotQuatJerkProc);
		mDataProcPipeline->addDataProc(mJointRotQuatJerkSmoothProc);
		mDataProcPipeline->addDataProc(mJointRotQuatJerkEulerProc);
		mDataProcPipeline->addDataProc(mJointRotJerkScalarProc);
		mDataProcPipeline->addDataProc(mJointRotTorsoFilterProc);
		mDataProcPipeline->addDataProc(mJointRotLeftArmFilterProc);
		mDataProcPipeline->addDataProc(mJointRotRightArmFilterProc);
		mDataProcPipeline->addDataProc(mJointRotLeftLegFilterProc);
		mDataProcPipeline->addDataProc(mJointRotRightLegFilterProc);
		mDataProcPipeline->addDataProc(mJointRotVelocityTorsoFilterProc);
		mDataProcPipeline->addDataProc(mJointRotVelocityLeftArmFilterProc);
		mDataProcPipeline->addDataProc(mJointRotVelocityRightArmFilterProc);
		mDataProcPipeline->addDataProc(mJointRotVelocityLeftLegFilterProc);
		mDataProcPipeline->addDataProc(mJointRotVelocityRightLegFilterProc);
		mDataProcPipeline->addDataProc(mJointRotAccelerationTorsoFilterProc);
		mDataProcPipeline->addDataProc(mJointRotAccelerationLeftArmFilterProc);
		mDataProcPipeline->addDataProc(mJointRotAccelerationRightArmFilterProc);
		mDataProcPipeline->addDataProc(mJointRotAccelerationLeftLegFilterProc);
		mDataProcPipeline->addDataProc(mJointRotAccelerationRightLegFilterProc);
		mDataProcPipeline->addDataProc(mJointRotJerkTorsoFilterProc);
		mDataProcPipeline->addDataProc(mJointRotJerkLeftArmFilterProc);
		mDataProcPipeline->addDataProc(mJointRotJerkRightArmFilterProc);
		mDataProcPipeline->addDataProc(mJointRotJerkLeftLegFilterProc);
		mDataProcPipeline->addDataProc(mJointRotJerkRightLegFilterProc);
		mDataProcPipeline->addDataProc(mJointRotWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointRotTorsoWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointRotLeftArmWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointRotRightArmWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointRotLeftLegWeightEffortProc);
		mDataProcPipeline->addDataProc(mJointRotRightLegWeightEffortProc);
		mDataProcPipeline->addDataProc(mRotWeightEffortCombineProc);
		mDataProcPipeline->addDataProc(mJointRotTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointRotTorsoTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointRotLeftArmTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointRotRightArmTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointRotLeftLegTimeEffortProc);
		mDataProcPipeline->addDataProc(mJointRotRightLegTimeEffortProc);
		mDataProcPipeline->addDataProc(mRotTimeEffortCombineProc);
		mDataProcPipeline->addDataProc(mJointRotFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointRotTorsoFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointRotLeftArmFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointRotRightArmFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointRotLeftLegFlowEffortProc);
		mDataProcPipeline->addDataProc(mJointRotRightLegFlowEffortProc);
		mDataProcPipeline->addDataProc(mRotFlowEffortCombineProc);
		mDataProcPipeline->addDataProc(mJointRotSpaceEffortProc);
		mDataProcPipeline->addDataProc(mJointRotTorsoSpaceEffortProc);
		mDataProcPipeline->addDataProc(mJointRotLeftArmSpaceEffortProc);
		mDataProcPipeline->addDataProc(mJointRotRightArmSpaceEffortProc);
		mDataProcPipeline->addDataProc(mJointRotLeftLegSpaceEffortProc);
		mDataProcPipeline->addDataProc(mJointRotRightLegSpaceEffortProc);
		mDataProcPipeline->addDataProc(mRotSpaceEffortCombineProc);

		mDataProcPipeline->connect(mJointRotInputProc, mJointRotGlobalToLocalProc);

		mDataProcPipeline->connect(mJointRotInputProc, mJointRotQuatEulerProc);
		mDataProcPipeline->connect(mJointRotInputProc, mJointRotQuatSmoothProc);
		//mDataProcPipeline->connect(mJointRotGlobalToLocalProc, mJointRotQuatEulerProc);
		//mDataProcPipeline->connect(mJointRotGlobalToLocalProc, mJointRotQuatSmoothProc);

		mDataProcPipeline->connect(mJointRotQuatSmoothProc, mJointRotQuatVelocityProc);
		//mDataProcPipeline->connect(mJointRotInputProc, mJointRotQuatVelocityProc);
		mDataProcPipeline->connect(mJointRotQuatVelocityProc, mJointRotQuatVelocityEulerProc);
		mDataProcPipeline->connect(mJointRotQuatVelocityEulerProc, mJointRotVelocityScalarProc);
		mDataProcPipeline->connect(mJointRotQuatVelocityProc, mJointRotQuatVelocitySmoothProc);
		mDataProcPipeline->connect(mJointRotQuatVelocitySmoothProc, mJointRotQuatAccelerationProc);
		mDataProcPipeline->connect(mJointRotQuatAccelerationProc, mJointRotQuatAccelerationEulerProc);
		mDataProcPipeline->connect(mJointRotQuatAccelerationEulerProc, mJointRotAccelerationScalarProc);
		mDataProcPipeline->connect(mJointRotQuatAccelerationProc, mJointRotQuatAccelerationSmoothProc);
		mDataProcPipeline->connect(mJointRotQuatAccelerationSmoothProc, mJointRotQuatJerkProc);
		mDataProcPipeline->connect(mJointRotQuatJerkProc, mJointRotQuatJerkEulerProc);
		mDataProcPipeline->connect(mJointRotQuatJerkEulerProc, mJointRotJerkScalarProc);
		mDataProcPipeline->connect(mJointRotQuatJerkProc, mJointRotQuatJerkSmoothProc);

		mDataProcPipeline->connect(mJointRotQuatEulerProc, mJointRotTorsoFilterProc);
		mDataProcPipeline->connect(mJointRotQuatEulerProc, mJointRotLeftArmFilterProc);
		mDataProcPipeline->connect(mJointRotQuatEulerProc, mJointRotRightArmFilterProc);
		mDataProcPipeline->connect(mJointRotQuatEulerProc, mJointRotLeftLegFilterProc);
		mDataProcPipeline->connect(mJointRotQuatEulerProc, mJointRotRightLegFilterProc);
		mDataProcPipeline->connect(mJointRotVelocityScalarProc, mJointRotVelocityTorsoFilterProc);
		mDataProcPipeline->connect(mJointRotVelocityScalarProc, mJointRotVelocityLeftArmFilterProc);
		mDataProcPipeline->connect(mJointRotVelocityScalarProc, mJointRotVelocityRightArmFilterProc);
		mDataProcPipeline->connect(mJointRotVelocityScalarProc, mJointRotVelocityLeftLegFilterProc);
		mDataProcPipeline->connect(mJointRotVelocityScalarProc, mJointRotVelocityRightLegFilterProc);
		mDataProcPipeline->connect(mJointRotAccelerationScalarProc, mJointRotAccelerationTorsoFilterProc);
		mDataProcPipeline->connect(mJointRotAccelerationScalarProc, mJointRotAccelerationLeftArmFilterProc);
		mDataProcPipeline->connect(mJointRotAccelerationScalarProc, mJointRotAccelerationRightArmFilterProc);
		mDataProcPipeline->connect(mJointRotAccelerationScalarProc, mJointRotAccelerationLeftLegFilterProc);
		mDataProcPipeline->connect(mJointRotAccelerationScalarProc, mJointRotAccelerationRightLegFilterProc);
		mDataProcPipeline->connect(mJointRotJerkScalarProc, mJointRotJerkTorsoFilterProc);
		mDataProcPipeline->connect(mJointRotJerkScalarProc, mJointRotJerkLeftArmFilterProc);
		mDataProcPipeline->connect(mJointRotJerkScalarProc, mJointRotJerkRightArmFilterProc);
		mDataProcPipeline->connect(mJointRotJerkScalarProc, mJointRotJerkLeftLegFilterProc);
		mDataProcPipeline->connect(mJointRotJerkScalarProc, mJointRotJerkRightLegFilterProc);

		mDataProcPipeline->connect(mJointRotVelocityScalarProc, mJointRotWeightEffortProc);
		mDataProcPipeline->connect(mJointRotVelocityTorsoFilterProc, mJointRotTorsoWeightEffortProc);
		mDataProcPipeline->connect(mJointRotVelocityLeftArmFilterProc, mJointRotLeftArmWeightEffortProc);
		mDataProcPipeline->connect(mJointRotVelocityRightArmFilterProc, mJointRotRightArmWeightEffortProc);
		mDataProcPipeline->connect(mJointRotVelocityLeftLegFilterProc, mJointRotLeftLegWeightEffortProc);
		mDataProcPipeline->connect(mJointRotVelocityRightLegFilterProc, mJointRotRightLegWeightEffortProc);
		mDataProcPipeline->connect(mJointRotWeightEffortProc, mRotWeightEffortCombineProc);
		mDataProcPipeline->connect(mJointRotTorsoWeightEffortProc, mRotWeightEffortCombineProc);
		mDataProcPipeline->connect(mJointRotLeftArmWeightEffortProc, mRotWeightEffortCombineProc);
		mDataProcPipeline->connect(mJointRotRightArmWeightEffortProc, mRotWeightEffortCombineProc);
		mDataProcPipeline->connect(mJointRotLeftLegWeightEffortProc, mRotWeightEffortCombineProc);
		mDataProcPipeline->connect(mJointRotRightLegWeightEffortProc, mRotWeightEffortCombineProc);

		mDataProcPipeline->connect(mJointRotAccelerationScalarProc, mJointRotTimeEffortProc);
		mDataProcPipeline->connect(mJointRotAccelerationTorsoFilterProc, mJointRotTorsoTimeEffortProc);
		mDataProcPipeline->connect(mJointRotAccelerationLeftArmFilterProc, mJointRotLeftArmTimeEffortProc);
		mDataProcPipeline->connect(mJointRotAccelerationRightArmFilterProc, mJointRotRightArmTimeEffortProc);
		mDataProcPipeline->connect(mJointRotAccelerationLeftLegFilterProc, mJointRotLeftLegTimeEffortProc);
		mDataProcPipeline->connect(mJointRotAccelerationRightLegFilterProc, mJointRotRightLegTimeEffortProc);
		mDataProcPipeline->connect(mJointRotTimeEffortProc, mRotTimeEffortCombineProc);
		mDataProcPipeline->connect(mJointRotTorsoTimeEffortProc, mRotTimeEffortCombineProc);
		mDataProcPipeline->connect(mJointRotLeftArmTimeEffortProc, mRotTimeEffortCombineProc);
		mDataProcPipeline->connect(mJointRotRightArmTimeEffortProc, mRotTimeEffortCombineProc);
		mDataProcPipeline->connect(mJointRotLeftLegTimeEffortProc, mRotTimeEffortCombineProc);
		mDataProcPipeline->connect(mJointRotRightLegTimeEffortProc, mRotTimeEffortCombineProc);

		mDataProcPipeline->connect(mJointRotJerkScalarProc, mJointRotFlowEffortProc);
		mDataProcPipeline->connect(mJointRotJerkTorsoFilterProc, mJointRotTorsoFlowEffortProc);
		mDataProcPipeline->connect(mJointRotJerkLeftArmFilterProc, mJointRotLeftArmFlowEffortProc);
		mDataProcPipeline->connect(mJointRotJerkRightArmFilterProc, mJointRotRightArmFlowEffortProc);
		mDataProcPipeline->connect(mJointRotJerkLeftLegFilterProc, mJointRotLeftLegFlowEffortProc);
		mDataProcPipeline->connect(mJointRotJerkRightLegFilterProc, mJointRotRightLegFlowEffortProc);
		mDataProcPipeline->connect(mJointRotFlowEffortProc, mRotFlowEffortCombineProc);
		mDataProcPipeline->connect(mJointRotTorsoFlowEffortProc, mRotFlowEffortCombineProc);
		mDataProcPipeline->connect(mJointRotLeftArmFlowEffortProc, mRotFlowEffortCombineProc);
		mDataProcPipeline->connect(mJointRotRightArmFlowEffortProc, mRotFlowEffortCombineProc);
		mDataProcPipeline->connect(mJointRotLeftLegFlowEffortProc, mRotFlowEffortCombineProc);
		mDataProcPipeline->connect(mJointRotRightLegFlowEffortProc, mRotFlowEffortCombineProc);

		mDataProcPipeline->connect(mJointRotQuatEulerProc, mJointRotSpaceEffortProc);
		mDataProcPipeline->connect(mJointRotTorsoFilterProc, mJointRotTorsoSpaceEffortProc);
		mDataProcPipeline->connect(mJointRotLeftArmFilterProc, mJointRotLeftArmSpaceEffortProc);
		mDataProcPipeline->connect(mJointRotRightArmFilterProc, mJointRotRightArmSpaceEffortProc);
		mDataProcPipeline->connect(mJointRotLeftLegFilterProc, mJointRotLeftLegSpaceEffortProc);
		mDataProcPipeline->connect(mJointRotRightLegFilterProc, mJointRotRightLegSpaceEffortProc);

		mDataProcPipeline->connect(mJointRotSpaceEffortProc, mRotSpaceEffortCombineProc);
		mDataProcPipeline->connect(mJointRotTorsoSpaceEffortProc, mRotSpaceEffortCombineProc);
		mDataProcPipeline->connect(mJointRotLeftArmSpaceEffortProc, mRotSpaceEffortCombineProc);
		mDataProcPipeline->connect(mJointRotRightArmSpaceEffortProc, mRotSpaceEffortCombineProc);
		mDataProcPipeline->connect(mJointRotLeftLegSpaceEffortProc, mRotSpaceEffortCombineProc);
		mDataProcPipeline->connect(mJointRotRightLegSpaceEffortProc, mRotSpaceEffortCombineProc);

		// new procs for calculating linear and angular jerk from xsens kinematics data
		mJointLinAccelInputProc = std::shared_ptr<dab::DataProcInput>(new dab::DataProcInput("joint linaccel input", { mJointLinAccelData }));
		mJointAngAccelInputProc = std::shared_ptr<dab::DataProcInput>(new dab::DataProcInput("joint angaccel input", { mJointAngAccelData }));

		// smooth procs
		mJointLinAccelSmoothProc = std::shared_ptr<dab::DataProcLowPass>(new dab::DataProcLowPass("joint lin accel smooth", "jointLinAccel", 0.9));
		mJointAngAccelSmoothProc = std::shared_ptr<dab::DataProcLowPass>(new dab::DataProcLowPass("joint ang accel smooth", "jointAngAccel", 0.9));

		mJointLinJerkSmoothProc = std::shared_ptr<dab::DataProcLowPass>(new dab::DataProcLowPass("joint lin jerk smooth", "jointLinJerk", 0.9));
		mJointAngJerkSmoothProc = std::shared_ptr<dab::DataProcLowPass>(new dab::DataProcLowPass("joint ang jerk smooth", "jointAngJerk", 0.9));

		// derivative procs
		mJointLinJerkProc = std::shared_ptr<dab::DataProcDerivative>(new dab::DataProcDerivative("joint lin jerk calc", "jointLinJerk"));
		mJointAngJerkProc = std::shared_ptr<dab::DataProcDerivative>(new dab::DataProcDerivative("joint lin ang calc", "jointAngJerk"));

		// scalar procs
		mJointLinJerkScalarProc = std::shared_ptr<dab::DataProcScalar>(new dab::DataProcScalar("joint lin jerk scalar calc", "jointLinJerkScalar", dab::DataProcScalar::Max));
		mJointAngJerkScalarProc = std::shared_ptr<dab::DataProcScalar>(new dab::DataProcScalar("joint lin jerk scalar calc", "jointAngJerkScalar", dab::DataProcScalar::Max));
	
		// add procs to pipeline
		mDataProcPipeline->addDataProc(mJointLinAccelInputProc);
		mDataProcPipeline->addDataProc(mJointAngAccelInputProc);
		mDataProcPipeline->addDataProc(mJointLinAccelSmoothProc);
		mDataProcPipeline->addDataProc(mJointAngAccelSmoothProc);
		mDataProcPipeline->addDataProc(mJointLinJerkProc);
		mDataProcPipeline->addDataProc(mJointAngJerkProc);
		mDataProcPipeline->addDataProc(mJointLinJerkSmoothProc);
		mDataProcPipeline->addDataProc(mJointAngJerkSmoothProc);
		mDataProcPipeline->addDataProc(mJointLinJerkScalarProc);
		mDataProcPipeline->addDataProc(mJointAngJerkScalarProc);

		// connect procs
		mDataProcPipeline->connect(mJointLinAccelInputProc, mJointLinAccelSmoothProc);
		mDataProcPipeline->connect(mJointLinAccelSmoothProc, mJointLinJerkProc);
		mDataProcPipeline->connect(mJointLinJerkProc, mJointLinJerkSmoothProc);
		mDataProcPipeline->connect(mJointLinJerkSmoothProc, mJointLinJerkScalarProc);
		mDataProcPipeline->connect(mJointAngAccelInputProc, mJointAngAccelSmoothProc);
		mDataProcPipeline->connect(mJointAngAccelSmoothProc, mJointAngJerkProc);
		mDataProcPipeline->connect(mJointAngJerkProc, mJointAngJerkSmoothProc);
		mDataProcPipeline->connect(mJointAngJerkSmoothProc, mJointAngJerkScalarProc);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("DATAPROC ERRROR: Failed to setup data processing pipeline", __FILE__, __FUNCTION__, __LINE__);
	}
}

void 
ofApp::setupOsc() throw (dab::Exception)
{
	try
	{
		mDataMessenger = std::make_shared<dab::DataMessenger>();

		mDataMessenger->createReceiver("MocapReceiver", mOscReceivePort);
		mDataMessenger->createDataReceiver(mJointPosData, "/mocap/0/joint/pos_world", "MocapReceiver");
		mDataMessenger->createDataReceiver(mJointRotData, "/mocap/0/joint/rot_world", "MocapReceiver");

		mDataMessenger->createSender("MocapProcSender", mOscSendAddress, mOscSendPort);

		mDataMessenger->createDataSender(mJointPosData, "/mocap/0/joint/pos_world", "MocapProcSender");
		mDataMessenger->createDataSender(mJointPosGlobalToLocalProc->data()[0], "/mocap/0/joint/pos_local", "MocapProcSender");
		mDataMessenger->createDataSender(mJointVelocityProc->data()[0], "/mocap/0/joint/velocity", "MocapProcSender");
		mDataMessenger->createDataSender(mJointAccelerationProc->data()[0], "/mocap/0/joint/acceleration", "MocapProcSender");
		mDataMessenger->createDataSender(mJointJerkProc->data()[0], "/mocap/0/joint/jerk", "MocapProcSender");
		mDataMessenger->createDataSender(mWeightEffortCombineProc->data()[0], "/mocap/0/bodypart/weight_effort", "MocapProcSender");
		mDataMessenger->createDataSender(mTimeEffortCombineProc->data()[0], "/mocap/0/bodypart/time_effort", "MocapProcSender");
		mDataMessenger->createDataSender(mFlowEffortCombineProc->data()[0], "/mocap/0/bodypart/flow_effort", "MocapProcSender");
		mDataMessenger->createDataSender(mSpaceEffortCombineProc->data()[0], "/mocap/0/bodypart/space_effort", "MocapProcSender");
		mDataMessenger->createDataSender(mJointRotData, "/mocap/0/joint/rot_world", "MocapProcSender");

		// additional senders for quaternion derivatives for debug reasons
		mDataMessenger->createDataSender(mJointRotQuatVelocityProc->data()[0], "/mocap/0/joint/rot_velocity_quat", "MocapProcSender");
		mDataMessenger->createDataSender(mJointRotQuatAccelerationProc->data()[0], "/mocap/0/joint/rot_acceleration_quat", "MocapProcSender");
		mDataMessenger->createDataSender(mJointRotQuatJerkProc->data()[0], "/mocap/0/joint/rot_jerk_quat", "MocapProcSender");
		
		mDataMessenger->createDataSender(mJointRotGlobalToLocalProc->data()[0], "/mocap/0/joint/rot_local", "MocapProcSender");
		//mDataMessenger->createDataSender(mJointLocRotInputProc->data()[0], "/mocap/joint/locrot_quat", "MocapProcSender");
		//mDataMessenger->createDataSender(mJointLocRotData, "/mocap/joint/locrot_quat", "MocapProcSender");

		mDataMessenger->createDataSender(mJointRotQuatEulerProc->data()[0], "/mocap/0/joint/rot_euler", "MocapProcSender");
		mDataMessenger->createDataSender(mJointRotQuatVelocityEulerProc->data()[0], "/mocap/0/joint/rot_velocity", "MocapProcSender");
		mDataMessenger->createDataSender(mJointRotQuatAccelerationEulerProc->data()[0], "/mocap/0/joint/rot_acceleration", "MocapProcSender");
		mDataMessenger->createDataSender(mJointRotQuatJerkEulerProc->data()[0], "/mocap/0/joint/rot_jerk", "MocapProcSender");
		mDataMessenger->createDataSender(mRotWeightEffortCombineProc->data()[0], "/mocap/0/bodypart/rot_weight_effort", "MocapProcSender");
		mDataMessenger->createDataSender(mRotTimeEffortCombineProc->data()[0], "/mocap/0/bodypart/rot_time_effort", "MocapProcSender");
		mDataMessenger->createDataSender(mRotFlowEffortCombineProc->data()[0], "/mocap/0/bodypart/rot_flow_effort", "MocapProcSender");
		mDataMessenger->createDataSender(mRotSpaceEffortCombineProc->data()[0], "/mocap/0/bodypart/rot_space_effort", "MocapProcSender");

		mDataMessenger->setDataSenderActive(mJointPosData, false);
		mDataMessenger->setDataSenderActive(mJointPosGlobalToLocalProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointVelocityProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointAccelerationProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointJerkProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mWeightEffortCombineProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mTimeEffortCombineProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mFlowEffortCombineProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mSpaceEffortCombineProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointRotData, false);

		mDataMessenger->setDataSenderActive(mJointRotQuatVelocityProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointRotQuatAccelerationProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointRotQuatJerkProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointRotGlobalToLocalProc->data()[0], false);
		//mDataMessenger->setDataSenderActive(mJointLocRotInputProc->data()[0], false);
		//mDataMessenger->setDataSenderActive(mJointLocRotData, false);

		mDataMessenger->setDataSenderActive(mJointRotQuatEulerProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointRotQuatVelocityEulerProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointRotQuatAccelerationEulerProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointRotQuatJerkEulerProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mRotWeightEffortCombineProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mRotTimeEffortCombineProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mRotFlowEffortCombineProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mRotSpaceEffortCombineProc->data()[0], false);

		/*
		// new procs for calculating linear and angular jerk from xsens kinematics data
		mDataMessenger->createDataReceiver(mJointLinAccelData, "/mocap/0/joint/lin_acc", "MocapReceiver");
		mDataMessenger->createDataReceiver(mJointAngAccelData, "/mocap/0/joint/rot_acc", "MocapReceiver");

		mDataMessenger->createDataSender(mJointLinJerkScalarProc->data()[0], "/mocap/0/joint/lin_jerk", "MocapProcSender");
		mDataMessenger->createDataSender(mJointAngJerkScalarProc->data()[0], "/mocap/0/joint/ang_jerk", "MocapProcSender");

		mDataMessenger->setDataSenderActive(mJointLinJerkScalarProc->data()[0], false);
		mDataMessenger->setDataSenderActive(mJointAngJerkScalarProc->data()[0], false);
		*/

		mDataMessenger->start();
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("OSC ERROR: failed to setup osc", __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void
ofApp::setupGraphics()
{
	mDataPlots.resize(18);

	ofVec2f plotPos(-59.0, 30.0);
	ofVec2f plotSize(400.0, 200.0);
	int plotHistoryLength = 20;

	mDataPlots[0] = std::make_shared<dab::DataPlot>("Pos", mJointPosData, plotHistoryLength, plotPos, plotSize);
	mDataPlots[0]->setDataRange({ -5.0, -5.0, 0.0 }, { 5.0, 5.0, 4.0 });

	mDataPlots[1] = std::make_shared<dab::DataPlot>("Velocity", mJointVelocityProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[1]->setDataRange({ -4.0, -4.0, -4.0 }, { 4.0, 4.0, 4.0 });

	mDataPlots[2] = std::make_shared<dab::DataPlot>("Acceleration", mJointAccelerationProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[2]->setDataRange({ -15.0, -15.0, -15.0 }, { 15.0, 15.0, 15.0 });

	mDataPlots[3] = std::make_shared<dab::DataPlot>("Jerk", mJointJerkProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[3]->setDataRange({ -35.0, -35.0, -35.0 }, { 35.0, 35.0, 35.0 });

	mDataPlots[4] = std::make_shared<dab::DataPlot>("Weight Effort", mWeightEffortCombineProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[4]->setDataRange({ 0.0 }, { 1.0 });

	mDataPlots[5] = std::make_shared<dab::DataPlot>("Time Effort", mTimeEffortCombineProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[5]->setDataRange({ 0.0 }, { 5.0 });

	mDataPlots[6] = std::make_shared<dab::DataPlot>("Flow Effort", mFlowEffortCombineProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[6]->setDataRange({ 0.0 }, { 20.0 });

	mDataPlots[7] = std::make_shared<dab::DataPlot>("Space Effort", mSpaceEffortCombineProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[7]->setDataRange({ 0.0 }, { 0.3 });

	mDataPlots[8] = std::make_shared<dab::DataPlot>("Rot", mJointRotQuatEulerProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[8]->setDataRange({ -PI, -PI, -PI }, { PI, PI, PI });

	mDataPlots[9] = std::make_shared<dab::DataPlot>("Rot Velocity", mJointRotQuatVelocityEulerProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[9]->setDataRange({ -PI, -PI, -PI }, { PI, PI, PI });

	mDataPlots[10] = std::make_shared<dab::DataPlot>("Rot Acceleration", mJointRotQuatAccelerationEulerProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[10]->setDataRange({ -PI, -PI, -PI }, { PI, PI, PI });

	mDataPlots[11] = std::make_shared<dab::DataPlot>("Rot Jerk", mJointRotQuatJerkEulerProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[11]->setDataRange({ -PI, -PI, -PI }, { PI, PI, PI });

	mDataPlots[12] = std::make_shared<dab::DataPlot>("Rot Weight Effort", mRotWeightEffortCombineProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[12]->setDataRange({ 0.0 }, { 1.0 });
	//mDataPlots[12]->setDataRange({ 0.0 }, { 5.0 });

	mDataPlots[13] = std::make_shared<dab::DataPlot>("Rot Time Effort", mRotTimeEffortCombineProc->data()[0], plotHistoryLength, plotPos, plotSize);
	//mDataPlots[13]->setDataRange({ 0.0 }, { 5.0 });
	mDataPlots[13]->setDataRange({ 0.0 }, { 0.5 });

	mDataPlots[14] = std::make_shared<dab::DataPlot>("Rot Flow Effort", mRotFlowEffortCombineProc->data()[0], plotHistoryLength, plotPos, plotSize);
	//mDataPlots[14]->setDataRange({ 0.0 }, { 5.0 });
	mDataPlots[14]->setDataRange({ 0.0 }, { 0.5 });

	mDataPlots[15] = std::make_shared<dab::DataPlot>("Rot Space Effort", mRotSpaceEffortCombineProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[15]->setDataRange({ 0.0 }, { 1.0 });

	// new procs for calculating linear and angular jerk from xsens kinematics data
	mDataPlots[16] = std::make_shared<dab::DataPlot>("Lin Jerk", mJointLinJerkScalarProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[16]->setDataRange({ 0.0 }, { 100.0 });
	mDataPlots[17] = std::make_shared<dab::DataPlot>("Ang Jerk", mJointAngJerkScalarProc->data()[0], plotHistoryLength, plotPos, plotSize);
	mDataPlots[17]->setDataRange({ 0.0 }, { 500.0 });

	mActivePlotIndex = -1;
}

void 
ofApp::setupGui()
{
	mMenuPassiveColor = ofColor(255, 0, 0);
	mMenuActiveColor = ofColor(0, 255, 0);

	mGui = new ofxDatGui(10, 10);

	// create data plot menu
	int dataPlotCount = mDataPlots.size();
	std::vector<std::string> dataPlotNames(dataPlotCount + 1);

	dataPlotNames[0] = "No Plot";

	for (int pI = 0; pI < dataPlotCount; ++pI)
	{
		int dataPlotCount = mDataPlots.size();
		dataPlotNames[pI+1] = mDataPlots[pI]->title();
	}

	mPlotMenu = mGui->addDropdown("Display Data", dataPlotNames);

	//mPlotMenu = new ofxDatGuiDropdown("Display Data", dataPlotNames);
	//mPlotMenu->setPosition(10.0, 10.0);
	mPlotMenu->onDropdownEvent(this, &ofApp::onDropdownEvent);

	for (int pI = 0; pI < dataPlotCount; ++pI)
	{
		mPlotMenu->getChildAt(pI)->setStripeColor(mMenuPassiveColor);
	}

	//mPlotMenu->expand();

	//  create osc send menu

	const std::vector< std::shared_ptr<dab::DataSender> >& dataSenders = mDataMessenger->dataSenders();
	int dataSenderCount = dataSenders.size();
	std::vector<std::string> dataSenderAddressPatterns(dataSenderCount);

	for (int sI = 0; sI < dataSenderCount; ++sI)
	{
		dataSenderAddressPatterns[sI] = dataSenders[sI]->addressPattern();
	}

	mOSCMenu = mGui->addDropdown("Send Data", dataSenderAddressPatterns);

	//mOSCMenu = new ofxDatGuiDropdown("Send Data", dataSenderAddressPatterns);
	//mOSCMenu->setPosition(200.0, 10.0);
	mOSCMenu->onDropdownEvent(this, &ofApp::onDropdownEvent);

	for (int sI = 0; sI < dataSenderCount; ++sI)
	{
		mOSCMenu->getChildAt(sI)->setStripeColor(mMenuPassiveColor);
	}
}

void 
ofApp::updateDataProc() throw (dab::Exception)
{
	try
	{
		if (mDataMessenger->checkUpdated())
		{
			mDataProcPipeline->update();
			mDataMessenger->resetUpdated();
			mDataMessenger->send();
			updateGraphics();
		}
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("DATAPROC ERROR: failed to update data processing pipeline", __FILE__, __FUNCTION__, __LINE__);
	}
}

void
ofApp::updateGraphics()
{
	if (mActivePlotIndex != -1) mDataPlots[mActivePlotIndex]->update();
}

void 
ofApp::onDropdownEvent(ofxDatGuiDropdownEvent e)
{
	//std::cout << "event e parent " << e.parent << "\n";

	if (e.target == mPlotMenu)
	{
		mPlotMenu->getChildAt(mActivePlotIndex + 1)->setStripeColor(mMenuPassiveColor);

		mActivePlotIndex = e.child - 1;

		mPlotMenu->getChildAt(mActivePlotIndex + 1)->setStripeColor(mMenuActiveColor);
	}
	else if (e.target == mOSCMenu)
	{
		const std::vector< std::shared_ptr<dab::DataSender> >& dataSenders = mDataMessenger->dataSenders();
		std::shared_ptr<dab::Data> senderData = dataSenders[e.child]->data();

		bool senderActive = mDataMessenger->dataSenderActive(senderData);

		senderActive = !senderActive;

		mDataMessenger->setDataSenderActive(senderData, senderActive);

		if(senderActive == true) mOSCMenu->getChildAt(e.child)->setStripeColor(mMenuActiveColor);
		else mOSCMenu->getChildAt(e.child)->setStripeColor(mMenuPassiveColor);
	}
}
