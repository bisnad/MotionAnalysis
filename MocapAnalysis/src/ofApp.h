/*
* Same as regular MocapAnalysis
* but also receives linear and angular acceleration from XSens MVN Analyze Pro
* and derives from this linear and angular jerk
*/

#pragma once

#include "ofMain.h"
#include "dab_osc_sender.h"
#include "dab_osc_receiver.h"
#include "dab_data.h"
#include "dab_data_proc.h"
#include "dab_data_proc_input.h"
#include "dab_data_proc_scale.h"
#include "dab_data_proc_lowpass.h"
#include "dab_data_proc_lowpass_quat.h"
#include "dab_data_proc_derivative.h"
#include "dab_data_proc_derivative_euler.h"
#include "dab_data_proc_derivative_quat.h"
#include "dab_data_proc_quat_euler.h"
#include "dab_data_proc_scalar.h"
#include "dab_data_proc_filter.h"
#include "dab_data_proc_combine.h"
#include "dab_data_proc_weight_effort.h"
#include "dab_data_proc_time_effort.h"
#include "dab_data_proc_flow_effort.h"
#include "dab_data_proc_space_effort.h"
#include "dab_data_proc_space_effort_2.h"
#include "dab_data_proc_pos_global_to_local.h"
#include "dab_data_proc_rot_global_to_local.h"
#include "dab_data_receiver.h"
#include "dab_data_messenger.h"
#include "dab_data_plot.h"
#include "ofxDatGui.h"


class ofApp : public ofBaseApp
{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

	protected:
		// config
		void loadConfig(const std::string& pFileName) throw (dab::Exception);
		void loadJointBodyPartFilters(const std::string& pFileName) throw (dab::Exception);
		void loadJointWeights(const std::string& pFileName) throw (dab::Exception);
		void loadJointConnectivity(const std::string& pFileName) throw (dab::Exception);

		// Data
		std::vector<int> mTorsoJointIndices;
		std::vector<int> mLeftArmJointIndices;
		std::vector<int> mRightArmJointIndices;
		std::vector<int> mLeftLegJointIndices;
		std::vector<int> mRightLegJointIndices;
		std::vector<float> mJointWeights;
		std::vector<float> mTorsoJointWeights;
		std::vector<float> mLeftArmJointWeights;
		std::vector<float> mRightArmJointWeights;
		std::vector<float> mLeftLegJointWeights;
		std::vector<float> mRightLegJointWeights;
		std::vector< std::vector<int> > mJointConnectivity;
		std::shared_ptr<dab::Data> mMarkerPosData;
		std::shared_ptr<dab::Data> mJointPosData;
		std::shared_ptr<dab::Data> mJointRotData;

		void setupData();

		// Data Processing
		std::shared_ptr<dab::DataProcPipeline> mDataProcPipeline;
		std::shared_ptr<dab::DataProcInput> mJointPosInputProc; // global
		std::shared_ptr<dab::DataProcInput> mJointRotInputProc; // global

		std::shared_ptr<dab::DataProcPosGlobalToLocal> mJointPosGlobalToLocalProc;

		std::shared_ptr<dab::DataProcScale> mJointPosScaleProc;
		std::shared_ptr<dab::DataProcLowPass> mJointPosSmoothProc;
		std::shared_ptr<dab::DataProcDerivative> mJointVelocityProc;
		std::shared_ptr<dab::DataProcLowPass> mJointVelocitySmoothProc;
		std::shared_ptr<dab::DataProcScalar> mJointVelocityScalarProc;
		std::shared_ptr<dab::DataProcDerivative> mJointAccelerationProc;
		std::shared_ptr<dab::DataProcLowPass> mJointAccelerationSmoothProc;
		std::shared_ptr<dab::DataProcScalar> mJointAccelerationScalarProc;
		std::shared_ptr<dab::DataProcDerivative> mJointJerkProc;
		std::shared_ptr<dab::DataProcLowPass> mJointJerkSmoothProc;
		std::shared_ptr<dab::DataProcScalar> mJointJerkScalarProc;

		std::shared_ptr<dab::DataProcFilter> mJointPosTorsoFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointPosLeftArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointPosRightArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointPosLeftLegFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointPosRightLegFilterProc;

		std::shared_ptr<dab::DataProcFilter> mJointVelocityTorsoFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointVelocityLeftArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointVelocityRightArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointVelocityLeftLegFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointVelocityRightLegFilterProc;

		std::shared_ptr<dab::DataProcFilter> mJointAccelerationTorsoFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointAccelerationLeftArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointAccelerationRightArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointAccelerationLeftLegFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointAccelerationRightLegFilterProc;

		std::shared_ptr<dab::DataProcFilter> mJointJerkTorsoFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointJerkLeftArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointJerkRightArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointJerkLeftLegFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointJerkRightLegFilterProc;

		std::shared_ptr<dab::DataProcWeightEffort> mJointWeightEffortProc;
		std::shared_ptr<dab::DataProcWeightEffort> mJointTorsoWeightEffortProc;
		std::shared_ptr<dab::DataProcWeightEffort> mJointLeftArmWeightEffortProc;
		std::shared_ptr<dab::DataProcWeightEffort> mJointRightArmWeightEffortProc;
		std::shared_ptr<dab::DataProcWeightEffort> mJointLeftLegWeightEffortProc;
		std::shared_ptr<dab::DataProcWeightEffort> mJointRightLegWeightEffortProc;
		std::shared_ptr<dab::DataProcCombine> mWeightEffortCombineProc;

		std::shared_ptr<dab::DataProcTimeEffort> mJointTimeEffortProc;
		std::shared_ptr<dab::DataProcTimeEffort> mJointTorsoTimeEffortProc;
		std::shared_ptr<dab::DataProcTimeEffort> mJointLeftArmTimeEffortProc;
		std::shared_ptr<dab::DataProcTimeEffort> mJointRightArmTimeEffortProc;
		std::shared_ptr<dab::DataProcTimeEffort> mJointLeftLegTimeEffortProc;
		std::shared_ptr<dab::DataProcTimeEffort> mJointRightLegTimeEffortProc;
		std::shared_ptr<dab::DataProcCombine> mTimeEffortCombineProc;

		std::shared_ptr<dab::DataProcFlowEffort> mJointFlowEffortProc;
		std::shared_ptr<dab::DataProcFlowEffort> mJointTorsoFlowEffortProc;
		std::shared_ptr<dab::DataProcFlowEffort> mJointLeftArmFlowEffortProc;
		std::shared_ptr<dab::DataProcFlowEffort> mJointRightArmFlowEffortProc;
		std::shared_ptr<dab::DataProcFlowEffort> mJointLeftLegFlowEffortProc;
		std::shared_ptr<dab::DataProcFlowEffort> mJointRightLegFlowEffortProc;
		std::shared_ptr<dab::DataProcCombine> mFlowEffortCombineProc;

		std::shared_ptr<dab::DataProcSpaceEffort2> mJointSpaceEffortProc;
		std::shared_ptr<dab::DataProcSpaceEffort2> mJointTorsoSpaceEffortProc;
		std::shared_ptr<dab::DataProcSpaceEffort2> mJointLeftArmSpaceEffortProc;
		std::shared_ptr<dab::DataProcSpaceEffort2> mJointRightArmSpaceEffortProc;
		std::shared_ptr<dab::DataProcSpaceEffort2> mJointLeftLegSpaceEffortProc;
		std::shared_ptr<dab::DataProcSpaceEffort2> mJointRightLegSpaceEffortProc;
		std::shared_ptr<dab::DataProcCombine> mSpaceEffortCombineProc;

		std::shared_ptr<dab::DataProcRotGlobalToLocal> mJointRotGlobalToLocalProc;

		std::shared_ptr<dab::DataProcQuatEuler> mJointRotQuatEulerProc;
		std::shared_ptr<dab::DataProcLowPassQuat> mJointRotQuatSmoothProc;

		std::shared_ptr<dab::DataProcDerivativeQuat> mJointRotQuatVelocityProc;
		std::shared_ptr<dab::DataProcLowPassQuat> mJointRotQuatVelocitySmoothProc;
		std::shared_ptr<dab::DataProcQuatEuler> mJointRotQuatVelocityEulerProc;
		std::shared_ptr<dab::DataProcScalar> mJointRotVelocityScalarProc;

		std::shared_ptr<dab::DataProcDerivativeQuat> mJointRotQuatAccelerationProc;
		std::shared_ptr<dab::DataProcLowPassQuat> mJointRotQuatAccelerationSmoothProc;
		std::shared_ptr<dab::DataProcQuatEuler> mJointRotQuatAccelerationEulerProc;
		std::shared_ptr<dab::DataProcScalar> mJointRotAccelerationScalarProc;

		std::shared_ptr<dab::DataProcDerivativeQuat> mJointRotQuatJerkProc;
		std::shared_ptr<dab::DataProcLowPassQuat> mJointRotQuatJerkSmoothProc;
		std::shared_ptr<dab::DataProcQuatEuler> mJointRotQuatJerkEulerProc;
		std::shared_ptr<dab::DataProcScalar> mJointRotJerkScalarProc;

		std::shared_ptr<dab::DataProcFilter> mJointRotTorsoFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotLeftArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotRightArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotLeftLegFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotRightLegFilterProc;

		std::shared_ptr<dab::DataProcFilter> mJointRotVelocityTorsoFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotVelocityLeftArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotVelocityRightArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotVelocityLeftLegFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotVelocityRightLegFilterProc;

		std::shared_ptr<dab::DataProcFilter> mJointRotAccelerationTorsoFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotAccelerationLeftArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotAccelerationRightArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotAccelerationLeftLegFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotAccelerationRightLegFilterProc;

		std::shared_ptr<dab::DataProcFilter> mJointRotJerkTorsoFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotJerkLeftArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotJerkRightArmFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotJerkLeftLegFilterProc;
		std::shared_ptr<dab::DataProcFilter> mJointRotJerkRightLegFilterProc;

		std::shared_ptr<dab::DataProcWeightEffort> mJointRotWeightEffortProc;
		std::shared_ptr<dab::DataProcWeightEffort> mJointRotTorsoWeightEffortProc;
		std::shared_ptr<dab::DataProcWeightEffort> mJointRotLeftArmWeightEffortProc;
		std::shared_ptr<dab::DataProcWeightEffort> mJointRotRightArmWeightEffortProc;
		std::shared_ptr<dab::DataProcWeightEffort> mJointRotLeftLegWeightEffortProc;
		std::shared_ptr<dab::DataProcWeightEffort> mJointRotRightLegWeightEffortProc;
		std::shared_ptr<dab::DataProcCombine> mRotWeightEffortCombineProc;

		std::shared_ptr<dab::DataProcTimeEffort> mJointRotTimeEffortProc;
		std::shared_ptr<dab::DataProcTimeEffort> mJointRotTorsoTimeEffortProc;
		std::shared_ptr<dab::DataProcTimeEffort> mJointRotLeftArmTimeEffortProc;
		std::shared_ptr<dab::DataProcTimeEffort> mJointRotRightArmTimeEffortProc;
		std::shared_ptr<dab::DataProcTimeEffort> mJointRotLeftLegTimeEffortProc;
		std::shared_ptr<dab::DataProcTimeEffort> mJointRotRightLegTimeEffortProc;
		std::shared_ptr<dab::DataProcCombine> mRotTimeEffortCombineProc;

		std::shared_ptr<dab::DataProcFlowEffort> mJointRotFlowEffortProc;
		std::shared_ptr<dab::DataProcFlowEffort> mJointRotTorsoFlowEffortProc;
		std::shared_ptr<dab::DataProcFlowEffort> mJointRotLeftArmFlowEffortProc;
		std::shared_ptr<dab::DataProcFlowEffort> mJointRotRightArmFlowEffortProc;
		std::shared_ptr<dab::DataProcFlowEffort> mJointRotLeftLegFlowEffortProc;
		std::shared_ptr<dab::DataProcFlowEffort> mJointRotRightLegFlowEffortProc;
		std::shared_ptr<dab::DataProcCombine> mRotFlowEffortCombineProc;

		std::shared_ptr<dab::DataProcSpaceEffort2> mJointRotSpaceEffortProc;
		std::shared_ptr<dab::DataProcSpaceEffort2> mJointRotTorsoSpaceEffortProc;
		std::shared_ptr<dab::DataProcSpaceEffort2> mJointRotLeftArmSpaceEffortProc;
		std::shared_ptr<dab::DataProcSpaceEffort2> mJointRotRightArmSpaceEffortProc;
		std::shared_ptr<dab::DataProcSpaceEffort2> mJointRotLeftLegSpaceEffortProc;
		std::shared_ptr<dab::DataProcSpaceEffort2> mJointRotRightLegSpaceEffortProc;
		std::shared_ptr<dab::DataProcCombine> mRotSpaceEffortCombineProc;

		// new procs for calculating linear and angular jerk from xsens kinematics data
		std::shared_ptr<dab::Data> mJointLinAccelData;
		std::shared_ptr<dab::Data> mJointAngAccelData;

		std::shared_ptr<dab::DataProcInput> mJointLinAccelInputProc;
		std::shared_ptr<dab::DataProcInput> mJointAngAccelInputProc;

		std::shared_ptr<dab::DataProcLowPass> mJointLinAccelSmoothProc;
		std::shared_ptr<dab::DataProcDerivative> mJointLinJerkProc;
		std::shared_ptr<dab::DataProcLowPass> mJointLinJerkSmoothProc;
		std::shared_ptr<dab::DataProcScalar> mJointLinJerkScalarProc;

		std::shared_ptr<dab::DataProcLowPass> mJointAngAccelSmoothProc;
		std::shared_ptr<dab::DataProcDerivative> mJointAngJerkProc;
		std::shared_ptr<dab::DataProcLowPass> mJointAngJerkSmoothProc;
		std::shared_ptr<dab::DataProcScalar> mJointAngJerkScalarProc;

		void setupDataProc() throw (dab::Exception);
		void updateDataProc() throw (dab::Exception);

		// Osc
		int mOscReceivePort;
		std::string mOscSendAddress;
		int mOscSendPort;

		std::shared_ptr<dab::DataMessenger> mDataMessenger;
		std::shared_ptr<dab::OscReceiver> mOscReceiver;
		std::shared_ptr<dab::OscSender> mOscSender;
		std::shared_ptr<dab::DataReceiver> mMarkerPosDataReceiver;
		std::shared_ptr<dab::DataReceiver> mJointPosDataReceiver;
		std::shared_ptr<dab::DataReceiver> mJointRotDataReceiver;

		void setupOsc() throw (dab::Exception);
		
		// Graphics
		int mActivePlotIndex = 0;
		std::vector<std::shared_ptr<dab::DataPlot>> mDataPlots;

		void setupGraphics();
		void updateGraphics();

		// Gui
		ofxDatGui* mGui;
		ofxDatGuiDropdown* mPlotMenu;
		ofxDatGuiDropdown* mOSCMenu;
		ofColor mMenuPassiveColor;
		ofColor mMenuActiveColor;

		void setupGui();
		void onDropdownEvent(ofxDatGuiDropdownEvent e);
};
