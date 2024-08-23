/** \file dab_data_proc.cpp
*/

#include "dab_data_proc.h"
#include "dab_data.h"
#include "dab_data_proc_input.h"
#include "ofUtils.h"
#include <memory>

using namespace dab;

#pragma mark DataProc implementation

DataProc::DataProc()
{}

DataProc::DataProc(const std::string& pName)
	: mName(pName)
	, mUpdated(false)
	, mInitialised(false)
{}

DataProc::DataProc(const DataProc& pDataProc)
	: mName(pDataProc.mName)
	, mInputProcs(pDataProc.mInputProcs)
	, mOutputProcs(pDataProc.mOutputProcs)
	, mUpdated(pDataProc.mUpdated)
{
	try
	{
		init();
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("DATAPROC ERRROR: Constructor for DataProc " + mName + " failed", __FILE__, __FUNCTION__, __LINE__);
		std::cout << e << "\n";
	}
}

DataProc::~DataProc()
{
	destroy();
}

DataProc& 
DataProc::operator=(const DataProc& pDataProc)
{
	mName = pDataProc.mName;
	mInputProcs = pDataProc.mInputProcs;
	mOutputProcs = pDataProc.mOutputProcs;
	mUpdated = pDataProc.mUpdated;

	try
	{
		init(pDataProc);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("DATAPROC ERRROR: Copy for DataProc " + mName + " failed", __FILE__, __FUNCTION__, __LINE__);
		std::cout << e << "\n";
	}
}

void 
DataProc::init() throw (dab::Exception)
{}

void 
DataProc::init(const DataProc& pDataProc) throw (dab::Exception)
{}

void 
DataProc::reset()
{
	mUpdated = false;
}

void 
DataProc::destroy()
{}

const std::string& 
DataProc::name() const
{
	return mName;
}

const std::vector<std::shared_ptr<Data>>& 
DataProc::data() const
{
	return mData;
}

bool 
DataProc::updated() const
{
	return mUpdated;
}

void 
DataProc::resetUpdate()
{
	mUpdated = false;
}

void 
DataProc::update() throw (dab::Exception)
{
	if (mUpdated == true) return;

	int inputProcCount = mInputProcs.size();
	for (int pI = 0; pI < inputProcCount; ++pI)
	{
		mInputProcs[pI]->update();
	}

	//std::cout << "DataProc " << mName << " update\n";

	try
	{
		process();
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("DATA ERROR: failed to update DataProc " + mName, __FILE__, __FUNCTION__, __LINE__);
	}

	mUpdated = true;
}

void 
DataProc::process() throw (dab::Exception)
{
	//std::cout << "DataProc " << mName << " process\n";
}

DataProc::operator std::string() const
{
	std::stringstream ss;

	ss << "DataProc:\n";
	ss << "name: " << mName << "\n";

	int inputProcCount = mInputProcs.size();
	if (inputProcCount > 0)
	{
		ss << "input procs:";
		for (int pI = 0; pI < inputProcCount; ++pI)
		{
			ss << " " << mInputProcs[pI]->name();
		}
		ss << "\n";
	}

	int outputProcCount = mOutputProcs.size();
	if (outputProcCount > 0)
	{
		ss << "output procs:";
		for (int pI = 0; pI < outputProcCount; ++pI)
		{
			ss << " " << mOutputProcs[pI]->name();
		}
		ss << "\n";
	}


	return ss.str();
}

# pragma mark DataProcPipeline implementation

double DataProcPipeline::sUpdateInterval = -1.0;

DataProcPipeline::DataProcPipeline()
	: mUpdateInterval(sUpdateInterval)
{
	mStartTime = ofGetElapsedTimef();
}

DataProcPipeline::~DataProcPipeline()
{}

void 
DataProcPipeline::setUpdateInterval(double pUpdateInterval)
{
	mUpdateInterval = pUpdateInterval;
}

void 
DataProcPipeline::addDataProc(std::shared_ptr<DataProc> pDataProc) throw (dab::Exception)
{
	if (std::find(mDataProcs.begin(), mDataProcs.end(), pDataProc) != mDataProcs.end()) throw dab::Exception("DATAPROC ERROR: DataProc " + pDataProc->name() + " is already in pipeline", __FILE__, __FUNCTION__, __LINE__);
	
	mDataProcs.push_back(pDataProc);
}

void 
DataProcPipeline::connect(std::shared_ptr<DataProc> pDataProc1, std::shared_ptr<DataProc> pDataProc2) throw (dab::Exception)
{
	// check if pDataProc1 and pDataProc2 are already connected
	if (std::find(pDataProc1->mOutputProcs.begin(), pDataProc1->mOutputProcs.end(), pDataProc2) != pDataProc1->mOutputProcs.end())
	{
		throw dab::Exception("DATA ERROR: DataProc " + pDataProc1->name() + " and DataProc " + pDataProc2->name() + " already connected", __FILE__, __FUNCTION__, __LINE__);
	}
	if (std::find(pDataProc2->mInputProcs.begin(), pDataProc2->mInputProcs.end(), pDataProc1) != pDataProc2->mInputProcs.end())
	{
		throw dab::Exception("DATA ERROR: DataProc " + pDataProc1->name() + " and DataProc " + pDataProc2->name() + " already connected", __FILE__, __FUNCTION__, __LINE__);
	}

	pDataProc1->mOutputProcs.push_back(pDataProc2);
	pDataProc2->mInputProcs.push_back(pDataProc1);
}

void 
DataProcPipeline::update()
{
	double currentTime = ofGetElapsedTimef();
	double elapsedTime = currentTime - mStartTime;
	double nextUpdateTime = static_cast<double>(mUpdateCounter) * mUpdateInterval;

	if (mUpdateInterval < 1.0 || nextUpdateTime >= elapsedTime)
	{
		//std::cout << "DataProcPipeline::update()\n";

		// reset update status for all data procs
		int procCount = mDataProcs.size();
		for (int pI = 0; pI < procCount; ++pI)
		{
			mDataProcs[pI]->resetUpdate();
		}

		// gather all data procs with zero outputs
		std::vector< std::shared_ptr<DataProc> > outputProcs;
		for (int pI = 0; pI < procCount; ++pI)
		{
			if (mDataProcs[pI]->mOutputProcs.size() == 0) outputProcs.push_back(mDataProcs[pI]);
		}

		// execute update function for all output procs
		int outputProcCount = outputProcs.size();

		//std::cout << "inputProcCount " << inputProcCount << "\n";

		for (int pI = 0; pI < outputProcCount; ++pI)
		{
			//std::cout << "pI " << pI << " outputProc " << outputProcs[pI]->name() << "\n";
			outputProcs[pI]->update();
		}

		mUpdateCounter++;
	}
}