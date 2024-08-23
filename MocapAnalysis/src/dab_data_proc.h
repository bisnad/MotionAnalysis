/** \file dab_data_proc.h
*/

#pragma once

#include "dab_singleton.h"
#include "dab_exception.h"

namespace dab
{
	class Data;
	class DataProcPipeline;

# pragma mark DataProc definition

	class DataProc
	{
	public:
		friend class DataProcPipeline;

		DataProc();
		DataProc(const std::string& pName);
		DataProc(const DataProc& pDataProc);
		~DataProc();

		DataProc& operator=(const DataProc& pDataProc);

		virtual void init() throw (dab::Exception);
		virtual void init(const DataProc& pDataProc) throw (dab::Exception);
		virtual void reset();
		virtual void destroy();

		const std::string& name() const;
		const std::vector<std::shared_ptr<Data>>& data() const;

		bool updated() const;
		void resetUpdate();
		void update() throw (dab::Exception);
		virtual void process() throw (dab::Exception);

		virtual operator std::string() const;

		friend std::ostream& operator << (std::ostream& pOstream, const DataProc& pDataProc)
		{
			std::string info = pDataProc;
			pOstream << info;
			return pOstream;
		};

	protected:
		std::string mName;
		std::vector<std::shared_ptr<Data>> mData;
		std::vector<std::shared_ptr<DataProc>> mInputProcs;
		std::vector<std::shared_ptr<DataProc>> mOutputProcs;

		bool mInitialised;
		bool mUpdated;
	};

# pragma mark DataProcPipeline definition

	class DataProcPipeline
	{
	public:
		DataProcPipeline();
		~DataProcPipeline();

		void setUpdateInterval(double pUpdateInterval);

		void addDataProc(std::shared_ptr<DataProc> pDataProc) throw (dab::Exception);
		void connect(std::shared_ptr<DataProc> pDataProc1, std::shared_ptr<DataProc> pDataProc2) throw (dab::Exception);

		void update();

	protected:
		static double sUpdateInterval;

		double mUpdateInterval;
		double mStartTime;
		long mUpdateCounter;

		std::vector<std::shared_ptr<DataProc>> mDataProcs;

	};

};
