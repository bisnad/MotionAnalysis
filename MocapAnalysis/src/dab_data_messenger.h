/** \file dab_data_messenger.h
*/

#pragma once

#include "dab_index_map.h"
#include "dab_exception.h"
#include "dab_osc_receiver.h"
#include "dab_osc_sender.h"
#include "dab_data.h"

namespace dab
{
	class DataSender;
	class DataReceiver;

# pragma mark DataMessenger definition

	class DataMessenger
	{
	public:
		DataMessenger();
		~DataMessenger();

		void start();

		void createReceiver(const std::string& pReceiverName, unsigned int pReceiverPort) throw (Exception);
		void createSender(const std::string& pSenderName, const std::string& pSenderIP, unsigned int pSenderPort) throw (Exception);
		
		void createDataReceiver(std::shared_ptr<Data> pData, const std::string& pAddressPattern, const std::string& pReceiverName) throw (Exception);
		void createDataSender(std::shared_ptr<Data> pData, const std::string& pAddressPattern, const std::string& pSenderName) throw (Exception);

		const std::vector< std::shared_ptr<DataSender> >& dataSenders() const;

		bool checkUpdated();
		void resetUpdated();

		bool checkUpdated(std::shared_ptr<Data> pData) throw (Exception);
		void resetUpdated(std::shared_ptr<Data> pData) throw (Exception);

		bool dataSenderActive(std::shared_ptr<Data> pData) throw (Exception);
		void setDataSenderActive(std::shared_ptr<Data> pData, bool pActive) throw (Exception);

		void send() throw (Exception);

	protected:
		IndexMap<std::string, std::shared_ptr<OscReceiver> > mReceivers;
		IndexMap<std::string, std::shared_ptr<OscSender> > mSenders;
		
		IndexMap<std::shared_ptr<Data>, std::shared_ptr<DataReceiver> > mDataReceivers;
		IndexMap<std::shared_ptr<Data>, std::shared_ptr<DataSender> > mDataSenders;

		IndexMap<std::shared_ptr<Data>, bool> mDataSendersActive;
	};
};