/** \file dab_data_messenger.cpp
*/

#include "dab_data_messenger.h"

using namespace dab;

/** \file dab_data_messenger.h
*/

#pragma once

#include "dab_data_sender.h"
#include "dab_data_receiver.h"

# pragma mark DataMessenger implementation

DataMessenger::DataMessenger()
{}

DataMessenger::~DataMessenger()
{}

void 
DataMessenger::createReceiver(const std::string& pReceiverName, unsigned int pReceiverPort) throw (Exception)
{
	if (mReceivers.contains(pReceiverName)) throw Exception("OSC ERROR: receiver with name " + pReceiverName + " aleady registered", __FILE__, __FUNCTION__, __LINE__);
	// std::cout << "new receiver " <<  pReceiverName.toStdString() << "\n";

	std::shared_ptr<OscReceiver> _receiver(new OscReceiver(pReceiverName, pReceiverPort));
	mReceivers.insert(pReceiverName, _receiver);
	//_receiver->start();
}

void 
DataMessenger::createSender(const std::string& pSenderName, const std::string& pSenderIP, unsigned int pSenderPort) throw (Exception)
{
	if (mSenders.contains(pSenderName)) throw Exception("OSC ERROR: sender with name " + pSenderName + " aleady registered", __FILE__, __FUNCTION__, __LINE__);

	std::shared_ptr<OscSender> _sender(new OscSender(pSenderName, pSenderIP, pSenderPort));
	mSenders.insert(pSenderName, _sender);
}

void 
DataMessenger::createDataReceiver(std::shared_ptr<Data> pData, const std::string& pAddressPattern, const std::string& pReceiverName) throw (Exception)
{
	if(mReceivers.contains(pReceiverName) == false) throw Exception("OSC ERROR: receiver with name " + pReceiverName + " not registered", __FILE__, __FUNCTION__, __LINE__);
	if(mDataReceivers.contains(pData) == true) throw Exception("DATA ERROR: Data " + pData->name() + " is already associated with a receiver", __FILE__, __FUNCTION__, __LINE__);

	std::shared_ptr<OscReceiver> _oscReceiver = mReceivers[pReceiverName];

	std::shared_ptr<DataReceiver> _dataReceiver = std::make_shared<DataReceiver>(pData, pAddressPattern);
	mDataReceivers.insert(pData, _dataReceiver);
	_oscReceiver->registerOscListener(std::weak_ptr<DataReceiver>(_dataReceiver));
}

void 
DataMessenger::createDataSender(std::shared_ptr<Data> pData, const std::string& pAddressPattern, const std::string& pSenderName) throw (Exception)
{
	if (mSenders.contains(pSenderName) == false) throw Exception("OSC ERROR: sender with name " + pSenderName + " not registered", __FILE__, __FUNCTION__, __LINE__);
	if (mDataSenders.contains(pData) == true) throw Exception("DATA ERROR: Data " + pData->name() + " is already associated with a sender", __FILE__, __FUNCTION__, __LINE__);

	std::shared_ptr<OscSender> _oscSender = mSenders[pSenderName];

	std::shared_ptr<DataSender> _dataSender = std::make_shared<DataSender>(pData, pAddressPattern, _oscSender);
	mDataSenders.insert(pData, _dataSender);

	mDataSendersActive.insert(pData, true);
}

const std::vector< std::shared_ptr<DataSender> >& 
DataMessenger::dataSenders() const
{
	return mDataSenders.values();
}

void
DataMessenger::start()
{
	int receiverCount = mReceivers.size();
	for (int rI = 0; rI < receiverCount; ++rI)
	{
		mReceivers[rI]->start();
	}
}

bool 
DataMessenger::checkUpdated()
{
	int dataCount = mDataReceivers.size();
	for (int dI = 0; dI < dataCount; ++dI)
	{
		//std::cout << "checkUpdated " << dI << " " << mDataReceivers[dI]->data()->name() << " : " << mDataReceivers[dI]->checkUpdated() << "\n";
		//std::cout << "mDataReceivers[dI] " << mDataReceivers[dI] << " data " << mDataReceivers[dI]->data() << "\n";

		if(mDataReceivers[dI]->checkUpdated() == false) return false;
	}
	return true;
}

void 
DataMessenger::resetUpdated()
{
	int dataCount = mDataReceivers.size();
	for (int dI = 0; dI < dataCount; ++dI)
	{
		mDataReceivers[dI]->resetUpdated();
	}
}

bool 
DataMessenger::checkUpdated(std::shared_ptr<Data> pData) throw (Exception)
{
	if(mDataReceivers.contains(pData) == false) throw Exception("DATA ERROR: Data " + pData->name() + " not associated with a receiver", __FILE__, __FUNCTION__, __LINE__);

	std::cout << "checkUpdated " << pData->name() << " : " << mDataReceivers[pData]->checkUpdated() << "\n";
	std::cout << "mDataReceivers[pData] " << mDataReceivers[pData] << " data " << pData << "\n";

	return mDataReceivers[pData]->checkUpdated();
}

void 
DataMessenger::resetUpdated(std::shared_ptr<Data> pData) throw (Exception)
{
	if (mDataReceivers.contains(pData) == false) throw Exception("DATA ERROR: Data " + pData->name() + " not associated with a receiver", __FILE__, __FUNCTION__, __LINE__);

	mDataReceivers[pData]->resetUpdated();
}

bool 
DataMessenger::dataSenderActive(std::shared_ptr<Data> pData) throw (Exception)
{
	if (mDataSendersActive.contains(pData) == false) throw Exception("DATA ERROR: Data " + pData->name() + " not associated with a sender", __FILE__, __FUNCTION__, __LINE__);

	return mDataSendersActive[pData];
}

void 
DataMessenger::setDataSenderActive(std::shared_ptr<Data> pData, bool pActive) throw (Exception)
{
	if(mDataSendersActive.contains(pData) == false) throw Exception("DATA ERROR: Data " + pData->name() + " not associated with a sender", __FILE__, __FUNCTION__, __LINE__);

	//std::cout << "DataMessenger::setDataSenderActive Data " << pData->name() << " active " << pActive << "\n";

	mDataSendersActive[pData] = pActive;
}

void 
DataMessenger::send() throw (Exception)
{
	try
	{
		int dataSenderCount = mDataSenders.size();

		for (int sI = 0; sI < dataSenderCount; ++sI)
		{
			std::shared_ptr<DataSender> _dataSender = mDataSenders[sI];

			if (mDataSendersActive[_dataSender->data()] == true)
			{
				_dataSender->send();
			}
		}
	}
	catch (dab::Exception& e)
	{
		throw;
	}
}