/** \file DataSender.cpp
*/

#include "dab_data_sender.h"
#include "dab_data.h"
#include "dab_osc_message.h"

using namespace dab;

#pragma mark DataSender implementation

DataSender::DataSender(std::shared_ptr<Data> pData, const std::string& pAddressPattern, std::shared_ptr<OscSender> pOscSender)
	: mData(pData)
	, mAddressPattern(pAddressPattern)
	, mOscSender(pOscSender)
{}

DataSender::~DataSender()
{}

std::shared_ptr<Data> 
DataSender::data()
{
	return mData;
}

const std::string& 
DataSender::addressPattern() const
{
	return mAddressPattern;
}

void 
DataSender::send() throw (Exception)
{
	try
	{
		std::shared_ptr<OscMessage> _oscMessage = std::make_shared<OscMessage>(mAddressPattern);
		const std::vector<float>& dataValues = mData->values();
		int dataSize = dataValues.size();
		for (int dI = 0; dI < dataSize; ++dI)
		{
			_oscMessage->add(dataValues[dI]);
		}

		mOscSender->send(_oscMessage);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("DATA ERROR: failed to send data via osc " + mData->name(), __FILE__, __FUNCTION__, __LINE__);
	}
}
