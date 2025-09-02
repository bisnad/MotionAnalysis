/** \file DataReceiver.cpp
*/

#include "dab_data_receiver.h"
#include "dab_data.h"
#include "ofUtils.h"
#include <regex>

using namespace dab;

DataReceiver::DataReceiver(std::shared_ptr<Data> pData, const std::string& pAddressPattern)
	: mData(pData)
	, mAddressPattern(pAddressPattern)
	, mUpdated(false)
{
	ofResetElapsedTimeCounter();
	mReceiveTime = 0.0;
}

DataReceiver::~DataReceiver()
{}

bool
DataReceiver::checkUpdated() const
{
	return mUpdated;
}

void 
DataReceiver::resetUpdated()
{
	mUpdated = false;
}

std::shared_ptr<Data>
DataReceiver::data()
{
	//mUpdated = false;
	return mData;
}

bool 
DataReceiver::matchWildcard(const std::string& pPattern, const std::string& pString) 
{
	// Escape any regex special chars except '*'
	std::string regexPattern;
	for (char c : pPattern) {
		if (c == '*') {
			regexPattern += ".*";
		}
		else if (std::string(".^$|()[]{}+?\\").find(c) != std::string::npos) {
			regexPattern += '\\';
			regexPattern += c;
		}
		else {
			regexPattern += c;
		}
	}
	std::regex re(regexPattern);
	return std::regex_match(pString, re);
}

void 
DataReceiver::notify(std::shared_ptr<OscMessage> pMessage)
{
	try
	{
		//std::cout << "DataReceiver::notify\n";

		const std::string& addressPattern = pMessage->address();
		const std::vector<_OscArg*> messageArguments = pMessage->arguments();

		//std::cout << "mAddressPattern " << mAddressPattern << " addressPattern " << addressPattern << "\n";

		if (matchWildcard(mAddressPattern, addressPattern)) update(messageArguments);
		//if (addressPattern == mAddressPattern) update(messageArguments);
	}
	catch (dab::Exception& e)
	{
		std::cout << e << "\n";
	}
}

void
DataReceiver::update(const std::vector<_OscArg*>& pMessageArguments) throw (Exception)
{
	double currentTime = ofGetElapsedTimef();
	mUpdateTime = currentTime - mReceiveTime;
	mReceiveTime = currentTime;

	//std::cout << "DataReceiver Address " << mAddressPattern << " mUpdateTime " << mUpdateTime << "\n";

	std::vector<float>& dataValues = mData->values();
	int argumentCount = pMessageArguments.size();
	int dataValueCount = dataValues.size();

	//std::cout << "argumentCount " << argumentCount << " " << dataValueCount << "\n";

	if (argumentCount != dataValueCount) throw Exception("OSC ERROR: wrong argument count, expected " + std::to_string(dataValueCount) + " received " + std::to_string(argumentCount), __FILE__, __FUNCTION__, __LINE__);

	try
	{
		for (int i = 0; i < dataValueCount; ++i)
		{
			dataValues[i] = *pMessageArguments[i];
		}
	}
	catch (dab::Exception& e)
	{
		throw e;
	}

	mUpdated = true;
}
