/** \file DataReceiver.h
*/

#pragma once

#include <iostream>
#include "dab_exception.h"
#include "dab_osc_message.h"
#include "dab_osc_receiver.h"

namespace dab
{
	class Data;

	class DataReceiver : public OscListener
	{
	public:
		DataReceiver(std::shared_ptr<Data> pData, const std::string& pAddressPattern);
		~DataReceiver();

		bool checkUpdated() const;
		void resetUpdated();
		std::shared_ptr<Data> data();

		void notify(std::shared_ptr<OscMessage> pMessage);
		void update(const std::vector<_OscArg*>& pMessageArguments) throw (Exception);

	protected:
		std::shared_ptr<Data> mData;

		std::string mAddressPattern;
		double mReceiveTime;
		double mUpdateTime;
		bool mUpdated;
	};

};
