/** \file DataSender.h
*/

#pragma once

#include <iostream>
#include "dab_exception.h"
#include "dab_osc_message.h"
#include "dab_osc_sender.h"

namespace dab
{
	class Data;

	#pragma mark DataSender definition

	class DataSender
	{
	public:
		DataSender(std::shared_ptr<Data> pData, const std::string& pAddressPattern, std::shared_ptr<OscSender> pOscSender);
		~DataSender();

		std::shared_ptr<Data> data();
		const std::string& addressPattern() const;

		void send() throw (Exception);

	protected:
		std::shared_ptr<Data> mData;
		std::string mAddressPattern;
		std::shared_ptr<OscSender> mOscSender;
	};
};
