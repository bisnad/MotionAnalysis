/** \file dab_data_plot.h
*/

#pragma once

#include "ofxGPlot.h"
#include "dab_ringbuffer.h"

namespace dab
{
	class Data;
	class DataPlot;

#pragma mark SingleDimPlot definition

	class SingleDimPlot
	{
	public:
		SingleDimPlot(std::shared_ptr<Data> pData, int pDimIndex, int pHistoryLenght, const ofVec2f& pPlotPos, const ofVec2f& pPlotSize);

		float minDataValue() const;
		float maxDataValue() const;

		void setDataRange(float pMinDataValue, float pMaxDataValue);

		void update();
		void draw();

	protected:
		std::shared_ptr<Data> mData;
		std::vector<float> mDataSingleDimValues;
		int mDimIndex;
		int mHistoryLength;
		ofVec2f mPlotPos;
		ofVec2f mPlotSize;
		std::string mFontName;
		ofTrueTypeFont mFont;

		float mMinDataValue;
		float mMaxDataValue;
		bool mDataRangeChanged;
		RingBuffer< std::vector<float> > mValueRingBuffer;

		std::shared_ptr<ofxGPlot> mPlot;
		std::vector<std::string> mDataIds;
		std::vector< std::vector<ofxGPoint> > mPlotPoints;
	};

#pragma mark DataPlot definition

	class DataPlot
	{
	public:
		DataPlot(const std::string& pTitle, std::shared_ptr<Data> pData, int pHistoryLength, const ofVec2f& pPlotPos, const ofVec2f& pPlotSize);
	
		const std::string& title() const;

		void setDataRange(const std::vector<float>& pMinDataValues, const std::vector<float>& pMaxDataValues);

		bool isSetup();
		void update();
		void draw();

	protected:

		void setup();

		std::string mTitle;
		std::string mFontName;
		ofTrueTypeFont mFont;
		ofVec2f mPlotPos;
		ofVec2f mPlotSize;

		std::shared_ptr<Data> mData;
		std::vector<float> mMinDataValues;
		std::vector<float> mMaxDataValues;
		int mHistoryLength;
		std::vector<std::shared_ptr<SingleDimPlot>> mSubPlots;

		bool mSetup = false;
		bool mLock = false;
	};

};