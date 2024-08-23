/** \file dab_data_plot.cpp
*/

#include "dab_data_plot.h"
#include "dab_data.h"
#include "ofColor.h"

using namespace dab;

#pragma mark SingleDimPlot implementation

SingleDimPlot::SingleDimPlot(std::shared_ptr<Data> pData, int pDimIndex, int pHistoryLenght, const ofVec2f& pPlotPos, const ofVec2f& pPlotSize)
	: mData(pData)
	, mDimIndex(pDimIndex)
	, mHistoryLength(pHistoryLenght)
	, mPlotPos(pPlotPos)
	, mPlotSize(pPlotSize)
	, mMinDataValue(FLT_MAX)
	, mMaxDataValue(-FLT_MAX)
	, mFontName("Arial")
	, mPlotPoints(mData->valueCount(), std::vector<ofxGPoint>(mHistoryLength))
	, mValueRingBuffer(std::vector<float>(mData->valueCount(), 0.0), mHistoryLength)
{
	int valueCount = mData->valueCount();

	mDataSingleDimValues.resize(valueCount, 0.0);
	mDataIds.resize(valueCount);
	for (int vI = 0; vI < valueCount; ++vI) mDataIds[vI] = std::to_string(vI + 1);

	mPlot = std::make_shared<ofxGPlot>();
	mPlot->setPos(mPlotPos.x, mPlotPos.y);
	mPlot->setDim(mPlotSize.x, mPlotSize.y);
	mPlot->setXLim(0.0, (float)(mHistoryLength - 1));
	mPlot->setYLim(mMinDataValue, mMaxDataValue);
	mPlot->setBoxBgColor(ofColor(255, 255, 255, 255));
	mPlot->setBoxLineColor(ofColor(0, 0, 0, 55));
	mPlot->setLineColor(ofColor(0, 0, 0, 200));
	mPlot->setGridLineColor(ofColor(0, 0, 0, 200));
	mPlot->setLineWidth(2);
	mPlot->setFontName(mFontName);
	mPlot->setFontSize(16);

	for (int vI = 0; vI < valueCount; ++vI)
	{
		mPlot->addLayer(mDataIds[vI], mPlotPoints[vI]);
	}

	std::array<float, 3> hsl = { 0.0, 1.0, 0.5 };
	float hsl_h_incr = 1.0 / (float)(valueCount + 1);

	for (int vI = 0; vI < valueCount; ++vI)
	{
		ofColor rgb = ofFloatColor::fromHsb(hsl[0], hsl[1], hsl[2]);
		//std::cout << "h " << hsl[0] << " r " << rgb[0] << " g " << rgb[1] << " b " << rgb[2] << "\n";

		hsl[0] += hsl_h_incr;

		mPlot->getLayer(mDataIds[vI]).setLineWidth(2);
		mPlot->getLayer(mDataIds[vI]).setLineColor(ofColor(rgb[0], rgb[1], rgb[2], 150));
	}
}

float 
SingleDimPlot::minDataValue() const
{
	return mMinDataValue;
}

float
SingleDimPlot::maxDataValue() const
{
	return mMaxDataValue;
}

void 
SingleDimPlot::setDataRange(float pMinDataValue, float pMaxDataValue)
{
	mMinDataValue = pMinDataValue;
	mMaxDataValue = std::max(pMaxDataValue, pMinDataValue + 0.0001f);
	mDataRangeChanged = true;
}

void
SingleDimPlot::update()
{
	int valueDim = mData->valueDim();
	int valueCount = mData->valueCount();
	const std::vector<float>& dataValues = mData->values();

	for (int vI = 0, dI = mDimIndex; vI < valueCount; ++vI, dI += valueDim)
	{
		mDataSingleDimValues[vI] = dataValues[dI];
	}

	mValueRingBuffer.update(mDataSingleDimValues);

	ofxGPoint plotPoint(0.0, 0.0);

	for (int hI = 0; hI < mHistoryLength; ++hI)
	{
		const std::vector<float>& historyValues = mValueRingBuffer[hI];

		for (int vI = 0; vI < valueCount; ++vI)
		{
			plotPoint.setX((float)hI);
			plotPoint.setY(historyValues[vI]);
			mPlotPoints[vI][hI] = plotPoint;

			//std::cout << "hI " << hI << " vI " << vI << " point " << plotPoint.getX() << " " << plotPoint.getY() << "\n";
		}
	}
}

void 
SingleDimPlot::draw()
{
	if (mDataRangeChanged == true)
	{
		mPlot->setYLim(mMinDataValue, mMaxDataValue);
		mDataRangeChanged = false;
	}

	int valueCount = mData->valueCount();
	for (int vI = 0; vI < valueCount; ++vI)
	{
		mPlot->setPoints(mPlotPoints[vI], mDataIds[vI]);
	}

	mPlot->beginDraw();
	mPlot->drawBox();
	//mPlot->drawXAxis();
	//mPlot->drawYAxis();
	//mPlot->drawTitle();
	//mPlot->drawGridLines(GRAFICA_VERTICAL_DIRECTION);
	mPlot->drawLines();
	//mPlot->drawLabels();
	mPlot->endDraw();
}


#pragma mark DataPlot implementation

DataPlot::DataPlot(const std::string& pTitle, std::shared_ptr<Data> pData, int pHistoryLength, const ofVec2f& pPlotPos, const ofVec2f& pPlotSize)
	: mTitle(pTitle)
	, mData(pData)
	, mHistoryLength(pHistoryLength)
	, mPlotPos(pPlotPos)
	, mPlotSize(pPlotSize)
	, mFontName("Arial")
{
	mFont.loadFont(mFontName, 12);
}

const std::string& 
DataPlot::title() const
{
	return mTitle;
}

bool 
DataPlot::isSetup()
{
	return mSetup;
}

void 
DataPlot::setup()
{
	if (mSetup == true) return;

	mLock = true;

	int valueDim = mData->valueDim();
	int valueCount = mData->valueCount();
	int arraySize = valueDim * valueCount;

	ofVec2f subPlotPos = mPlotPos;

	for (int d = 0; d < valueDim; ++d)
	{
		std::shared_ptr<SingleDimPlot> _subPlot = std::make_shared<SingleDimPlot>(mData, d, mHistoryLength, subPlotPos, mPlotSize);
		mSubPlots.push_back(_subPlot);

		subPlotPos.y += mPlotSize.y + 40.0;
	}

	if (mMinDataValues.size() == valueDim && mMaxDataValues.size() == valueDim)
	{
		for (int d = 0; d < valueDim; ++d)
		{
			mSubPlots[d]->setDataRange(mMinDataValues[d], mMaxDataValues[d]);
		}
	}

	mSetup = true;
	mLock = false;
}

void 
DataPlot::setDataRange(const std::vector<float>& pMinDataValues, const std::vector<float>& pMaxDataValues)
{
	mMinDataValues = pMinDataValues;
	mMaxDataValues = pMaxDataValues;

	int subplotCount = mSubPlots.size();

	if (pMinDataValues.size() != subplotCount || pMaxDataValues.size() != subplotCount) return;

	for (int pI = 0; pI < subplotCount; ++pI)
	{
		mSubPlots[pI]->setDataRange(pMinDataValues[pI], pMaxDataValues[pI]);
	}
}

void 
DataPlot::update()
{
	if (mLock == true) return;

	int valueDim = mData->valueDim();

	if (mSubPlots.size() != valueDim)
	{
		setup();
	}

	for (int d = 0; d < valueDim; ++d)
	{
		mSubPlots[d]->update();
	}
}

void 
DataPlot::draw()
{
	if (mLock == true) return;

	//ofPushMatrix();
	//ofTranslate(mPlotPos.x + 10, mPlotPos.y + 10 );
	////ofRotate(-90.0);
	//ofSetColor(0, 0, 0, 255);
	//mFont.drawString(mTitle, 0, 0);
	//ofSetColor(255, 255, 255, 255);
	//ofPopMatrix();

	int subPlotCount = mSubPlots.size();

	for (int d = 0; d < subPlotCount; ++d)
	{
		mSubPlots[d]->draw();
	}
}