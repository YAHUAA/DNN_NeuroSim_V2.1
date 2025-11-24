#include <cmath>
#include <iostream>
#include <vector>
#include "../constant.h"
#include "../formula.h"
#include "ContrateUnit.h"

using namespace std;

// ContrateUnit::~ContrateUnit() {}

ContrateUnit::ContrateUnit(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), dff(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void ContrateUnit::Initialize(int _numUnit, int _numBit, double _clkFreq) {
	if (initialized)
		cout << "[ContrateUnit] Warning: Already initialized!" << endl;

	numUnit = _numUnit;
	numBit = _numBit;
	clkFreq = _clkFreq;
	numDff = numBit * numUnit;	
	
	dff.Initialize(numDff, clkFreq);
	
	initialized = true;
}

void ContrateUnit::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[BitShifter] Error: Require initialization first!" << endl;
	} else {
		double hInv, wInv, hNand, wNand;
		area = 0;
		height = 0;
		width = 0;
		dff.CalculateArea(NULL, NULL, NONE);
		area = dff.area;
		
		if (_newWidth && _option==NONE) {
			width = _newWidth;
			height = area/width;
		} else {
			height = _newHeight;
            width = area/height;
		}
		
		// Modify layout
		newHeight = _newHeight;
		newWidth = _newWidth;
		switch (_option) {
			case MAGIC:
				MagicLayout();
				break;
			case OVERRIDE:
				OverrideLayout();
				break;  
			default:    // NONE
				break;
		}

	}
}

void ContrateUnit::CalculateLatency(double numShiftBits) {
	if (!initialized) {
		cout << "[BitShifter] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		dff.CalculateLatency(1e20, 1);
		readLatency += dff.readLatency; // read out parallely
		readLatency *= numShiftBits;    
	}
}

void ContrateUnit::CalculatePower(double numShiftBits) {
	if (!initialized) {
		cout << "[ContrateUnit] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		
		dff.CalculatePower(numShiftBits, numDff);	
		readDynamicEnergy += dff.readDynamicEnergy;
		leakage += dff.leakage;
	}
}

void ContrateUnit::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}