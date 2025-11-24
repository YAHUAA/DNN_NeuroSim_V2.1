#include <cmath>
#include <iostream>
#include <vector>
#include "../constant.h"
#include "../formula.h"
#include "Acim_SubArray.h"

using namespace std;

//acim subarray的构造函数
Acim_SubArray::Acim_SubArray(InputParameter& _inputParameter, Technology& _tech, MemCell& _cell):
						inputParameter(_inputParameter), tech(_tech), cell(_cell),
						wlDecoder(_inputParameter, _tech, _cell),
						wlDecoderDriver(_inputParameter, _tech, _cell),
						wlNewDecoderDriver(_inputParameter, _tech, _cell),
						wlSwitchMatrix(_inputParameter, _tech, _cell),
						wlNewSwitchMatrix(_inputParameter, _tech, _cell),
						slSwitchMatrix(_inputParameter, _tech, _cell),
						mux(_inputParameter, _tech, _cell),
						muxDecoder(_inputParameter, _tech, _cell),
						precharger(_inputParameter, _tech, _cell),
						senseAmp(_inputParameter, _tech, _cell),
						sramWriteDriver(_inputParameter, _tech, _cell),
						rowCurrentSenseAmp(_inputParameter, _tech, _cell),
						adder(_inputParameter, _tech, _cell),
						dff(_inputParameter, _tech, _cell),
						multilevelSenseAmp(_inputParameter, _tech, _cell),
						multilevelSAEncoder(_inputParameter, _tech, _cell),
						sarADC(_inputParameter, _tech, _cell),
						shiftAddInput(_inputParameter, _tech, _cell),
						shiftAddWeight(_inputParameter, _tech, _cell),
						/* for BP (Transpose SubArray) */
						wlDecoderBP(_inputParameter, _tech, _cell),
						wlSwitchMatrixBP(_inputParameter, _tech, _cell),
						prechargerBP(_inputParameter, _tech, _cell),
						senseAmpBP(_inputParameter, _tech, _cell),
						sramWriteDriverBP(_inputParameter, _tech, _cell),
						muxBP(_inputParameter, _tech, _cell),
						muxDecoderBP(_inputParameter, _tech, _cell),
						rowCurrentSenseAmpBP(_inputParameter, _tech, _cell),
						adderBP(_inputParameter, _tech, _cell),
						dffBP(_inputParameter, _tech, _cell),
						multilevelSenseAmpBP(_inputParameter, _tech, _cell),
						multilevelSAEncoderBP(_inputParameter, _tech, _cell),
						sarADCBP(_inputParameter, _tech, _cell),
						shiftAddBPInput(_inputParameter, _tech, _cell),
						shiftAddBPWeight(_inputParameter, _tech, _cell){
	initialized = false;
	readDynamicEnergyArray = writeDynamicEnergyArray = 0;
}

void Acim_SubArray::Initialize(int _numRow, int _numCol, double _unitWireRes){  //initialization module
	
	numRow = _numRow;    //import parameters
	numCol = _numCol;
	unitWireRes = _unitWireRes;
	
	double MIN_CELL_HEIGHT = MAX_TRANSISTOR_HEIGHT;  //set real layout cell height
	double MIN_CELL_WIDTH = (MIN_GAP_BET_GATE_POLY + POLY_WIDTH) * 2;  //set real layout cell width
	//if array is RRAM
	double cellHeight = cell.heightInFeatureSize; 
	double cellWidth = cell.widthInFeatureSize;  
	if (cell.accessType == CMOS_access) {  // 1T1R
		if (relaxArrayCellWidth) {
			lengthRow = (double)numCol * MAX(cellWidth, MIN_CELL_WIDTH*2) * tech.featureSize;	// Width*2 because generally switch matrix has 2 pass gates per column, even the SL/BL driver has 2 pass gates per column in traditional 1T1R memory
		} else {
			lengthRow = (double)numCol * cellWidth * tech.featureSize;
		}
		if (relaxArrayCellHeight) {
			lengthCol = (double)numRow * MAX(cellHeight, MIN_CELL_HEIGHT) * tech.featureSize;
		} else {
			lengthCol = (double)numRow * cellHeight * tech.featureSize;
		}
	} else {	// Cross-point, if enter anything else except 'CMOS_access'
		if (relaxArrayCellWidth) {
			lengthRow = (double)numCol * MAX(cellWidth*cell.featureSize, MIN_CELL_WIDTH*2*tech.featureSize);	// Width*2 because generally switch matrix has 2 pass gates per column, even the SL/BL driver has 2 pass gates per column in traditional 1T1R memory
		} else {
			lengthRow = (double)numCol * cellWidth * cell.featureSize;
		}
		if (relaxArrayCellHeight) {
			lengthCol = (double)numRow * MAX(cellHeight*cell.featureSize, MIN_CELL_HEIGHT*tech.featureSize);
		} else {  
			lengthCol = (double)numRow * cellHeight * cell.featureSize;
		}
	}
    //finish setting array size
	
	capRow1 = lengthRow * 0.2e-15/1e-6;	// BL for 1T1R, WL for Cross-point and SRAM
	capRow2 = lengthRow * 0.2e-15/1e-6;	// WL for 1T1R
	capCol = lengthCol * 0.2e-15/1e-6;
	
	resRow = lengthRow * unitWireRes; 
	resCol = lengthCol * unitWireRes;
	
	//start to initializing the subarray modules
	if (cell.accessType == CMOS_access) {	// 1T1R
		cell.resCellAccess = cell.resistanceOn * IR_DROP_TOLERANCE;    //calculate access CMOS resistance
		cell.widthAccessCMOS = CalculateOnResistance(tech.featureSize, NMOS, inputParameter.temperature, tech) * LINEAR_REGION_RATIO / cell.resCellAccess;   //get access CMOS width
		if (cell.widthAccessCMOS > cell.widthInFeatureSize) {	// Place transistor vertically
			printf("Transistor width of 1T1R=%.2fF is larger than the assigned cell width=%.2fF in layout\n", cell.widthAccessCMOS, cell.widthInFeatureSize);
			exit(-1);
		}

		cell.resMemCellOn = cell.resCellAccess + cell.resistanceOn;        //calculate single memory cell resistance_ON
		cell.resMemCellOff = cell.resCellAccess + cell.resistanceOff;      //calculate single memory cell resistance_OFF
		cell.resMemCellAvg = cell.resCellAccess + cell.resistanceAvg;      //calculate single memory cell resistance_AVG

		capRow2 += CalculateGateCap(cell.widthAccessCMOS * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap
		capCol += CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) * numRow;	// If capCol is found to be too large, increase cell.widthInFeatureSize to relax the limit
	} else {	// Cross-point
		// The nonlinearity is from the selector, assuming RRAM itself is linear
		if (cell.nonlinearIV) {   //introduce nonlinearity to the RRAM resistance
			cell.resMemCellOn = cell.resistanceOn;
			cell.resMemCellOff = cell.resistanceOff;
			cell.resMemCellOnAtHalfVw = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage/2);
			cell.resMemCellOffAtHalfVw = NonlinearResistance(cell.resistanceOff, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage/2);
			cell.resMemCellOnAtVw = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage);
			cell.resMemCellOffAtVw = NonlinearResistance(cell.resistanceOff, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage);
			cell.resMemCellAvg = cell.resistanceAvg;
			cell.resMemCellAvgAtHalfVw = (cell.resMemCellOnAtHalfVw + cell.resMemCellOffAtHalfVw) / 2;
			cell.resMemCellAvgAtVw = (cell.resMemCellOnAtVw + cell.resMemCellOffAtVw) / 2;
		} else {  //simply assume RRAM resistance is linear
			cell.resMemCellOn = cell.resistanceOn;
			cell.resMemCellOff = cell.resistanceOff;
			cell.resMemCellOnAtHalfVw = cell.resistanceOn;
			cell.resMemCellOffAtHalfVw = cell.resistanceOff;
			cell.resMemCellOnAtVw = cell.resistanceOn;
			cell.resMemCellOffAtVw = cell.resistanceOff;
			cell.resMemCellAvg = cell.resistanceAvg;
			cell.resMemCellAvgAtHalfVw = cell.resistanceAvg;
			cell.resMemCellAvgAtVw = cell.resistanceAvg;
		}
	}
	
	double resTg = cell.resMemCellOn / numRow;
	if (cell.accessType == CMOS_access) {
		wlNewSwitchMatrix.Initialize(numRow, activityRowRead, clkFreq);         
	} else {
		wlSwitchMatrix.Initialize(ROW_MODE, numRow, resTg*numRow/numCol, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);
	}
	slSwitchMatrix.Initialize(COL_MODE, numCol, resTg*numRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);     
	if (numColMuxed>1) {
		mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resTg, FPGA);       
		muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
	}
	
	if (SARADC) {
		sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
	} else {
		multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
		multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
	}
	
	if (numReadPulse > 1) {
		shiftAddInput.Initialize(ceil(numCol/numColMuxed), log2(levelOutput)+numCellPerSynapse, clkFreq, spikingMode, numReadPulse);
	}
	
	initialized = true;  //finish initialization
}

void Acim_SubArray::CalculateLatency(double columnRes, const vector<double> &columnResistance, const vector<double> &rowResistance) {   //calculate latency for different mode 
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		writeLatency = 0;
		//RRAM
		// conventional parallel
		double capBL = lengthCol * 0.2e-15/1e-6;
		int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
		double colRamp = 0;
		double tau = (capCol)*(cell.resMemCellAvg/(numRow/2));
		colDelay = horowitz(tau, 0, 1e20, &colRamp);
		colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
		
		if (cell.accessType == CMOS_access) {
			wlNewSwitchMatrix.CalculateLatency(1e20, capRow2, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
		} else {
			wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
		}
		slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
		if (numColMuxed>1) {
			mux.CalculateLatency(colRamp, 0, numColMuxed);
			muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
		}
		if (SARADC) {
			sarADC.CalculateLatency(numColMuxed);
		} else {
			multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
			multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
		}
		
		
		if (numReadPulse > 1) {
			shiftAddInput.CalculateLatency(ceil(numColMuxed/numCellPerSynapse));		
		}
		
		// Read
		readLatency = 0;
		readLatency += MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
		readLatency += multilevelSenseAmp.readLatency;
		readLatency += multilevelSAEncoder.readLatency;
		readLatency += shiftAddInput.readLatency;
		readLatency += colDelay/numReadPulse;
		readLatency += sarADC.readLatency;
		
		readLatencyADC = multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency;
		readLatencyAccum = shiftAddInput.readLatency ;
		readLatencyOther = MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse) + colDelay/numReadPulse;

		// Write
		writeLatency = 0;
		writeLatencyArray = 0;
		writeLatencyArray += totalNumWritePulse * cell.writePulseWidth;
		writeLatency += MAX(wlNewSwitchMatrix.writeLatency + wlSwitchMatrix.writeLatency, slSwitchMatrix.writeLatency);
		writeLatency += writeLatencyArray;
	} 
}

void Acim_SubArray::CalculatePower(const vector<double> &columnResistance, const vector<double> &rowResistance) {
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
		int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
		double capBL = lengthCol * 0.2e-15/1e-6;
	
		if (cell.accessType == CMOS_access) {
			wlNewSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
		} else {
			wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
		}
		slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
		if (numColMuxed > 1) {
			mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
			muxDecoder.CalculatePower(numColMuxed, 1);
		}
		
		if (SARADC) {
			sarADC.CalculatePower(columnResistance, 1);
		} else {
			multilevelSenseAmp.CalculatePower(columnResistance, 1);
			multilevelSAEncoder.CalculatePower(numColMuxed);
		}
		
		if (numReadPulse > 1) {
			shiftAddInput.CalculatePower(ceil(numColMuxed/numCellPerSynapse));		
		}

		// Read
		readDynamicEnergyArray = 0;
		readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
		readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow * activityRowRead; // Selected WL
		readDynamicEnergyArray *= numColMuxed;
		
		readDynamicEnergy = 0;
		readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy;
		readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
		readDynamicEnergy += ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
		readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
		readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
		readDynamicEnergy +=  shiftAddInput.readDynamicEnergy;
		readDynamicEnergy += readDynamicEnergyArray;
		readDynamicEnergy += sarADC.readDynamicEnergy;
		
		readDynamicEnergyADC = readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;
		readDynamicEnergyAccum =  shiftAddInput.readDynamicEnergy;
		readDynamicEnergyOther = wlNewSwitchMatrix.readDynamicEnergy + wlSwitchMatrix.readDynamicEnergy + ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
		
		
		// Write
		writeDynamicEnergyArray = writeDynamicEnergyArray;
		writeDynamicEnergy = 0;
		writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy;
		writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
		writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
		writeDynamicEnergy += writeDynamicEnergyArray;
		
		// Leakage
		leakage += wlSwitchMatrix.leakage;
		leakage += wlNewSwitchMatrix.leakage;
		leakage += slSwitchMatrix.leakage;
		leakage += mux.leakage;
		leakage += muxDecoder.leakage;
		leakage += multilevelSenseAmp.leakage;
		leakage += multilevelSAEncoder.leakage;
		leakage +=  shiftAddInput.leakage;
	}
}

void Acim_SubArray::PrintProperty() {
	cout << endl << endl;
	cout << "Array:" << endl;
	cout << "Area = " << heightArray*1e6 << "um x " << widthArray*1e6 << "um = " << areaArray*1e12 << "um^2" << endl;
	cout << "Read Dynamic Energy = " << readDynamicEnergyArray*1e12 << "pJ" << endl;
	//cout << "Write Dynamic Energy = " << writeDynamicEnergyArray*1e12 << "pJ" << endl;
	//cout << "Write Latency = " << writeLatencyArray*1e9 << "ns" << endl;

	if (cell.accessType == CMOS_access) {
		wlNewSwitchMatrix.PrintProperty("wlNewSwitchMatrix");
	} else {
		wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
	}
	slSwitchMatrix.PrintProperty("slSwitchMatrix");
	mux.PrintProperty("mux");
	muxDecoder.PrintProperty("muxDecoder");
	multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
	multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
	if (numReadPulse > 1) {
		shiftAddInput.PrintProperty("shiftAddInput");
	}
	FunctionUnit::PrintProperty("Acim_SubArray");
	cout << "Used Area = " << usedArea*1e12 << "um^2" << endl;
	cout << "Empty Area = " << emptyArea*1e12 << "um^2" << endl;
}

void Acim_SubArray::CalculateArea() {  //calculate layout area for total design
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;  //ensure initialization first
	} else {  //if initialized, start to do calculation
		area = 0;
		usedArea = 0;
		
		// Array only
		heightArray = lengthCol;
		widthArray = lengthRow;
		areaArray = heightArray * widthArray;
		
		if (cell.accessType == CMOS_access) {
			wlNewSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
		} else {
			wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
		}
		slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
		if (numColMuxed > 1) {
			mux.CalculateArea(NULL, widthArray, NONE);
			muxDecoder.CalculateArea(NULL, NULL, NONE);
			double minMuxHeight = MAX(muxDecoder.height, mux.height);
			mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
		}
		if (SARADC) {
			sarADC.CalculateUnitArea();
			sarADC.CalculateArea(NULL, widthArray, NONE);
		} else {
			multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
			multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
		}
		
		if (numReadPulse > 1) {
			shiftAddInput.CalculateArea(NULL, widthArray, NONE);
		}
		
		
		height = slSwitchMatrix.height + heightArray + mux.height + multilevelSenseAmp.height + multilevelSAEncoder.height + shiftAddInput.height  + sarADC.height;
		width = MAX(wlNewSwitchMatrix.width + wlSwitchMatrix.width, muxDecoder.width) + widthArray;
		usedArea = areaArray + wlSwitchMatrix.area + wlNewSwitchMatrix.area + slSwitchMatrix.area + mux.area + multilevelSenseAmp.area + muxDecoder.area + multilevelSAEncoder.area + shiftAddInput.area  + sarADC.area;
		
		areaADC = multilevelSenseAmp.area + multilevelSAEncoder.area + sarADC.area;
		areaAccum = shiftAddInput.area ;
		areaOther = wlNewSwitchMatrix.area + wlSwitchMatrix.area + slSwitchMatrix.area + mux.area + muxDecoder.area;
		
		area = height * width;
		emptyArea = area - usedArea;
			 
	}
}