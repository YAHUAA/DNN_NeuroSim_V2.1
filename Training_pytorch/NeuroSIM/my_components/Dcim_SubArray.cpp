#include <cmath>
#include <iostream>
#include <vector>
#include "../constant.h"
#include "../formula.h"
#include "Dcim_SubArray.h"

using namespace std;

//dcim subarray的构造函数
Dcim_SubArray::Dcim_SubArray(InputParameter& _inputParameter, Technology& _tech, MemCell& _cell):
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

void Dcim_SubArray::Initialize(int _numRow, int _numCol, double _unitWireRes){  //initialization module
    numRow = _numRow;    //import parameters
	numCol = _numCol;
	unitWireRes = _unitWireRes;
	
	double MIN_CELL_HEIGHT = MAX_TRANSISTOR_HEIGHT;  //set real layout cell height
	double MIN_CELL_WIDTH = (MIN_GAP_BET_GATE_POLY + POLY_WIDTH) * 2;  //set real layout cell width
    //if array is SRAM
    if (relaxArrayCellWidth) {  //if want to relax the cell width
        lengthRow = (double)numCol * MAX(cell.widthInFeatureSize, MIN_CELL_WIDTH) * tech.featureSize;
    } else { //if not relax the cell width
        lengthRow = (double)numCol * cell.widthInFeatureSize * tech.featureSize;
    }
    if (relaxArrayCellHeight) {  //if want to relax the cell height
        lengthCol = (double)numRow * MAX(cell.heightInFeatureSize, MIN_CELL_HEIGHT) * tech.featureSize;
    } else {  //if not relax the cell height
        lengthCol = (double)numRow * cell.heightInFeatureSize * tech.featureSize;
    }

    capRow1 = lengthRow * 0.2e-15/1e-6;	// BL for 1T1R, WL for Cross-point and SRAM
	capRow2 = lengthRow * 0.2e-15/1e-6;	// WL for 1T1R
	capCol = lengthCol * 0.2e-15/1e-6;
	
	resRow = lengthRow * unitWireRes; 
	resCol = lengthCol * unitWireRes;

    //start to initializing the subarray modules
	//if array is SRAM
    //firstly calculate the CMOS resistance and capacitance
    resCellAccess = CalculateOnResistance(cell.widthAccessCMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
    capCellAccess = CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech);
    cell.capSRAMCell = capCellAccess + CalculateDrainCap(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) + CalculateDrainCap(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, cell.widthInFeatureSize * tech.featureSize, tech) + CalculateGateCap(cell.widthSRAMCellNMOS * tech.featureSize, tech) + CalculateGateCap(cell.widthSRAMCellPMOS * tech.featureSize, tech);
    // conventionalParallel
    wlSwitchMatrix.Initialize(ROW_MODE, numRow, resRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
    if (numColMuxed>1) {
        mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resCellAccess/numRow/2, FPGA);       
        muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
    }

    double resTg = cell.resMemCellOn / numRow;
    if (cell.accessType == CMOS_access) {
		wlNewSwitchMatrix.Initialize(numRow, activityRowRead, clkFreq);         
	} else {
		wlSwitchMatrix.Initialize(ROW_MODE, numRow, resTg*numRow/numCol, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulseAVG, clkFreq);
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

    precharger.Initialize(numCol, resCol, activityColWrite, numReadCellPerOperationNeuro, numWriteCellPerOperationNeuro);
    sramWriteDriver.Initialize(numCol, activityColWrite, numWriteCellPerOperationNeuro);
    // prechargerBP.Initialize(numRow, resRow, activityColWrite, numReadCellPerOperationNeuro, numWriteCellPerOperationNeuro);
    // sramWriteDriverBP.Initialize(numRow, activityColWrite, numWriteCellPerOperationNeuro);
    initialized = true;  //finish initialization
}

void Dcim_SubArray::CalculateLatency(double columnRes, const vector<double> &columnResistance, const vector<double> &rowResistance) {   //calculate latency for different mode 
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
        int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
        int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
        
        wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
        precharger.CalculateLatency(1e20, capCol, numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
        sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
        if (numColMuxed > 1) {
            mux.CalculateLatency(0, 0, numColMuxed);
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
        double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
        double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
        tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
        double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
        double beta = 1 / (resPullDown * gm);
        double colRamp = 0;
        colDelay = horowitz(tau, beta, wlSwitchMatrix.rampOutput, &colRamp);

        readLatency = 0;
        readLatency += MAX(wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
        readLatency += precharger.readLatency;
        readLatency += colDelay;
        readLatency += multilevelSenseAmp.readLatency;
        readLatency += multilevelSAEncoder.readLatency;
        readLatency += shiftAddInput.readLatency ;
        readLatency += sarADC.readLatency;
        
        readLatencyADC = precharger.readLatency + colDelay + multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency;
        readLatencyAccum = shiftAddInput.readLatency ;
        readLatencyOther = MAX(wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);

        // Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
        
        double resPull;
        resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
        tau = resPull * cell.capSRAMCell;
        gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
        beta = 1 / (resPull * gm);
        
        writeLatency += horowitz(tau, beta, 1e20, NULL) * numWriteOperationPerRow * numRow * activityRowWrite;
        writeLatency += wlSwitchMatrix.writeLatency;
        writeLatency += precharger.writeLatency;
        writeLatency += sramWriteDriver.writeLatency;
    }
}

void Dcim_SubArray::CalculatePower(const vector<double> &columnResistance, const vector<double> &rowResistance) {
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
        readDynamicEnergy = 0;
		writeDynamicEnergy = 0;
		readDynamicEnergyArray = 0;
		
		double numReadOperationPerRow;   // average value (can be non-integer for energy calculation)
		if (numCol > numReadCellPerOperationNeuro)
			numReadOperationPerRow = numCol / numReadCellPerOperationNeuro;
		else
			numReadOperationPerRow = 1;

		double numWriteOperationPerRow;   // average value (can be non-integer for energy calculation)
		if (numCol * activityColWrite > numWriteCellPerOperationNeuro)
			numWriteOperationPerRow = numCol * activityColWrite / numWriteCellPerOperationNeuro;
		else
			numWriteOperationPerRow = 1;

        // Array leakage (assume 2 INV)
        leakage = 0;
        leakage += CalculateGateLeakage(INV, 1, cell.widthSRAMCellNMOS * tech.featureSize,
                cell.widthSRAMCellPMOS * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;
        leakage *= numRow * numCol;

        wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
        precharger.CalculatePower(numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
        sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
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
        // Array
        readDynamicEnergyArray = 0; // Just BL discharging
        writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
        // Read
        readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
        readDynamicEnergy += precharger.readDynamicEnergy;
        readDynamicEnergy += readDynamicEnergyArray;
        readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
        readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
        readDynamicEnergy += ((numColMuxed > 1)==true? (mux.readDynamicEnergy/numReadPulse):0);
        readDynamicEnergy += ((numColMuxed > 1)==true? (muxDecoder.readDynamicEnergy/numReadPulse):0);
        readDynamicEnergy +=  shiftAddInput.readDynamicEnergy;
        readDynamicEnergy += sarADC.readDynamicEnergy;

        readDynamicEnergyADC = precharger.readDynamicEnergy + readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;
        readDynamicEnergyAccum =  shiftAddInput.readDynamicEnergy;
        readDynamicEnergyOther = wlSwitchMatrix.readDynamicEnergy + ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
        
        
        // Write
        writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
        writeDynamicEnergy += precharger.writeDynamicEnergy;
        writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
        writeDynamicEnergy += writeDynamicEnergyArray;
        
        
        // Leakage
        leakage += wlSwitchMatrix.leakage;
        leakage += precharger.leakage;
        leakage += sramWriteDriver.leakage;
        leakage += multilevelSenseAmp.leakage;
        leakage += multilevelSAEncoder.leakage;
        leakage += shiftAddInput.leakage ;
    }
}

void Dcim_SubArray::PrintProperty() {
    cout << endl << endl;
    cout << "Array:" << endl;
    cout << "Area = " << heightArray*1e6 << "um x " << widthArray*1e6 << "um = " << areaArray*1e12 << "um^2" << endl;
    cout << "Read Dynamic Energy = " << readDynamicEnergyArray*1e12 << "pJ" << endl;
    //cout << "Write Dynamic Energy = " << writeDynamicEnergyArray*1e12 << "pJ" << endl;
    
    precharger.PrintProperty("precharger");
    sramWriteDriver.PrintProperty("sramWriteDriver");

    wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
    multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
    multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
    if (numReadPulse > 1) {
       
        shiftAddInput.PrintProperty("shiftAddInput");
    }

    FunctionUnit::PrintProperty("Dcim_SubArray");
	cout << "Used Area = " << usedArea*1e12 << "um^2" << endl;
	cout << "Empty Area = " << emptyArea*1e12 << "um^2" << endl;
}

void Dcim_SubArray::CalculateArea() {  //calculate layout area for total design
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;  //ensure initialization first
	} else {  //if initialized, start to do calculation
		area = 0;
		usedArea = 0;
		
		// Array only
		heightArray = lengthCol;
		widthArray = lengthRow;
		areaArray = heightArray * widthArray;
		
		wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
        if (numColMuxed>1) {
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
        
        height = precharger.height + sramWriteDriver.height + heightArray + multilevelSenseAmp.height + multilevelSAEncoder.height + shiftAddInput.height  + mux.height + sarADC.height;
        width = MAX(wlSwitchMatrix.width, muxDecoder.width) + widthArray;
        usedArea = areaArray + wlSwitchMatrix.area + precharger.area + sramWriteDriver.area + multilevelSenseAmp.area + multilevelSAEncoder.area + shiftAddInput.area + mux.area + muxDecoder.area + sarADC.area;
        
        areaADC = multilevelSenseAmp.area + precharger.area + multilevelSAEncoder.area + sarADC.area;
        areaAccum = shiftAddInput.area ;
        areaOther = wlSwitchMatrix.area + sramWriteDriver.area + mux.area + muxDecoder.area;
        
        area = height * width;
        emptyArea = area - usedArea;
			 
	}
}