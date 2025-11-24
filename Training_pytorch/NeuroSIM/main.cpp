/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cctype>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Tile.h"
#include "Chip.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "Definition.h"

using namespace std;

vector<vector<double> > getNetStructure(const string &inputfile);
vector<vector<int>> parseLayerSplit(const string &filePath, size_t expectedLayers);

int main(int argc, char * argv[]) {   

	auto start = chrono::high_resolution_clock::now();
	
	gen.seed(0);
	
	vector<vector<double> > netStructure;
	//netStructure = getNetStructure(argv[2]);
	netStructure = getNetStructure("./NetWork_resnet50.csv");
	
	// define weight/input/memory precision from wrapper
	// param->synapseBit = atoi(argv[3]);             		 // precision of synapse weight
	// param->numBitInput = atoi(argv[4]);            		 // precision of input neural activation

	param->synapseBit = 8;             		 // precision of synapse weight
	param->numBitInput = 8;
	
	if (param->cellBit > param->synapseBit) {
		cout << "ERROR!: Memory precision is even higher than synapse precision, please modify 'cellBit' in Param.cpp!" << endl;
		param->cellBit = param->synapseBit;
	}

	//DEBUG:
	cout << "Starting NeuroSim main.cpp..." << endl;
	
	/*** initialize operationMode as default ***/
	param->conventionalParallel = 0;
	param->conventionalSequential = 0;
	param->BNNparallelMode = 0;                // parallel BNN
	param->BNNsequentialMode = 0;              // sequential BNN
	param->XNORsequentialMode = 0;           // Use several multi-bit RRAM as one synapse
	param->XNORparallelMode = 0;         // Use several multi-bit RRAM as one synapse
	switch(param->operationmode) {
		case 6:	    param->XNORparallelMode = 1;               break;     
		case 5:	    param->XNORsequentialMode = 1;             break;     
		case 4:	    param->BNNparallelMode = 1;                break;     
		case 3:	    param->BNNsequentialMode = 1;              break;    
		case 2:	    param->conventionalParallel = 1;           break;     
		case 1:	    param->conventionalSequential = 1;         break;    
		case -1:	break;
		default:	exit(-1);
	}
	
	if (param->XNORparallelMode || param->XNORsequentialMode) {
		param->numRowPerSynapse = 2;
	} else {
		param->numRowPerSynapse = 1;
	}
	if (param->BNNparallelMode) {
		param->numColPerSynapse = 2;
	} else if (param->XNORparallelMode || param->XNORsequentialMode || param->BNNsequentialMode) {
		param->numColPerSynapse = 1;
	} else {
		param->numColPerSynapse = ceil((double)param->synapseBit/(double)param->cellBit); 
	}

	param->numRowPerSynapse_2 = param->numRowPerSynapse;
	if (param->cellBit_2 <= 0) {
		param->cellBit_2 = param->cellBit;
	}
	if (param->synapseBit_2 <= 0) {
		param->synapseBit_2 = param->synapseBit;
	}
	if (param->cellBit_2 > param->synapseBit_2) {
		param->cellBit_2 = param->synapseBit_2;
	}
	param->numColPerSynapse_2 = max(1, (int)ceil((double)param->synapseBit_2/(double)max(param->cellBit_2, 1)));
	
	switch(param->transistortype) {
		case 3:	    inputParameter.transistorType = TFET;          break;
		case 2:	    inputParameter.transistorType = FET_2D;        break;
		case 1:	    inputParameter.transistorType = conventional;  break;
		case -1:	break;
		default:	exit(-1);
	}
	
	switch(param->deviceroadmap) {
		case 2:	    inputParameter.deviceRoadmap = LSTP;  break;
		case 1:	    inputParameter.deviceRoadmap = HP;    break;
		case -1:	break;
		default:	exit(-1);
	}
	
	/* Create SubArray object and link the required global objects (not initialization) */
	inputParameter.temperature = param->temp;   // Temperature (K)
	inputParameter.processNode = param->technode;    // Technology node
	tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);

	double maxPESizeNM = 1, maxTileSizeCM = 1, numPENM = 1;

	vector<int> markNM;
	vector<int> pipelineSpeedUp;
	markNM = ChipDesignInitialize(inputParameter, tech, cell, false, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
	pipelineSpeedUp = ChipDesignInitialize(inputParameter, tech, cell, true, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
	
	double desiredNumTileNM = 1, desiredPESizeNM = 1, desiredNumTileCM = 1, desiredTileSizeCM = 1, desiredPESizeCM = 1;
	int numTileRow = 1, numTileCol = 1;
	int numArrayWriteParallel = 1;
	
	vector<vector<double> > numTileEachLayer;
	vector<vector<double> > utilizationEachLayer;
	vector<vector<double> > speedUpEachLayer;
	vector<vector<double> > tileLocaEachLayer;
	
	size_t numLayers = netStructure.size();
	numTileEachLayer.assign(2, vector<double>(numLayers, 1.0));      // 2 rows (row/col), numLayers columns, default 1.0
	utilizationEachLayer.assign(numLayers, vector<double>(1, 0.5));  // numLayers rows, 1 column, default 0.5
	speedUpEachLayer.assign(2, vector<double>(numLayers, 1.0));
	tileLocaEachLayer.assign(2, vector<double>(numLayers, 2.0));
	
	double numComputation = 0;
	for (int i=0; i<netStructure.size(); i++) {
		numComputation += 2*(netStructure[i][0] * netStructure[i][1] * netStructure[i][2] * netStructure[i][3] * netStructure[i][4] * netStructure[i][5]);
	}
	
	double numSubarrayRow_a, numSubarrayCol_a,
			numPEsperTile_a,
			numPEs_a, numTiles_a,
			numSubarrayRow_d, numSubarrayCol_d,
			numPEsperTile_d,
			numPEs_d, numTiles_d;

	int numTileRow_a, numTileCol_a, numTileRow_d, numTileCol_d;
	numTileRow_a = sqrt(param->numTiles_1);
	numTileCol_a = sqrt(param->numTiles_1);
	numPEsperTile_a = param->numPEperTile_1;
	numTileRow_d = sqrt(param->numTiles_2);
	numTileCol_d = sqrt(param->numTiles_2);
	numPEsperTile_d = param->numPEperTile_2;
	numTiles_a = param->numTiles_1;
	numTiles_d = param->numTiles_2;
	double numSubarraysperPE_a = param->numSubArrayperPE_1;
	double numSubarraysperPE_d = param->numSubArrayperPE_2;

	// 打印netStructure, markNM, numTileEachLayer
	// for (size_t i = 0; i < netStructure.size(); ++i) {
	// 	cout << "Layer " << i+1 << ": ";
	// 	for (size_t j = 0; j < netStructure[i].size(); ++j) {
	// 		cout << netStructure[i][j] << " ";
	// 	}
	// 	// print markNM safely if available
	// 	cout << "| markNM: ";
	// 	if (i < markNM.size()) {
	// 		cout << markNM[i];
	// 	} else {
	// 		cout << "N/A";
	// 	}
	// 	cout << endl;
	// }
	HybridChipInitialize(inputParameter, tech, cell, cell_2, netStructure, markNM,
		 numTileRow_a,  numTileCol_a,  numTileRow_d,  numTileCol_d,
		 numSubarraysperPE_a,  numPEsperTile_a, numTiles_a,
		 numSubarraysperPE_d,  numPEsperTile_d, numTiles_d,			
		&numArrayWriteParallel);
	
	double chipHeight, chipWidth, chipArea, chipAreaIC, chipAreaADC, chipAreaAccum, chipAreaOther, chipAreaWG, chipAreaArray;
	double CMTileheight = 0;
	double CMTilewidth = 0;
	double NMTileheight = 0;
	double NMTilewidth = 0;


	
	
	
	cout << "-------------------------------------- Hardware Performance --------------------------------------" <<  endl;
	
	

	ofstream layerfile;
	string layerfile_name = "/home/zjh/Project/MY_Project/DNN_NeuroSim_V2.1/Training_pytorch/NeuroSIM/result_log/resnet50_layer_result.txt";
	
	//DEBUG: 读取每层的数模划分阈值
	vector<vector<int>> splitBit;

	// Parse splitBit from command line arguments if provided
	// if (argc > 1) {
	// 	// Expecting argv[1] to be a string like "[1,1,1,1,1,1,1,1]"
	// 	string splitBitStr = argv[1];
		
	// 	// Remove brackets and whitespace
	// 	splitBitStr.erase(remove(splitBitStr.begin(), splitBitStr.end(), '['), splitBitStr.end());
	// 	splitBitStr.erase(remove(splitBitStr.begin(), splitBitStr.end(), ']'), splitBitStr.end());
	// 	splitBitStr.erase(remove(splitBitStr.begin(), splitBitStr.end(), ' '), splitBitStr.end());
		
	// 	stringstream ss(splitBitStr);
	// 	string token;
		
	// 	while (getline(ss, token, ',')) {
	// 		if (!token.empty()) {
	// 			splitBit.push_back(stoi(token));
	// 		}
	// 	}
		
	// 	if (splitBit.size() == netStructure.size()) {
	// 		cout << "Loaded splitBit from command line: [";
	// 		for (size_t i = 0; i < splitBit.size(); ++i) {
	// 			if (i > 0) cout << ",";
	// 			cout << splitBit[i];
	// 		}
	// 		cout << "]" << endl;
	// 	} else {
	// 		cerr << "Warning: splitBit size (" << splitBit.size() 
	// 			 << ") doesn't match network layers (" << netStructure.size() << ")" << endl;
	// 		splitBit.assign(netStructure.size(), 4);
	// 	}
	// } else {
	// 	// No command line argument provided, use default
	// 	cout << "No splitBit provided, using default configuration" << endl;
	// 	splitBit.assign(netStructure.size(), 4);
	// }
	
	//splitBit = parseLayerSplit("/home/zjh/Project/MY_Project/DNN_NeuroSim_V2.1/Training_pytorch/NeuroSIM/resnet50_layer_split_bit.txt", netStructure.size());
	splitBit = parseLayerSplit("/home/zjh/Project/MY_Project/DNN_NeuroSim_V2.1/Training_pytorch/NeuroSIM/result_log/da_thresholds.txt", netStructure.size());
	// 终止运行
	// cout << " splitBit size: " << splitBit.size() << endl;
	// cout << "netStructure size: " << netStructure.size() << endl;
	// cout << "netStructure size: " << netStructure[0].size() << endl;

	HybridChipCalculateArea(inputParameter, tech, cell, cell_2, numPEsperTile_a, numPEsperTile_d);
	// layer-by-layer process
	// show the detailed hardware performance for each layer
	for (int k=0; k<splitBit.size(); k++) {
		double chipReadLatency = 0;
		double chipReadDynamicEnergy = 0;
		double chipReadLatencyAG = 0;
		double chipReadDynamicEnergyAG = 0;
		double chipReadLatencyWG = 0;
		double chipReadDynamicEnergyWG = 0;
		double chipWriteLatencyWU = 0;
		double chipWriteDynamicEnergyWU = 0;
		
		double chipReadLatencyPeakFW = 0;
		double chipReadDynamicEnergyPeakFW = 0;
		double chipReadLatencyPeakAG = 0;
		double chipReadDynamicEnergyPeakAG = 0;
		double chipReadLatencyPeakWG = 0;
		double chipReadDynamicEnergyPeakWG = 0;
		double chipWriteLatencyPeakWU = 0;
		double chipWriteDynamicEnergyPeakWU = 0;
		
		double chipLeakageEnergy = 0;
		double chipLeakage = 0;
		double chipbufferLatency = 0;
		double chipbufferReadDynamicEnergy = 0;
		double chipicLatency = 0;
		double chipicReadDynamicEnergy = 0;
		
		double chipLatencyADC = 0;
		double chipLatencyAccum = 0;
		double chipLatencyOther = 0;
		double chipEnergyADC = 0;
		double chipEnergyAccum = 0;
		double chipEnergyOther = 0;
		
		double chipDRAMLatency = 0;
		double chipDRAMDynamicEnergy = 0;
		
		double layerReadLatency = 0;
		double layerReadDynamicEnergy = 0;
		double layerReadLatencyAG = 0;
		double layerReadDynamicEnergyAG = 0;
		double layerReadLatencyWG = 0;
		double layerReadDynamicEnergyWG = 0;
		double layerWriteLatencyWU = 0;
		double layerWriteDynamicEnergyWU = 0;
		
		double layerReadLatencyPeakFW = 0;
		double layerReadDynamicEnergyPeakFW = 0;
		double layerReadLatencyPeakAG = 0;
		double layerReadDynamicEnergyPeakAG = 0;
		double layerReadLatencyPeakWG = 0;
		double layerReadDynamicEnergyPeakWG = 0;
		double layerWriteLatencyPeakWU = 0;
		double layerWriteDynamicEnergyPeakWU = 0;
		
		double layerDRAMLatency = 0;
		double layerDRAMDynamicEnergy = 0;
		
		double tileLeakage = 0;
		double layerbufferLatency = 0;
		double layerbufferDynamicEnergy = 0;
		double layericLatency = 0;
		double layericDynamicEnergy = 0;
		
		double coreLatencyADC = 0;
		double coreLatencyAccum = 0;
		double coreLatencyOther = 0;
		double coreEnergyADC = 0;
		double coreEnergyAccum = 0;
		double coreEnergyOther = 0;
	for (int i=0; i<netStructure.size(); i++) {
		//cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
		
		// param->activityRowReadWG = atof(argv[4*i+8]);
		//             param->activityRowWriteWG = atof(argv[4*i+8]);
		//             param->activityColWriteWG = atof(argv[4*i+8]);

		param->activityRowReadWG = 0.50;
		param->activityRowWriteWG = 0.50;
		param->activityColWriteWG = 0.50;

		// int totalTileCount = static_cast<int>(ceil(numTileEachLayer[0][i]) * ceil(numTileEachLayer[1][i]));
		// if (totalTileCount <= 0) {
		// 	totalTileCount = 1;
		// }

		HybridChipCalculatePerformance(inputParameter, tech, cell,cell_2, i, netStructure[i][6],
					netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
					numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth, numArrayWriteParallel,
					&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerReadLatencyAG, &layerReadDynamicEnergyAG, &layerReadLatencyWG, &layerReadDynamicEnergyWG, 
					&layerWriteLatencyWU, &layerWriteDynamicEnergyWU, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
					&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, &layerDRAMLatency, &layerDRAMDynamicEnergy,
					&layerReadLatencyPeakFW, &layerReadDynamicEnergyPeakFW, &layerReadLatencyPeakAG, &layerReadDynamicEnergyPeakAG,
					&layerReadLatencyPeakWG, &layerReadDynamicEnergyPeakWG, &layerWriteLatencyPeakWU, &layerWriteDynamicEnergyPeakWU,
					splitBit[k]);
		
		double numTileOtherLayer = 0;
		double layerLeakageEnergy = 0;		
		// for (int j=0; j<netStructure.size(); j++) {
		// 	if (j != i) {
		// 		numTileOtherLayer += numTileEachLayer[0][j] * numTileEachLayer[1][j];
		// 	}
		// }
		// layerLeakageEnergy = numTileOtherLayer*tileLeakage*(layerReadLatency+layerReadLatencyAG);
		layerLeakageEnergy = tileLeakage;
		
		// Write to layer result file
		layerfile.open(layerfile_name, ios::app);  // append mode
		if (layerfile.is_open()) {
			// Write splitBit information at the beginning of file (only for first layer)
			if (i == 0) {
				layerfile << "Layer split bits: [";
				for (size_t _idx = 0; _idx < splitBit[k].size(); ++_idx) {
					if (_idx) layerfile << ",";
					layerfile << splitBit[k][_idx];
				}
				layerfile << "]" << endl;
				layerfile << "Layer,Forward_Latency(ns),Forward_Energy(pJ)" << endl;
			}
			layerfile << i+1 << "," << layerReadLatency*1e9 << "ns" << "," << layerReadDynamicEnergy*1e12 << "pJ" << endl;
			layerfile.close();
		} else {
			cout << "Error: the layer result file cannot be opened!" << endl;
		}
		
		
		
		chipReadLatency += layerReadLatency;
		chipReadDynamicEnergy += layerReadDynamicEnergy;
		chipReadLatencyAG += layerReadLatencyAG;
		chipReadDynamicEnergyAG += layerReadDynamicEnergyAG;
		chipReadLatencyWG += layerReadLatencyWG;
		chipReadDynamicEnergyWG += layerReadDynamicEnergyWG;
		chipWriteLatencyWU += layerWriteLatencyWU;
		chipWriteDynamicEnergyWU += layerWriteDynamicEnergyWU;
		chipDRAMLatency += layerDRAMLatency;
		chipDRAMDynamicEnergy += layerDRAMDynamicEnergy;
		
		chipReadLatencyPeakFW += layerReadLatencyPeakFW;
		chipReadDynamicEnergyPeakFW += layerReadDynamicEnergyPeakFW;
		chipReadLatencyPeakAG += layerReadLatencyPeakAG;
		chipReadDynamicEnergyPeakAG += layerReadDynamicEnergyPeakAG;
		chipReadLatencyPeakWG += layerReadLatencyPeakWG;
		chipReadDynamicEnergyPeakWG += layerReadDynamicEnergyPeakWG;
		chipWriteLatencyPeakWU += layerWriteLatencyPeakWU;
		chipWriteDynamicEnergyPeakWU += layerWriteDynamicEnergyPeakWU;
		
		chipLeakageEnergy += layerLeakageEnergy;
		chipLeakage += tileLeakage*numTileEachLayer[0][i] * numTileEachLayer[1][i];
		chipbufferLatency += layerbufferLatency;
		chipbufferReadDynamicEnergy += layerbufferDynamicEnergy;
		chipicLatency += layericLatency;
		chipicReadDynamicEnergy += layericDynamicEnergy;
		
		chipLatencyADC += coreLatencyADC;
		chipLatencyAccum += coreLatencyAccum;
		chipLatencyOther += coreLatencyOther;
		chipEnergyADC += coreEnergyADC;
		chipEnergyAccum += coreEnergyAccum;
		chipEnergyOther += coreEnergyOther;
	}
	
	// cout << "------------------------------ Summary --------------------------------" <<  endl;
	// cout << endl;
	

	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::seconds>(stop-start);
    // cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
	// cout << "Total Run-time of NeuroSim: " << duration.count() << " seconds" << endl;
	// cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
	
	// save results to top level csv file (only total results)
	ofstream outfile;
	outfile.open ("NeuroSim_Output.csv", ios::app);
	if (outfile.is_open()) {
		outfile << chipReadLatency << "," << chipReadLatencyAG << "," << chipReadLatencyWG << "," << chipWriteLatencyWU << ",";
		outfile << chipReadDynamicEnergy << "," << chipReadDynamicEnergyAG << "," << chipReadDynamicEnergyWG << "," << chipWriteDynamicEnergyWU << ",";
		outfile << chipReadLatencyPeakFW << "," << chipReadLatencyPeakAG << "," << chipReadLatencyPeakWG << "," << chipWriteLatencyPeakWU << ",";
		outfile << chipReadDynamicEnergyPeakFW << "," << chipReadDynamicEnergyPeakAG << "," << chipReadDynamicEnergyPeakWG << "," << chipWriteDynamicEnergyPeakWU << ",";
		outfile << numComputation/((chipReadDynamicEnergy+chipLeakageEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU)*1e12) << ",";
		outfile << numComputation/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e-12 << ",";
		outfile << numComputation/((chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*1e12) << ",";
		outfile << numComputation/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e-12 << endl;
	} else {
		cout << "Error: the output file cannot be opened!" << endl;
	}
	outfile.close();

	// 将chipReadDynamicEnergy，chipReadLatency写入txt文件
	ofstream chipfile;
	string chipfile_name = "/home/zjh/Project/MY_Project/DNN_NeuroSim_V2.1/Training_pytorch/NeuroSIM/result_log/resnet50_chip_result.txt";
	chipfile.open(chipfile_name, ios::app);  // append mode
	if (chipfile.is_open()) {
		chipfile << "Layer split bits: [";
		for (size_t _idx = 0; _idx < splitBit[k].size(); ++_idx) {
			if (_idx) chipfile << ",";
			chipfile << splitBit[k][_idx];
		}
		chipfile << "]" << endl;
		chipfile << "Chip_Forward_Latency(ns), Chip_Forward_Energy(pJ)" << endl;
		chipfile << (chipReadLatency)*1e9 << "ns" << "," << (chipReadDynamicEnergy)*1e12 << "pJ" << endl;
		chipfile.close();
	} else {
		cout << "Error: the chip result file cannot be opened!" << endl;
	}

	// 将chipReadDynamicEnergy，chipReadLatency写入json文件，用于搜索框架
	ofstream jsonfile;
	string jsonfile_name = "/home/zjh/Project/MY_Project/DNN_NeuroSim_V2.1/Training_pytorch/NeuroSIM/result_log/performance.json";
	jsonfile.open(jsonfile_name);
	if (jsonfile.is_open()) {
		jsonfile << "{" << endl;
		jsonfile << "  \"chipReadLatency\": " << chipReadLatency*1e9 << "," << endl;
		jsonfile << "  \"chipReadDynamicEnergy\": " << chipReadDynamicEnergy*1e12 << endl;
		jsonfile << "}" << endl;
		jsonfile.close();
	} else {
		cout << "Error: the json file cannot be opened!" << endl;
	}
}
	// // 将chipReadDynamicEnergy，chipReadLatency写入json文件
	// ofstream chipfile;
	// string chipfile_name = "/home/zjh/Project/MY_Project/DNN_NeuroSim_V2.1/Training_pytorch/NeuroSIM/result_log/performance.json";
	// jsonfile.open(jsonfile_name);
	// if (jsonfile.is_open()) {
	// 	jsonfile << "{" << endl;
	// 	jsonfile << "  \"chipReadLatency\": " << chipReadLatency << "," << endl;
	// 	jsonfile << "  \"chipReadDynamicEnergy\": " << chipReadDynamicEnergy << endl;
	// 	jsonfile << "}" << endl;
	// 	jsonfile.close();
	// } else {
	// 	cout << "Error: the json file cannot be opened!" << endl;
	// }

	return 0;
}

vector<vector<double> > getNetStructure(const string &inputfile) {
	ifstream infile(inputfile.c_str());      
	string inputline;
	string inputval;
	
	int ROWin=0, COLin=0;      
	if (!infile.good()) {        
		cerr << "Error: the input file cannot be opened!" << endl;
		exit(1);
	}else{
		while (getline(infile, inputline, '\n')) {       
			ROWin++;                                
		}
		infile.clear();
		infile.seekg(0, ios::beg);      
		if (getline(infile, inputline, '\n')) {        
			istringstream iss (inputline);      
			while (getline(iss, inputval, ',')) {       
				COLin++;
			}
		}	
	}
	infile.clear();
	infile.seekg(0, ios::beg);          

	vector<vector<double> > netStructure;               
	for (int row=0; row<ROWin; row++) {	
		vector<double> netStructurerow;
		getline(infile, inputline, '\n');             
		istringstream iss;
		iss.str(inputline);
		for (int col=0; col<COLin; col++) {       
			while(getline(iss, inputval, ',')){	
				istringstream fs;
				fs.str(inputval);
				double f=0;
				fs >> f;				
				netStructurerow.push_back(f);			
			}			
		}		
		netStructure.push_back(netStructurerow);
	}
	infile.close();
	
	return netStructure;
}

vector<vector<int>> parseLayerSplit(const string &filePath, size_t expectedLayers) {
    vector<vector<int>> layerSplit;

    if (filePath.empty()) {
        throw runtime_error("parseLayerSplit: empty file path");
    }
    
    ifstream infile(filePath);
    if (!infile.is_open()) {
        throw runtime_error("parseLayerSplit: cannot open file: " + filePath);
    }

    string line;
    while (getline(infile, line)) {
        // Skip empty lines and lines with only whitespace
        if (line.empty() || line.find_first_not_of(" \t\r\n") == string::npos) {
            continue;
        }
        
        vector<int> row;
        stringstream ss(line);
        string token;
        
        while (getline(ss, token, ',')) {
            // Trim whitespace
            token.erase(token.begin(),
                find_if(token.begin(), token.end(), [](unsigned char ch) { return !isspace(ch); }));
            token.erase(
                find_if(token.rbegin(), token.rend(), [](unsigned char ch) { return !isspace(ch); }).base(),
                token.end());
            
            if (!token.empty()) {
                row.push_back(stoi(token));
            }
        }
        
        if (!row.empty()) {
            layerSplit.push_back(row);
        }
    }
    infile.close();

    // Print warning if size doesn't match, but don't throw error
    if (layerSplit.size() != expectedLayers) {
        cout << "Warning: parseLayerSplit expected " << expectedLayers 
             << " rows but got " << layerSplit.size() << " rows" << endl;
    }

    return layerSplit;
}
