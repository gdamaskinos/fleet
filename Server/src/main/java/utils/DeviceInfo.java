/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.jblas.DoubleMatrix;

public class DeviceInfo {

    public double availMemory;
    public int runningProcess;
    public int coreNumber;
    //private double netSpeed;
    //public double cacheSize;
    //public double cpuSpeed;
    public int threadNumberLittle;
    public int threadNumberBig;
    public double bogoMips;

    public double networkLatency = -1;
    public double sizeLatency = -1;
    public double meanSizeLatency = -1;
    public double deviceLatency = -1;
    
    public double batchSize = -1;

    public double heapSize = -1;
    public double ramSize = -1;
    public double bandwidth = -1;
    
    public double[] additionalStats = new double[24];
    
    public static final int NUMBER_OF_FEATURES = 13;
 	public double[] latencyFeatures;

    public double[] cpuMaxFrequency;
    public double[] cpuCurFrequency;

 	public String androidInfo;
	//private double target;
	
	public String id;
	
    public double sizeEnergy = 0;
    public double deviceEnergy = 0;
    public double idleEnergy = 0;

    public double deviceTotalRam = 0;
    public double deviceAvailableRam = 0;
    public double deviceCpuUsage = 0;

    public double temperature = 0;
    public double batteryLevel = 0;
    public double volt = 0;
	
	public DeviceInfo(String deviceId, DoubleMatrix deviceMat, String androidInfo) {
		int i = 0;

		this.id = deviceId;
		
		this.availMemory = deviceMat.get(0, i++);
		this.runningProcess = (int) deviceMat.get(0, i++);
		this.coreNumber = (int) deviceMat.get(0, i++);
		//this.cpuSpeed = deviceMat.get(0,3);
		this.threadNumberLittle = (int) deviceMat.get(0, i++);
		this.threadNumberBig = (int) deviceMat.get(0, i++);
		this.bogoMips = deviceMat.get(0,i++);
		
		//latency per example
		this.networkLatency = deviceMat.get(0, i++);
		this.sizeLatency = deviceMat.get(0, i++);
		this.meanSizeLatency = deviceMat.get(0, i++);
		this.deviceLatency = deviceMat.get(0, i++);
		this.heapSize = deviceMat.get(0, i++);
		this.ramSize = deviceMat.get(0, i++);
		this.bandwidth = deviceMat.get(0, i++);
		this.batchSize = deviceMat.get(0, i++);

		this.sizeEnergy = deviceMat.get(0, i++);
		this.idleEnergy = deviceMat.get(0, i++);
		this.deviceEnergy = deviceMat.get(0, i++);

		this.deviceAvailableRam = deviceMat.get(0, i++);
		this.deviceTotalRam = deviceMat.get(0, i++);
		this.deviceCpuUsage = deviceMat.get(0, i++);

		this.temperature = deviceMat.get(0, i++);
		this.batteryLevel = deviceMat.get(0, i++);
		this.volt = deviceMat.get(0, i++);

		this.cpuMaxFrequency = new double[8];
		for (int j=0; j < 8; j++){
			this.cpuMaxFrequency[j] = (int) deviceMat.get(0, i+j);
			//System.out.println("Freq: " + cpuMaxFrequency[j]);
		}
		i+=8;

		this.cpuCurFrequency = new double[8];
		for (int j=0; j < 8; j++){
			this.cpuCurFrequency[j] = (int) deviceMat.get(0, i+j);
			//System.out.println("Freq: " + cpuCurFrequency[j]);
		}
		
		this.androidInfo = androidInfo;
		
		this.latencyFeatures = new double[]{this.batchSize, this.availMemory, this.runningProcess, this.coreNumber, 
				this.bogoMips};
		this.latencyFeatures = ArrayUtils.addAll(this.latencyFeatures, this.cpuCurFrequency);
		
		this.additionalStats = new double[]{this.networkLatency, this.sizeLatency, this.meanSizeLatency, this.deviceLatency,
				this.heapSize, this.ramSize, this.bandwidth, this.sizeEnergy, this.idleEnergy, this.deviceEnergy,
                this.deviceAvailableRam, this.deviceTotalRam, this.deviceCpuUsage,
                this.temperature, this.batteryLevel, this.volt};
		this.additionalStats = ArrayUtils.addAll(this.additionalStats, this.cpuMaxFrequency);
	}

	public void setBatchSize(double batchSize) {
		this.batchSize = batchSize;
		this.latencyFeatures[0] = batchSize;
	}

	public DeviceInfo(String clientId, double[] features, double[] additionalStats, String androidInfo) {
		int i=0, j=0;
		
		this.id = clientId;
		
		this.latencyFeatures = features;
		this.additionalStats = additionalStats;
		this.androidInfo = androidInfo;
		
		this.batchSize = features[i++];
		this.availMemory = features[i++];
		this.runningProcess = (int) features[i++];
		this.coreNumber = (int) features[i++];
		this.bogoMips = features[i++];

		cpuCurFrequency = new double[8];
		for (int k=0; k < 8; k++){
			this.cpuCurFrequency[k] = (int) features[i + k];
			//System.out.println("Freq: " + cpuCurFrequency[j]);
		}
		
		this.networkLatency = additionalStats[j++];
		this.sizeLatency = additionalStats[j++];
		this.meanSizeLatency = additionalStats[j++];
		this.deviceLatency = additionalStats[j++];
		this.heapSize = additionalStats[j++];
		this.ramSize = additionalStats[j++];
		this.bandwidth = additionalStats[j++];
		
		this.sizeEnergy = additionalStats[j++];
        this.idleEnergy = additionalStats[j++];
        this.deviceEnergy = additionalStats[j++];

        this.deviceAvailableRam = additionalStats[j++];
        this.deviceTotalRam = additionalStats[j++];
        this.deviceCpuUsage = additionalStats[j++];

        this.temperature = additionalStats[j++];
        this.batteryLevel = additionalStats[j++];
        this.volt = additionalStats[j++];

        cpuMaxFrequency = new double[8];
		for (int k=0; k < 8; k++){
			this.cpuMaxFrequency[k] = (int) additionalStats[j + k];
			//System.out.println("Freq: " + cpuMaxFrequency[j]);
		}
	}
	
	/**
	 * Returns header for the toString() printing sequence
	 */
	public static String header() {
		return "clientID,afterPush,profilerEpoch," 
				+ "android_model,android_version,android_serialNumber," // android;
				+ "availableMemory(MB),runningProcesses,cores,littleThreads,bigThreads,bogoMips," // features
				+ "networkLatency(ms),sizeLatency(ms),meanSizeLatency(ms),deviceLatency(ms),heapSize(MB),ramSize(MB),bandwidth(Kbps)," // additional info
				+ "batchSize," // predictions
				+ "sizeEnergy(mAh),deviceEnergy(mAh),idleEnergy(mAh),"
                + "deviceTotalRam,deviceAvailableRam,deviceCpuUsage,"
                + "temperature,batteryLevel,volt(mV),"
                + "cpuMaxFrequency[0],cpuMaxFrequency[1],cpuMaxFrequency[2],cpuMaxFrequency[3],"
                + "cpuMaxFrequency[4],cpuMaxFrequency[5],cpuMaxFrequency[6],cpuMaxFrequency[7],"
				+ "cpuCurFrequency[0],cpuCurFrequency[1],cpuCurFrequency[2],cpuCurFrequency[3],"
				+ "cpuCurFrequency[4],cpuCurFrequency[5],cpuCurFrequency[6],cpuCurFrequency[7],cpuMaxFreqMean";
	}
	
	/**
	 * 
	 * @param profilerEpoch
	 * @param afterPush 0: stats were obtained before the computation or after; 1: obtained after
	 * @param includeHeader
	 * @return
	 */
	public String toString(int profilerEpoch, int afterPush, boolean includeHeader) {
		String infoStr = "";
        if (includeHeader) {
            infoStr  += header() + "\n";
        }
		
		infoStr += this.id + ",";
		infoStr += afterPush + ",";
		infoStr += profilerEpoch + ",";
		infoStr += this.androidInfo + ",";
		
		infoStr += this.availMemory + ",";
		infoStr += this.runningProcess + ",";
		infoStr += this.coreNumber + ",";
		//infoStr += this.cpuSpeed + ",";
		infoStr += this.threadNumberLittle + ",";
		infoStr += this.threadNumberBig + ",";
		infoStr += this.bogoMips + ",";
		
		infoStr += this.networkLatency + ",";
		infoStr += this.sizeLatency + ",";
		infoStr += this.meanSizeLatency + ",";
		infoStr += this.deviceLatency + ",";
		infoStr += this.heapSize + ",";
		infoStr += this.ramSize + ",";
		infoStr += this.bandwidth + ",";
		
		infoStr += this.batchSize + ",";
		
		infoStr += this.sizeEnergy + ",";
        infoStr += this.deviceEnergy + ",";
        infoStr += this.idleEnergy + ",";

        infoStr += this.deviceTotalRam + ",";
        infoStr += this.deviceAvailableRam + ",";
        infoStr += this.deviceCpuUsage + ",";

        infoStr += this.temperature + ",";
        infoStr += this.batteryLevel + ",";
        infoStr += this.volt + ",";

        for (int k=0; k < 8; k++){
        	infoStr += this.cpuMaxFrequency[k] + ",";
		}

        for (int k=0; k < 8; k++){
        	infoStr += this.cpuCurFrequency[k] + ",";
		}

        infoStr += "-1,"; // this feature is to be hardcoded by the phone specs
        
		return infoStr.substring(0, infoStr.length() - 1);
		
	}
}
