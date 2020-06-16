/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import au.com.bytecode.opencsv.CSVReader;
import utils.DeviceInfo;
import utils.RegModel;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.ParseException;
import java.util.*;

import javax.sound.sampled.Line;

import org.bytedeco.javacpp.avcodec.AVBitStreamFilter.Init_AVBSFContext;

public class LASSOProfiler extends OldProfiler {

    private class LinearModel {
        /**
         * keeps the client trainingFeatures for updating the regression model
         */
        private ArrayList<double[]> clientFeatures;
        /**
         * keeps the client target (meanSizeLatency) for updating the regression model
         */
        private ArrayList<Double> clientTargets;
        /**
         *training data read from local file
         */
        private boolean initialDataset;
        private double[][] trainingFeatures;
        private double[] trainingTargets;
       
        private RegModel model;
        private int epoch = 0;    	
    	private int update_size = 1;	//the update_size must be lower than the window_size

        public LinearModel(String trainingDataPath) {
        	        	        
            this.clientFeatures = new ArrayList<>();
            this.clientTargets = new ArrayList<>();
        	
            if (!trainingDataPath.equals("")) {
            	initialDataset=true;
            	this.retrieveTrainingFile(trainingDataPath);
            	this.model = new RegModel(trainingFeatures, trainingTargets);
            }else {
            	initialDataset=false;
                this.trainingFeatures = new double[0][];
                this.trainingTargets = new double[0];
            }
        }

        /**
         * read the local file to construct feature set and target set
         */
        private void retrieveTrainingFile(String trainingDataPath) {

            try {
                CSVReader reader = new CSVReader(new FileReader(trainingDataPath));
                List<String[]> entries = reader.readAll();

                int size = entries.size();

                this.trainingFeatures = new double[size][];
                this.trainingTargets = new double[size];

                int index = 0;

                DecimalFormat format = new DecimalFormat("0.#", DecimalFormatSymbols.getInstance(Locale.ENGLISH));

                for (String[] entry : entries) {
                    int featureSize = entry.length - 1;
                    double[] curFeature = new double[featureSize];

                    for (int i = 0; i < featureSize; i++) {
                        try {
                            curFeature[i] = format.parse(entry[i].trim().replace("?", "")).floatValue();
                        } catch (ParseException e) {
                            e.printStackTrace();
                        }

                    }

                    this.trainingFeatures[index] = curFeature;
                    this.trainingTargets[index] = Double.parseDouble(entry[featureSize]);

                    index += 1;
                }

                reader.close();

            } catch (IOException e) {
                e.printStackTrace();
            }

        }
        
        private void updateModel() {   	
    		int batch = update_size;
    		
    		//even in batch_update mode the first updates must be single
    		if (latencyModel.epoch < 10) batch = 1;
    		//update when the new data have reached the predefined update_size 
            if (clientFeatures.size() > (batch-1)) {
                System.out.println("[LinearModel] Retrain the linear regression model.");
                reconstruct();
                clientFeatures.clear();
                clientTargets.clear();
                
                model = new RegModel(trainingFeatures, trainingTargets);
                System.out.println("[LinearModel] Finish retraining model.");
            }
        }

        /**
         * stores data for updating the model 
         * @param features
         * @param target
         */
        private void addClientData(double[] features, double target) {
            this.clientFeatures.add(features);
            this.clientTargets.add(target);
        }


        /**
         * when the number of new training samples is big enough, construct
         * the feature set and target set with new training samples
         */
        protected void reconstruct() {
            int size = clientFeatures.size();
            
            //if no initialDataset, train the model with the first client query
            //use 6 instances because a single data point is not adequate for the training process 
            if ((!initialDataset) && (epoch == 0)) {
                this.trainingFeatures = new double[DeviceInfo.NUMBER_OF_FEATURES][];
                this.trainingTargets = new double[DeviceInfo.NUMBER_OF_FEATURES];
                for (int i=0; i<trainingTargets.length; i++) {
                	trainingFeatures[i] = clientFeatures.get(0);
                	trainingTargets[i] = clientTargets.get(0);
                }
            }
            //train the model with the last seen data
            int totalSize = Math.min(WINDOW_SIZE, trainingTargets.length+size);

            double[][] nFeatures = new double[totalSize][];
            double[] nTargets = new double[totalSize];

            for (int i = 0; i < size; i++) {
                nFeatures[i] = clientFeatures.get(i);
                nTargets[i] = clientTargets.get(i); 
            }
            
            for (int i = 0; i < totalSize-size; i++) {
                nFeatures[i + size] = trainingFeatures[i];
                nTargets[i + size] = trainingTargets[i];
            }
          
            this.trainingFeatures = nFeatures;
            this.trainingTargets = nTargets;
        	System.err.println("-----------------------UPDATED Lasso w" + WINDOW_SIZE + "_" + initialDataset + " UPDATED--------------------, number of new data: " + size + ", total training data: " + totalSize);


        }

        private double predict(double[] features) {
            return this.model.getModel().predict(features);
        }
    }

    /**
     * keeps the latest info per client for serving prediction requests
     */
    private Map<String, DeviceInfo> clientInfo;
    
	/**
	 * If alignedLogins is enabled the login time and execution 
	 * duration must be defined in the constructor
	 */
	private boolean alignedLogins;
	private Map<String, Integer> loginEpoch;
	private Map<Integer, String> loginEpochReverse;
	private Map<String, Integer> logoutRequest;
	private String nextCLient;
	
	/**
	 *the number of the latest data used for training the model 
	 */
    private final int WINDOW_SIZE;

    private LinearModel latencyModel;
    private LinearModel energyModel;
    private double LATENCY_SLO;
    private double ENERGY_SLO;

    
    public LASSOProfiler(String samplesPath, double latencySLO, double energySLO, ServerSocket profilerSocket, int window, boolean alignedLogins) {
		super(profilerSocket);
        System.out.println("[LASSO Profiler] " + DeviceInfo.header());
        
        this.clientInfo = new HashMap<>();
        this.WINDOW_SIZE = window;
        
        this.ENERGY_SLO = energySLO;
        this.LATENCY_SLO = latencySLO;
        this.latencyModel = new LinearModel(samplesPath);
//        this.energyModel = new LinearModel(this.getClass().getResource("/datasets/profiler_energy_samples_mojo.csv").getFile());

		this.alignedLogins = alignedLogins;
		if (this.alignedLogins) {
			this.loginEpoch = new HashMap<String, Integer>();
			this.logoutRequest = new HashMap<String, Integer>();
			this.loginEpochReverse = new HashMap<Integer, String>();		
		
			//TODO set the login time of the client
			this.loginEpoch.put("Client1", 0);
			this.loginEpoch.put("Client2", 5);
			this.loginEpoch.put("Client3", 10);
			this.loginEpoch.put("Client4", 15);
//			this.loginEpoch.put("Client5", 40);
//			this.loginEpoch.put("Client6", 50);
//			this.loginEpoch.put("Client7", 60);
			
			//TODO set the total number of requests for each client
			this.logoutRequest.put("Client1", 20);
			this.logoutRequest.put("Client2", 20);
			this.logoutRequest.put("Client3", 20);
			this.logoutRequest.put("Client4", 20);
//			this.logoutRequest.put("Client5", 20);
//			this.logoutRequest.put("Client6", 20);
//			this.logoutRequest.put("Client7", 20);
		
			for (Map.Entry<String, Integer> entry : loginEpoch.entrySet()) {
			    loginEpochReverse.put(entry.getValue(), entry.getKey());
			}
			this.nextCLient =  (loginEpoch.isEmpty() || alignedLogins==false ? null : loginEpoch.entrySet().iterator().next().getKey());
		}
    }
    
    public void updateModels() {
        latencyModel.updateModel();
//        energyModel.updateModel();
    }
    
    public LinearModel getEnergyModel() {
    	return energyModel;
    }
    
    public String getNextClient() {
    	return nextCLient;
    }


    private void forwardBatchSize(ObjectOutputStream output, String clientId, DeviceInfo deviceInfo) throws IOException{
		
    	System.out.println("[Profiler] Forward batch size");
        int batchSize = 1;
        double predictedLatency = 0.0;
	
		int action;
		//Decides to evaluate the query or to abort
		if (nextCLient!=null) {
			if (clientId.equals(nextCLient))	action=1; 
			else	action=0;
		}
		else {
			if (alignedLogins) {
				if (latencyModel.epoch < loginEpoch.get(clientId))	action=0;
				else if (logoutRequest.get(clientId) > 0)	action = 1;
				else action = -1;
			}else action = 1;
		}
				
		if (action == 1) {
			
            if (clientInfo.isEmpty()) {
            	batchSize = 1;
                System.out.println("[Profiler] No client info for " + clientId + " forwarding default batchSize");
            }else {
            	long t = System.currentTimeMillis();
                predictedLatency = latencyModel.predict(deviceInfo.latencyFeatures); // h_theta(x)
    			System.out.println("LASSO Inference latency (ms): " + (System.currentTimeMillis() - t));
                int batchSizeLatency = (int) ((LATENCY_SLO - deviceInfo.deviceLatency) / predictedLatency); // (int) (SLO - l_D) / h_theta(x)
//                double predictedEnergy = profiler.energyModel.predict(this.getEnergyFeatures(deviceInfo)); // set predictedEnergy = h_theta(x)
//                deviceInfo.predictedMeanSizeEnergy = predictedEnergy;
//                int batchSizeEnergy = (int)((ENERGY_SLO - deviceInfo.deviceEnergy) / predictedEnergy); // (int) (SLO - l_D) / h_theta(x)

//                batchSize = Math.min(batchSizeEnergy, batchSizeLatency);
                batchSize = batchSizeLatency;
                if (batchSize < 1) batchSize = 1;   // at least 1 sample needs to be sent
            }
			
			clientInfo.put(clientId, deviceInfo);
		}

		output.writeInt(action);           
        output.writeInt(batchSize);
        output.flush();
        
    }

    private void receiveClientInfo(DeviceInfo devInfo, String clientId) throws IOException, ClassNotFoundException {
		
		if (nextCLient==null || clientId.equals(nextCLient)) {
        	long t = System.currentTimeMillis();
            latencyModel.addClientData(devInfo.latencyFeatures, devInfo.meanSizeLatency);
//            profiler.energyModel.addClientData(this.getEnergyFeatures(devInfo), devInfo.meanSizeEnergy);
            clientInfo.put(devInfo.id, devInfo);
            updateModels();
			System.out.println("LASSO Training latency (ms): " + (System.currentTimeMillis() - t));

            System.out.println("[Profiler] Complete Info:\n" + devInfo.toString(latencyModel.epoch, -1, true));
            
            //output file
            // FIXME eclipse issues (no such file) -> works on the server	            
            String outputCsvPath = "src/main/resources/output/output" + WINDOW_SIZE + ".csv";
            if (latencyModel.initialDataset) outputCsvPath = "src/main/resources/output/output" + WINDOW_SIZE + "_initialDataset.csv";
            File file = new File(outputCsvPath);
            boolean writeHeader = !file.exists();
            FileWriter writer = new FileWriter(file, true);
            writer.write(devInfo.toString(latencyModel.epoch, -1, writeHeader) + "\n");
            writer.close();
            				
			if (alignedLogins) {
				int count = logoutRequest.containsKey(clientId) ? logoutRequest.get(clientId) : 0;
				logoutRequest.put(clientId, count-1);
			}	
			nextCLient=null;
            latencyModel.epoch++;
//          energyModel.epoch++;
		}
    }

    private double [] getEnergyFeatures(DeviceInfo deviceInfo) {
        return new double[]{deviceInfo.deviceAvailableRam, deviceInfo.deviceTotalRam,
                deviceInfo.deviceCpuUsage, deviceInfo.temperature, deviceInfo.batteryLevel, deviceInfo.volt};
    }

    public void handleClientRequest(Socket socket) {
        try {
            ObjectInputStream input = new ObjectInputStream(new DataInputStream(socket.getInputStream()));
            ObjectOutputStream output = new ObjectOutputStream(new DataOutputStream(socket.getOutputStream()));

            int requestType = input.readInt();
            String clientId = (String) input.readObject();
            
            double[] latencyFeatures;
            double[] additionalStats;
            String androidInfo;
            DeviceInfo devInfo;

            switch (requestType) {
                case 0:                // receive the request to get batch size
                    latencyFeatures = (double[]) input.readObject();
                    additionalStats = (double[]) input.readObject();
                    androidInfo = (String) input.readObject();
                    devInfo = new DeviceInfo(clientId, latencyFeatures, additionalStats, androidInfo);
                    forwardBatchSize(output, clientId, devInfo);
                    break;
                case 1:             // receive the request to forward client info
                	synchronized (latencyModel) { 
                		
                    latencyFeatures = (double[]) input.readObject();
                    additionalStats = (double[]) input.readObject();
                    androidInfo = (String) input.readObject();
                    devInfo = new DeviceInfo(clientId, latencyFeatures, additionalStats, androidInfo);
                    
                    System.out.println("[Profiler] Receive client info");
                    receiveClientInfo(devInfo, clientId);

                    
                    if (alignedLogins && loginEpochReverse.containsKey(latencyModel.epoch)) nextCLient = loginEpochReverse.get(latencyModel.epoch);
                    
                    break;
                	}
                default:
                    System.out.println("[Profiler] Unknown request type!");
            }

        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        } finally {

            try {
                socket.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

    }
    
}
