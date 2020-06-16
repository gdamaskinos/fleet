/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import java.util.Map;

import org.bytedeco.javacpp.opencv_stitching.FeaturesMatcher;

import au.com.bytecode.opencsv.CSVReader;
import jsat.classifiers.linear.PassiveAggressive.Mode;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import utils.PAModel;
import utils.DeviceInfo;

public class PAProfiler extends OldProfiler {

	/**
	 * Passive-Aggressive Model
	 */
	private PAModel model;
	private Mode mode; 
	private int profilerEpoch = 0;
	private String PA_name;
	
	//FIXME Either remove it or make the required modifications in PAProfler methods 
	/**
	 * If enabled the PAProfiler it tunes the aggressiveness parameter C 
	 * when multiple logins occur simultaneously. Currently disabled - 
	 * PAProfiler operates desirably without altering the C value while running 
	 */
	private boolean hp_tuning;
	
	/**
	 * Features and statistics
	 */
	private ArrayList<Double> clientTargets;
	private ArrayList<double[]> clientFeatures;
	public Map<String, DeviceInfo> clientInfo;
	private double[] targets;
	private double[][] features;
	
	/**
	 * If alignedLogins is enabled the login time and execution 
	 * duration for every client must be defined in the constructor
	 */
	private boolean alignedLogins;
	private Map<String, Integer> loginEpoch;
	private Map<Integer, String> loginEpochReverse;
	private Map<String, Integer> logoutRequest;
	private String nextCLient;
	
	/**
	 * normalized features 
	 */
	private ArrayList<ArrayList<Double>> featureMinMax;
	private boolean normalize;
	
	/**
	 * batch updates
	 */
	private int update_size;
	static int singe_update = 3;	

	/**
	 * if enabled, the features sent with the client's query are stored and used
	 * for the model update otherwise, the ones at client's stats response are used
	 */
	private boolean updateBefore;
	
	// Misc
	private double SLO;
	private String filePath;
	
	
	public PAProfiler(String samplesPath, double latency, ServerSocket profilerSocket, Mode mode, boolean hp_tuning, int update_size,
			double epsilon, double default_C, String PA_name, boolean updateBefore, boolean normalize, boolean alignedLogins) {
		super(profilerSocket);
        System.out.println("[PA Profiler] " + DeviceInfo.header());


		this.PA_name = PA_name;
		this.mode = mode;
		this.hp_tuning = hp_tuning;
		this.model = new PAModel(50, epsilon, default_C, mode, hp_tuning);
		
		this.clientInfo = new HashMap<String, DeviceInfo>();
		this.clientFeatures = new ArrayList<double[]> ();
		this.clientTargets = new ArrayList<Double>();		
			
		this.alignedLogins = alignedLogins;
		if (alignedLogins) {
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
		
		this.updateBefore = updateBefore;
		this.update_size = update_size;
		
		this.SLO = latency;
		this.filePath = samplesPath;
						
		this.normalize = normalize;
		if (normalize) featureMinMax = new ArrayList<>();		
			
		// Model init and pre-train
		if (!filePath.equals("")) {
			System.err.println("Loading initial Dataset");
			this.retrieveTrainingFile();
			model.initialize(features, targets);
		}

	}
	
	public PAProfiler( double latency, ServerSocket profilerSocket, Mode mode) {
		this("", latency, profilerSocket, mode, false, 1, 0.01, 0.0001, "", true, false, false);
	}
	
	public PAProfiler( double latency, ServerSocket profilerSocket) {
		this("", latency, profilerSocket, Mode.PA, false, 1, 0.01, 0.0001, "", true, false, false);
	}	
	
    public String getNextClient() {
    	return nextCLient;
    }
	
    /**
     * updates the PA model with the last seen features-target pair for a given client
     * 
     * @param clientId
     */
	public void learn(String clientId) {
		if (clientInfo.containsKey(clientId)) {
			long t = System.currentTimeMillis();
			DeviceInfo d = clientInfo.get(clientId);
			model.update(d.latencyFeatures, d.meanSizeLatency);
			System.out.println("PA Training latency (ms): " + (System.currentTimeMillis() - t));
            System.err.println("------------>UPDATED "+ PA_name + " UPDATED, new data: 1");
		}       
	}
	
	/**
	 * updates the PA model with the features-targets pairs of the evaluated queries
	 * 
	 */
	public void learn() {
		int batch = update_size;
		
		//single update for the first 3 queries even in batch update mode
		if (profilerEpoch < 3) batch = 1;			
        if (clientFeatures.size() > (batch-1)) {
			long t = System.currentTimeMillis();
			int count = 0;
			for (int i=0; i<clientTargets.size(); i++) {
				model.update(clientFeatures.get(i), clientTargets.get(i));
				count++;
			}
            clientFeatures.clear();
            clientTargets.clear();
			System.out.println("PA Training latency (ms): " + (System.currentTimeMillis() - t));
            System.err.println("------------>UPDATED "+ PA_name + " UPDATED, new data: " + count);
        }
	}

	/**
	 * Sets a filename for the output file
	 * based on the fields of the PA instance 
	 * 
	 * @return the filename
	 */
	private String logFileName() {
		String name;
		switch (mode) {
		case PA1:
			name = "/pa1_output";
			break;
			
		case PA2:
			name = "/pa2_output";
			break;
			
		case PA:
			name = "/pa_output";
			break;
			
		default:
			name = "/pa_output";
			break;
		}

		if (PA_name.length()>0) name = name + "_" + PA_name;
				
		return name + (hp_tuning ? "_tuning" : "") + ".csv";
	}

	
	private void retrieveTrainingFile() {
		try {
			CSVReader reader = new CSVReader(new FileReader(this.filePath));
			List<String[]> entries = reader.readAll();
			int size = entries.size();
			int index = 0;

			this.features = new double[size][];
			this.targets = new double[size];

			for (String[] entry: entries) {
				int featureSize = entry.length - 1;
				double[] curFeature = new double[featureSize];

				for (int i = 0; i < featureSize; i++)
					curFeature[i] = Double.parseDouble(entry[i]);

				this.features[index] = curFeature;
				this.targets[index] = Double.parseDouble(entry[featureSize]);

				index += 1;
			}

			reader.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Gets the min and max values of an attribute and 
	 * computes the normalized value in the range (0,1)
	 * 
	 * @param value
	 * @param min
	 * @param max
	 * @return normalized value
	 */
    private double minMaxScale(double value, double min, double max) {       	
    	return (value-min)/(max-min);
    }
    
    /** 
     * Handles the client queries
     * case 0: computes the batch size
     * case 1: stores the statistics
     * This method accesses the profiler model, clientInfo, featureMinMax
     */
    public void handleClientRequest(Socket socket) {
		try {
			ObjectInputStream input = new ObjectInputStream(new DataInputStream(socket.getInputStream()));
			ObjectOutputStream output = new ObjectOutputStream(new DataOutputStream(socket.getOutputStream()));
			
			int batchSize = 1;					
			int action = 0;
			double predictedLatency = 0.0;
			
			int requestType = input.readInt();
			String clientId = (String) input.readObject();
			double[] features = (double[]) input.readObject();
			double[] additionalStats = (double[]) input.readObject();
			String androidInfo = (String)input.readObject();
			DeviceInfo devInfo = new DeviceInfo(clientId, features, additionalStats, androidInfo);
			
			switch (requestType) {			
			
			case 0:		// receive the request to get batch size
				
				//Decides to evaluate the query or to abort
				if (nextCLient!=null) {
					if (clientId.equals(nextCLient))	action=1; 
					else	action=0;
				}
				else {
					if (alignedLogins) {
						if (profilerEpoch < loginEpoch.get(clientId))	action=0;
						else if (logoutRequest.get(clientId) > 0)	action = 1;
						else action = -1;
					}else action = 1;
				}
				
				if (action == 1) {						
					//computes the normalized features when normalized mode is chosen 
					double[] normFeatures = new double[features.length];
					if (normalize) {
						if (featureMinMax.isEmpty()) {
							for (Double f: features) {
								Double min = 0.99*f;
								Double max = 1.01*f;
								ArrayList<Double> entry = new ArrayList<Double>(Arrays.asList(min, max));
								featureMinMax.add(entry);
							} 
						}							
						for (int i=0; i<featureMinMax.size(); i++) {
							featureMinMax.get(i).set(0, Math.min(featureMinMax.get(i).get(0), features[i]));
							featureMinMax.get(i).set(1, Math.max(featureMinMax.get(i).get(1), features[i]));
							normFeatures[i] = minMaxScale(features[i], featureMinMax.get(i).get(0), featureMinMax.get(i).get(1));
							System.err.println("Actual: " + features[i] + ", min: "+ featureMinMax.get(i).get(0) +
									", max: " + featureMinMax.get(i).get(1) + ", normalized: " + normFeatures[i]);
						}
						
						devInfo.latencyFeatures = normFeatures;
					}
					
					//computes the batch size
					if (clientInfo.isEmpty())	batchSize = 1;
					else {
						learn(clientId);
						
						if (normalize) System.arraycopy(normFeatures, 0, features, 0, features.length);							
						long t = System.currentTimeMillis();
						predictedLatency = model.predict(features);
						System.out.println("PA Inference latency (ms): " + (System.currentTimeMillis() - t));
						batchSize = (int) ((SLO - devInfo.deviceLatency) / predictedLatency);
						if (batchSize < 1) batchSize = 1;	// at least 1 sample needs to be sent						
					}
					
					//store client info
					clientInfo.put(clientId, devInfo);
				}
									
				output.writeInt(action);					
				output.writeInt(batchSize);
				output.flush();
									
				break;
				
			case 1:		// receive the request to forward client info	
				
				synchronized (model) {
										
				if (nextCLient==null || clientId.equals(nextCLient)) {
					
					//store client info
					if (updateBefore==true)	devInfo.latencyFeatures = clientInfo.get(clientId).latencyFeatures;													
					clientInfo.put(clientId, devInfo);						
					clientFeatures.add(devInfo.latencyFeatures);
					clientTargets.add(devInfo.meanSizeLatency);
					
					if(!clientInfo.containsKey(clientId))	model.newLoginAtEpoch(profilerEpoch);
					
					//if there is no initial dataset the model is initialized with the features of the first client query
					if (!model.isUp() && filePath.equals("")) {
						double[][] initFeatures = new double[2][];
						double[] initTargets = new double[2];
						for (int i=0; i<initTargets.length; i++) {
							initFeatures[i] = devInfo.latencyFeatures;
							initTargets[i] = devInfo.meanSizeLatency;
						}
						model.initOnTheFly(initFeatures, initTargets);
					}
					
					System.out.println("PA " + model.getMode().toString());
					System.out.println("Complete Info:\n" + devInfo.toString(profilerEpoch, 1, true));
					
					//output file
		            // FIXME eclipse issues (no such file) -> works on the server
		            String outputCsvPath = "src/main/resources/output" + logFileName();
		            File file = new File(outputCsvPath);
		            boolean writeHeader = !file.exists();
		            FileWriter writer = new FileWriter(file, true);
		            writer.write(devInfo.toString(profilerEpoch, 1, writeHeader) + "\n");
		            writer.close();
					
					model.tick(profilerEpoch+1);
					nextCLient=null;
					
					if (alignedLogins) {
						int count = logoutRequest.containsKey(clientId) ? logoutRequest.get(clientId) : 0;
						logoutRequest.put(clientId, count-1);
					}						
					learn();
					profilerEpoch++;
				}
				
	            if (alignedLogins && loginEpochReverse.containsKey(profilerEpoch)) nextCLient = loginEpochReverse.get(profilerEpoch);					
				break;
				
				}
				
			default:
				System.out.println("Unknow request type!");
			}
			
		} catch (SocketException e) {
			// ???
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
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
