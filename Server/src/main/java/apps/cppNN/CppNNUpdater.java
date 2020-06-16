/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.cppNN;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.exception.MathArithmeticException;
import org.apache.commons.math3.geometry.Point;
import org.apache.commons.math3.geometry.Space;
import org.apache.commons.math3.geometry.Vector;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.CollectionSerializer;
import com.esotericsoftware.kryo.serializers.DefaultArraySerializers.ByteArraySerializer;
import com.esotericsoftware.kryo.serializers.MapSerializer;

import apps.SPSampler;
import coreComponents.SGDUpdater;
import coreComponents.Sampler;
import utils.ByteVec;
import utils.Helpers.*;
import utils.Helpers;
import utils.Kardam;
import utils.StalenessSimulator;

public class CppNNUpdater implements SGDUpdater {

	private Kryo kryoR, kryo1, kryo2, kryo3, kryo4, kryo5;

	ArrayList<Double> lrates;
	/**
	 * Collected (gradients, class distribution, epoch, clientID) since the last model update
	 */
	private ArrayList<Quadruple<byte[], int[], Integer, Integer>> acc;
	
	private StalenessSimulator<byte[]> staleSim;

	BufferedWriter bw;
	private long startTime;
	
	private int clientRequestsNum;
	private int clientRequestsSoFar;
	
	/**
	 * Number of required gradients for each model update (M-softsync)
	 */
	private int M;
	
	/**
	 * Number of local updates to run
	 */
	private int E;
	
	/**  
	 * Maximum size for {@link CppNNUpdater#models}
	 */
	private int staleSize;
	
	/**
	 * coldStartSize >= size
	 * if larger => the cold start phase for the staleness simulator will continue until the given number of updates
	 * useful for comparing the effect of different staleness distribution with fixed cold-start updates  
	 */
	private int coldStartSize;

	/**
	 * Staleness-aware dampening (0: average, 1: inverse, 2:exp)
	 */
	private int policy;
	
	/**
	 * Param alpha exponent used for exponential dampening
	 */
	private double alpha;

	/**
	 * noise std for DP learning; set to 0 to deactivate
	 */
	private double sigma;
	
	/**
	 * Gradient scaling; only used when sigma > 0
	 */
	private double C;
	private MapSerializer mapser;

	/**
	 * Learning model hashCode
	 */
	private int hashCode;
	
	/**
	 * Algorithm for BFT
	 */
	private Kardam<ByteVec, ByteVec> kardam;

	/**
	 * Last used gradients
	 * useful for feeding into the Kardam test
	 */
	private ByteVec lastGrad;
	
	/**
	 * Last model version
	 * useful for feeding into the Kardam test
	 */
	private ByteVec lastModel;
	
	/**
	 * percentile thresholds 
	 */
	private int batch_size_threshold;
	private int similarity_threshold;
	
	/**
	 * List of batch size computed so far
	 */
	private ArrayList<Integer> batchSizes;
	
	/**
	 * List of similarities received so far
	 */
	private ArrayList<Double> similarities;
	
	/**
	 * Global label vector at the server
	 * label vector: x[i] = number of examples for class i
	 */
	private int[] global_label_vector;
	
	private native byte[] getParametersNative(int priority);
	private native byte[] getModelParametersNative(int priority);
	private native void fetchParamsNative(byte[] inBuffer);
	private native void printParamsNative(byte[] inBuffer);
	private native void descentNative(byte[] gradients, int clientBatchSize, int staleSize);
	private native void initUpdater(double[] lrate, int E, double sigma, double C);
	private native int modelsSize();
	private native void setPriority(int priority);
	private native int getPriority();
	private native int getCurrEpoch();
	private native void setCurrEpoch(int currEpoch);
	private native double getLrate();
	private native byte[] getFlatGradient(byte[] grad);
	private native byte[] mergeFlatGradient(byte[] grad, byte[] flatGrad);
	private native boolean hasOutlier();
	private native int getNumLabels();



	@SuppressWarnings("unchecked")
	@Override
	public void initialize(InputStream input, Sampler sampler) {
		// Example: get learning rate for each epoch + Read message ("COPY
		// DAT!") from the Driver
		kryoR = new Kryo();
		kryo2 = new Kryo();
		kryo2.register(Integer.class);
		kryo1 = new Kryo();
		kryo1.register(byte[].class, new ByteArraySerializer());

		Input in = new Input(input);

		clientRequestsNum = kryo2.readObject(in, Integer.class);
		System.out.println("Number of client requests: " + clientRequestsNum);
		M = kryo2.readObject(in, Integer.class);
		System.out.println("M: " + M);
		E = kryo2.readObject(in, Integer.class);
		System.out.println("E: " + E);
		staleSize = kryoR.readObject(in, Integer.class);
		System.out.println("Staleness size: " + staleSize);
		policy = kryoR.readObject(in, Integer.class);
		System.out.println("Policy: " + policy);
		batch_size_threshold = kryoR.readObject(in, Integer.class);
		System.out.println("Batch size threshold: " + batch_size_threshold);
		similarity_threshold = kryoR.readObject(in, Integer.class);
		System.out.println("Similarity threshold: " + similarity_threshold);
		alpha = kryoR.readObject(in, Double.class);
		System.out.println("alpha: " + alpha);
		sigma = kryoR.readObject(in, Double.class);
		System.out.println("sigma: " + sigma);
		C = kryoR.readObject(in, Double.class);
		System.out.println("C: " + C);

		clientRequestsSoFar = 0;

		acc = new ArrayList<>();
		
		coldStartSize = staleSize; // default
		
		similarities = new ArrayList<>();
		batchSizes = new ArrayList<>();
			
		// load initial epoch
		int initEpoch = kryoR.readObject(in, Integer.class);
		System.out.println("Initial epoch: " + initEpoch);
		
		lrates = kryoR.readObject(in, ArrayList.class, new CollectionSerializer());
		System.out.println("Initial learning rate: " + lrates.get(0) + " " + lrates.get(1));

		byte[] nativeOutput = kryo1.readObject(in, byte[].class);

		hashCode = nativeOutput.hashCode();
		
		kardam = new Kardam<ByteVec, ByteVec>(10);
		lastGrad = null;
		lastModel = null;
		
		// System.out.println("Received: " + new String(nativeOutput,
		// StandardCharsets.UTF_8));
		fetchParamsNative(nativeOutput);
		
		double[] lrates_vec = new double[lrates.size()];
		for (int i = 0; i < lrates_vec.length; i++) {
			lrates_vec[i] = lrates.get(i);
			//System.out.println("Learning rate " + i + " is " + lrates_vec[i]);
		}
		initUpdater(lrates_vec, E, sigma, C);

		global_label_vector = new int[getNumLabels()]; // initialize to zeros

		setCurrEpoch(initEpoch);

		staleSim = new StalenessSimulator<>();
	
		
		String path = System.getProperty("user.home") + "/ServerOut";
		System.out.println("Logging to: " + path);
		FileOutputStream fos = null;
		try {
			fos = new FileOutputStream(path);
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		bw = new BufferedWriter(new OutputStreamWriter(fos));

		startTime = System.currentTimeMillis();
	}

	@Override
	public void getParameters(Output output, boolean isComputationRequest) {
		// lock for concurrent computation and evaluation requests
		synchronized (acc) {
			// System.out.println("Received: " + new String(msg,
			// StandardCharsets.UTF_8));
			int sendEpoch;
			
			byte[] model;
			if (isComputationRequest) {
			//	ByteVec modelParams = new ByteVec(getModelParametersNative(getPriority()));
			//	if (!kardam.checkModelVersion(kardam.currentWorkerId(), modelParams)) {
					sendEpoch = getCurrEpoch() - (modelsSize() - getPriority() - 1);
			//		System.out.println("Kardam: Pushing model: " + sendEpoch + " to client: " + kardam.currentWorkerId());
			//		kardam.setModel(kardam.currentWorkerId(), modelParams); // update simulation kardam info
					model = getParametersNative(getPriority());
			/*	}
				else {
					sendEpoch = getCurrEpoch() - (modelsSize() - (-1) - 1);
					System.out.println("Kardam: Pushing model prev: " + sendEpoch + " to client: " + kardam.currentWorkerId());
					modelParams = new ByteVec(getModelParametersNative(-1));
					kardam.setModel(kardam.currentWorkerId(), modelParams); // update simulation kardam info
					model = getParametersNative(getPriority()-1);
				}*/
			}
			else {
				model = getParametersNative(modelsSize()-1);
				sendEpoch = getCurrEpoch();
			}
			
			kryo2.writeObject(output, sendEpoch);
			kryo2.writeObject(output, hashCode);
			if (isComputationRequest) {
				kryo2.writeObject(output, kardam.currentWorkerId());
				kardam.nextWorkerId();
			}
			long trainTime = System.currentTimeMillis() - startTime;
			kryo2.register(Long.class);
			kryo2.writeObject(output, trainTime);
			kryo2.register(Integer.class);
			kryo1.writeObject(output, model);
		}
	}

	private double getDampen(int tau, double similarity) {
	    // staleness-aware learning
	    double l_tau = 1;
		
	    if (policy == 0) // average
	    	l_tau = 1;
	    else if (policy == 1) // inverse
	    	l_tau = 1 / (double) (tau + 1);
	    else if (policy == 2) {
	    	// inverse + class-aware dampening
	    	l_tau = 1 / (double) (tau + 1);
	    	if (hasOutlier() && tau > 1.5 * staleSize)
	    		l_tau /= Math.max(0.1, similarity);
	    }
	    else if (policy == 3) // exponential dampening
	    	l_tau = Math.exp((-1) * alpha * Math.min(tau, staleSize));
	    else if (policy == 4) {
	    	// exponential + class-aware dampening
	    	l_tau = Math.exp((-1) * alpha * Math.min(tau, staleSize));
	    	if (hasOutlier() && tau > 1.5 * staleSize)
	    		l_tau /= Math.max(0.1, similarity);
	    }
    	if (hasOutlier() && tau > 1.5 * staleSize)
    	    System.out.println("Outlier dampening: " + l_tau);

	    System.out.println("Dampening: " + l_tau);
	    return l_tau;
	}
	
	@Override
	public void update(InputStream input) {
		// TODO
		// lock for concurrent computation and evaluation requests
		synchronized (acc) {
			Input in = new Input(input);
			ArrayList<Quadruple<byte[], int[], Integer, Integer>> aggregated = null; // M aggregated gradients for updating

			int hashCode = kryoR.readObject(in, Integer.class);
			int id = kryoR.readObject(in, Integer.class);
			int epoch = kryoR.readObject(in, Integer.class);
			int clientBatchSize = kryoR.readObject(in, Integer.class);
			
			System.out.println("Received epoch: " + epoch);
			
			clientRequestsSoFar++;
			if (clientRequestsSoFar > clientRequestsNum) {
				System.out.println("Client request number reached. Exiting Server ...");
				System.exit(0);
			}

			if (hashCode != this.hashCode) {
				System.out.println("Gradients refer to different model. Dropping...");
				return;
			}

			byte[] g = kryoR.readObject(in, byte[].class);
			int[] local_label_vector = kryoR.readObject(in, int[].class);
			
			byte[] pickedG = null;
			int[] picked_local_label_vector = null;
			int pickedEpoch = -1;
			int pickedTau, pickedId;
			// check if prev gradient
	/*		if (kardam.checkGradVersion(id, epoch)) {
	
				// add gradient to prev
		//		acc.add(new Triple<>(kardam.getGrad(id).x.v, kardam.getGrad(id).y, id));

				System.out.println("Kardam: Pushing prev gradient for client: " + id);
				ByteVec tempGrad = new ByteVec(getFlatGradient(g));
				tempGrad = (ByteVec) tempGrad.scalarMultiply(getLrate() * getDampen(getCurrEpoch() - epoch));
				kardam.setGrad(id, tempGrad, epoch);
				kardam.updateLip(id);
				
				
			}
			else {*/
//				if (id == 2)
//				{
//					System.out.println("Dropping gradient from weak worker!");
//					return;
//				}
				
				acc.add(new Quadruple<>(g, local_label_vector, epoch, id));
		
		     	System.out.println("Read bytes: " + Helpers.humanReadableByteCount(in.total(), false));

		     	/* M-soft sync */
		     	if (acc.size() < M) {
		     		System.out.println("Gradients left to update: " + (M - acc.size()));
		     		return;
		     	}
		//	}
			
		     	if (staleSize > 0) {
		     	// 	simulate staleness
		     		int outlierClass = -1;
		     		if (hasOutlier())
		     			outlierClass = 0;
		     		Tuple<Integer, ArrayList<Quadruple<byte[], int[], Integer, Integer>>> temp = staleSim.stalenessSim(
		     				acc, getCurrEpoch(), staleSize-1, coldStartSize-1, modelsSize(), M, outlierClass, (staleSize-1) * 4);
		     		
		     		aggregated = temp.getSecond();
		     		//if (modelsSize() >= size) // if enough model versions => update priority; else update only if passes filter
		     		setPriority(temp.getFirst());
		     		if (aggregated == null) // not possible to update with the available gradients
		     			return;
		     	}
		     	else {
		     		aggregated = acc; // FIXME evaluation priority issues
		     		setPriority(0);
		     	}
		     	
		     	// update with M picked grads
		     	Quadruple<byte[], int[], Integer, Integer> picked;
		     	ByteVec avg = null;
	     		int avgSize = 0; // number of gradients that passed the filter and can be averaged (avgSize <= M)
	     		ByteVec currModel = new ByteVec(getModelParametersNative(modelsSize()-1));

	     		int[] window_label_vector = new int[getNumLabels()];
		     	for (int i=0; i<M; i++) {
		     		picked = aggregated.remove(0);
		     		
		     		pickedG = picked.getFirst();
		     		picked_local_label_vector  = picked.getSecond();
		     		pickedEpoch = picked.getThird();
		     		pickedId = picked.getFourth();
		     		pickedTau = getCurrEpoch() - pickedEpoch;
			
		     		String out = "\tresponse: clientRequestID|epoch|staleness|time:," + pickedId + "," + getCurrEpoch() + "," +  + pickedTau + 
		     				"," + (System.currentTimeMillis() - startTime) + "\n";
		     		System.out.print(out);
			
					System.out.println("Local label vector: " + Arrays.toString(picked_local_label_vector));
					System.out.println("Global label vector: " + Arrays.toString(global_label_vector));

					// batchSize-based pruning
				    int batchSize = 0;
				    for (int k=0; k<picked_local_label_vector.length; k++)
				    	batchSize += picked_local_label_vector[k];
				    System.out.println("Batch size: " + batchSize);
					batchSizes.add(batchSize);	
					int batch_thres = 0;
					if (batch_size_threshold > 0)
						batch_thres = (int) Helpers.percentile(Helpers.toDouble(batchSizes), batch_size_threshold);
					if (batchSize < batch_thres) {
						System.out.println("Dropping batch size: " + batchSize);
						return;
					}
				    
					// similarity-based pruning
				    double similarity = Helpers.similarity(picked_local_label_vector, global_label_vector);
				    System.out.println("Similarity: " + similarity);
				    similarities.add(similarity);
					double sim_thres = 0;
					if (similarity_threshold > 0)
						sim_thres = Helpers.percentile(similarities, similarity_threshold);
				    if (similarity < sim_thres) {
				    	System.out.println("Dropping similarity: " + similarity);
				    	return;
				    }
				    
		     		// staleness-aware dampening
		     		ByteVec pickedGrad = new ByteVec(getFlatGradient(pickedG));
		     		pickedGrad = (ByteVec) pickedGrad.scalarMultiply(getDampen(pickedTau, similarity));
		   
		     		// update window label vector
		     		for (int j=0; j<window_label_vector.length; j++)
		     			window_label_vector[j] += picked_local_label_vector[j];
		     		
		     		// update kardam info
		     		ByteVec pickedModel = null;
		     		if (modelsSize()-1 - pickedTau >= 0) {
		     			pickedModel = new ByteVec(getModelParametersNative(modelsSize()-1 - pickedTau));
		     			if ((modelsSize() == staleSize)) {
		     				System.out.println("Kardam: Pushing info for client: " + pickedId);
		     				kardam.setModel(pickedId, pickedModel);
		     				// multiply with learning rate for Kardam info
		     				if (kardam.setGrad(pickedId, (ByteVec) pickedGrad.scalarMultiply(getLrate()), pickedEpoch)) {
		     					kardam.updateLip(pickedId);
		     				}
		     			}
		     		}
			     	else
			     		System.out.println("Unable to fetch model for Kardam with priority: " + (modelsSize()-1 - pickedTau));
		     		
		     		// if not enough model versions yet (stale_size+1) || kardam check
//		    		if (kardam.checkByz(id, currGrad, lastGrad, currModel, lastModel)) {
                    if (true || (modelsSize() < staleSize) || kardam.checkByz(pickedId, pickedGrad, lastGrad, currModel, lastModel, pickedTau)) {
//		    			setPriority(temp.getFirst()); // update priority if gradient passes filter
		    			if (avg == null)
		    				avg = pickedGrad;
		    			else
		    				avg = (ByteVec) avg.add(pickedGrad);
		    			
		    			avgSize++;
		    		}
		    		else
		    			// TODO add the filtered gradient into a candidate list; after the prevPush check this gradient again
		    			System.out.println("Kardam: Filtered Byzantine gradient");
		     	}
		     	
		     	// update global label vector
	     		for (int j=0; j<window_label_vector.length; j++)
	     			global_label_vector[j] += window_label_vector[j];
		     	
		     	if (avg != null) {
		     		avg = (ByteVec) avg.scalarMultiply((double) 1/avgSize);
		     		pickedG = mergeFlatGradient(pickedG, avg.v);
	    			descentNative(pickedG, clientBatchSize, coldStartSize); 
	    			lastGrad = avg;
	    			lastModel = currModel;
		     	}	

			
			// flush accumulator in case simulation is not used 
			aggregated.clear();
		}
	}

}
