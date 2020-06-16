/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.cppNN;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.CollectionSerializer;
import com.esotericsoftware.kryo.serializers.DefaultArraySerializers.ByteArraySerializer;

import coreComponents.Model;
import utils.Helpers;
import utils.MyDataset;
import utils.Result;

/**
 * Dl4j based model used for Single Label classification (e.g. mnist, spambase)
 *
 */
public class CppNNModel implements Model {

	public int clientRequestsNum;
	public double base_lrate, alpha, sigma, C;
	public int M, E, size, policy, similarity_threshold, batch_size_threshold;
	protected Kryo kryo;
	private int epoch;
	private long trainTime;
	private String dataPath;
	private String saveLoadPath;
	private boolean loadState;
	private byte[] model;
	/**
	 * Epoch of last saved model
	 */
	private int savedEpoch;
	/**
	 * Learning model hashCode
	 */
	private int hashCode;

	private native void initializeNative(String dataPath);
	private native void evaluateNative();
	private native float accuracyNative();
	private native float errorNative();
	private native byte[] getParamsNative();
	private native void fetchParamsNative(byte[] inBuffer);
	private native void printParamsNative(byte[] inBuffer);
	
	/**
	 * @param clientRequestsNum number of anticipated updates to create a learning rate
	 * @param base_lrate initial learning rate value
	 * @param M number of gradients to aggregate before updating the model (M-softsync)
	 * @param E number of local (client) updates to run for each global update
	 * @param sigma noise std for DP learning; set to 0 to deactivate DP learning
	 * @param C gradient scaling; only used if sigma > 0
	 * @param similarity_threshold for pruning based on class-dist
	 * @param batch_size_threshold for pruning based on batch size
	 * @param size staleness range will be = [0, size - 1]; set to 0 to deactivate staleness simulation
	 * @param policy for staleness-aware dampening (0: average, 1: inverse, 2:exp)
	 * @param alpha exponent used for exponential dampening
	 * @param dataPath path to original dataset
	 * @param loadState indicates whether to load the checkpointed model state from the file
	 * @param saveLoadPath path to file for loading and saving the model checkpoints (if !null => saves model every 100 epochs)
	 */
	public CppNNModel(int clientRequestsNum, double base_lrate, int M, int E, double sigma, double C, int batch_size_threshold, int similarity_threshold, int size, int policy, double alpha, String dataPath, boolean loadState, String saveLoadPath) {
		kryo = new Kryo();
		this.clientRequestsNum = clientRequestsNum;
		this.base_lrate = base_lrate;
		this.M = M;
		this.E = E;
		this.sigma = sigma;
		this.C = C;
		this.similarity_threshold = similarity_threshold;
		this.batch_size_threshold = batch_size_threshold;
		this.size = size;
		this.policy = policy;
		this.alpha = alpha;
		
		this.dataPath = dataPath;
		this.loadState = loadState;
		this.saveLoadPath = saveLoadPath;
		initializeNative(dataPath);
		if (loadState)
			restoreState();
	}

	@Override
	public void initialize() {
	}

	
	public void train(MyDataset miniBatch) {
	}


	
	public byte[] getParams() {
		
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		Output output = new Output(outputStream);
		
		kryo.register(Integer.class);
		kryo.writeObject(output, clientRequestsNum);
		kryo.writeObject(output, M);
		kryo.writeObject(output, E);
		kryo.writeObject(output, size); 
		kryo.writeObject(output, policy);
		kryo.writeObject(output, batch_size_threshold);
		kryo.writeObject(output, similarity_threshold);
		
		kryo.register(Double.class);
		kryo.writeObject(output, alpha);
		kryo.writeObject(output, sigma);
		kryo.writeObject(output, C);

	
		kryo.register(Integer.class);
		// write initial epoch (0 or load epoch)
		kryo.writeObject(output, epoch);
		
		// write learning rate schedule 
		ArrayList<Double> lrates = new ArrayList<Double>();
		double lrate;
		System.out.println("Learning rate schedule: if (lrate > base_lrate / 100) lrate = base_lrate / Math.pow(1 + i, 0.3);");
		for (int i=0; i<clientRequestsNum; i++) {
			lrate = base_lrate / Math.pow(1 + i, 0.3);
			if (lrate > base_lrate / 100)
				lrates.add(lrate);
				//lrates.add(base_lrate);
			else {
				System.out.println("Total lrates: " + lrates.size());
				break;
			}
		}
		System.out.println("Total lrates: " + lrates.size());
		System.out.println("Last lrate: " + lrates.get(lrates.size() - 1));

		kryo.register(ArrayList.class, new CollectionSerializer());
		kryo.writeObject(output, lrates);
		
		// write initial parameters
		kryo.register(byte[].class, new ByteArraySerializer());
		byte[] msg = getParamsNative();
		printParamsNative(msg);
		kryo.writeObject(output, msg);
		
		output.close();
		return outputStream.toByteArray();
	}

	@Override
	public void fetchParams(InputStream input) {
		Input in = new Input(input);
		kryo.register(Integer.class);
		epoch = kryo.readObject(in, Integer.class);
        hashCode = kryo.readObject(in, Integer.class);
        trainTime = kryo.readObject(in, Long.class);
		
        kryo.register(byte[].class, new ByteArraySerializer());
        try {
        	model = kryo.readObject(in, byte[].class);
		}
        catch (Exception e) {
        	System.out.println(e.getMessage());
        	epoch = -1; // invalidate this evaluation round
        	return;
        }

		//System.out.println("Received: " + new String(nativeOutput, StandardCharsets.UTF_8));
		fetchParamsNative(model);
		System.out.println("Read Bytes: " + Helpers.humanReadableByteCount(in.total(), false));

		if ((saveLoadPath != null) && (epoch - savedEpoch > 100)) 
			saveState();
	}
	
	@Override
	public Result evaluate(MyDataset dataset) {
	
		evaluateNative();
		double error = errorNative();
		double accuracy = accuracyNative();
		
		return new Result(error, accuracy);
	}

	@Override
	public void printParams() {
	}

	@Override
	public int getcurrEpoch() {
		return epoch;
	}

	@Override
	public long getTrainTime() {
		return trainTime;
	}


	@Override
	public int[][] predict() {
		return null;
	}

	@Override
	public void saveState() {
		try {
			Output out = new Output(new FileOutputStream(saveLoadPath, false));
			kryo.register(Integer.class);
			kryo.writeObject(out, epoch);
			kryo.register(Integer.class);
			kryo.writeObject(out, trainTime);
			kryo.register(byte[].class, new ByteArraySerializer());
			kryo.writeObject(out, model);
			savedEpoch = epoch;
			System.out.println("Saved model at epoch: " + savedEpoch);
			out.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public void restoreState() {
		Input in;
		try {
			in = new Input(new FileInputStream(saveLoadPath));
			kryo.register(Integer.class);
			epoch = kryo.readObject(in, Integer.class);
			savedEpoch = epoch;
	        trainTime = kryo.readObject(in, Long.class);
	        kryo.register(byte[].class, new ByteArraySerializer());
			model = kryo.readObject(in, byte[].class);
			
			fetchParamsNative(model);
			System.out.println("Restored model at epoch: " + savedEpoch);
			in.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	@Override
	public void cleanUp() {
	}

}
