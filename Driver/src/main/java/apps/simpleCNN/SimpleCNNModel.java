/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.simpleCNN;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
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
public class SimpleCNNModel implements Model {

	public int iterations;
	public double base_lrate;
	public int M, size;
	protected Kryo kryo;
	private int epoch;
	private long trainTime;
	private int size_x, size_y, size_z; // input features size useful for the first layer size
	/**
	 * Learning model hashCode
	 */
	private int hashCode;

	private native void initializeNative(int featureSize_x, int featureSize_y, int featureSize_z);
	private native float accuracyNative();
	private native float errorNative();
	private native byte[] getParamsNative();
	private native void fetchParamsNative(byte[] inBuffer, int x, int y, int z);
	private native void printParamsNative(byte[] inBuffer);
	
	/**
	 * @param iterations number of iterations to create a learning rate
	 * @param base_lrate initial learning rate value
	 * @param M number of gradients to aggregate before updating the model (M-softsync)
	 * @param size staleness range will be = [0, size - 1]
	 * @param width input width
	 * @param height input height
	 * @param depth input depth
	 */
	public SimpleCNNModel(int iterations, double base_lrate, int M, int size, int width, int height, int depth) {
		kryo = new Kryo();
		this.iterations = iterations;
		this.base_lrate = base_lrate;
		this.M = M;
		this.size = size;
		this.size_x = width;
		this.size_y = height;
		this.size_z = depth;
		initializeNative(size_x, size_y, size_z);
	}

	@Override
	public void initialize() {
	}

	
	public void train(MyDataset miniBatch) {
	}


	
	public byte[] getParams() {
		
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		Output output = new Output(outputStream);
		
		// write input feature size
		kryo.register(Integer.class);
		kryo.writeObject(output, size_x);
		kryo.writeObject(output, size_y);
		kryo.writeObject(output, size_z);
		
		// write M-softasync param
		kryo.writeObject(output, M);
		
		// write staleness size
		kryo.writeObject(output, size);

		
		// write learning rate schedule 
		ArrayList<Double> lrates = new ArrayList<Double>();
		double lrate;
		//System.out.println("Learning rate schedule: if (lrate > base_lrate / 2) lrate = base_lrate / Math.pow(1 + i, 0.1);");
		for (int i=0; i<iterations; i++) {
			lrate = base_lrate / Math.pow(1 + i, 0.1);
			if (lrate > base_lrate / 2)
				//lrates.add(lrate);
				lrates.add(base_lrate);
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
		//printParamsNative(msg);
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
		byte[] nativeOutput = kryo.readObject(in, byte[].class);

		//System.out.println("Received: " + new String(nativeOutput, StandardCharsets.UTF_8));
		fetchParamsNative(nativeOutput, size_x, size_y, size_z);
		System.out.println("Read Bytes: " + Helpers.humanReadableByteCount(in.total(), false));

	}
	
	@Override
	public Result evaluate(MyDataset dataset) {
		
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
	}

	@Override
	public void restoreState() {
	}

	@Override
	public void cleanUp() {
	}

}
