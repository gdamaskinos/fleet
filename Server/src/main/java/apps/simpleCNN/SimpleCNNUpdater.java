/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.simpleCNN;

import java.io.BufferedWriter;
import java.io.InputStream;
import java.util.ArrayList;

import org.apache.commons.math3.util.Pair;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.CollectionSerializer;
import com.esotericsoftware.kryo.serializers.DefaultArraySerializers.ByteArraySerializer;
import com.esotericsoftware.kryo.serializers.MapSerializer;

import apps.dl4j.Dl4jUpdater;
import coreComponents.SGDUpdater;
import coreComponents.Sampler;
import utils.Helpers;

public class SimpleCNNUpdater implements SGDUpdater {

	private Kryo kryoR, kryo1, kryo2, kryo3, kryo4, kryo5;

	ArrayList<Double> lrates;
	/**
	 * Collected (gradients, epoch) since the last model update
	 */
	private ArrayList<Pair<byte[], Integer>> acc;

	BufferedWriter bw;
	private long startTime;
	private int epoch;
	/**
	 * Number of required gradients for each model update (M-softsync)
	 */
	private int M;
	/**  
	 * Maximum size for {@link Dl4jUpdater#models}
	 */
	private int size;
	private MapSerializer mapser;
	/**
	 * input features size useful for the first layer size
	 */
	private int size_x, size_y, size_z; 

	/**
	 * Learning model hashCode
	 */
	private int hashCode;

	private native byte[] getParametersNative();

	private native void fetchParamsNative(byte[] inBuffer, int x, int y, int z);

	private native void printParamsNative(byte[] inBuffer);

	private native void initUpdater(double lrate, double momentum);

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

		size_x = kryo2.readObject(in, Integer.class);
		size_y = kryo2.readObject(in, Integer.class);
		size_z = kryo2.readObject(in, Integer.class);
		System.out.println("Feature size: " + size_x + " " + size_y + " " + size_z);
		
		M = kryo2.readObject(in, Integer.class);
		System.out.println("M: " + M);

		// load staleness size for simulator
		size = kryoR.readObject(in, Integer.class);
		acc = new ArrayList<>();
		
		System.out.println("Staleness size: " + size);
		
		lrates = kryoR.readObject(in, ArrayList.class, new CollectionSerializer());
		System.out.println("Initial learning rate: " + lrates.get(0) + " " + lrates.get(1));

		byte[] nativeOutput = kryo1.readObject(in, byte[].class);

		hashCode = nativeOutput.hashCode();

		// System.out.println("Received: " + new String(nativeOutput,
		// StandardCharsets.UTF_8));
		fetchParamsNative(nativeOutput, size_x, size_y, size_z);

		initUpdater(lrates.get(0), 0.9);

		epoch = 0;
		startTime = System.currentTimeMillis();
	}

	@Override
	public void getParameters(Output output, boolean isComputationRequest) {
		// lock for concurrent computation and evaluation requests
		synchronized (acc) {
			// System.out.println("Received: " + new String(msg,
			// StandardCharsets.UTF_8));
			kryo2.writeObject(output, epoch++);
			kryo2.writeObject(output, hashCode);
			long trainTime = System.currentTimeMillis() - startTime;
			kryo2.register(Long.class);
			kryo2.writeObject(output, trainTime);
			kryo2.register(Integer.class);

			kryo1.writeObject(output, getParametersNative());
		}
	}

	@Override
	public void update(InputStream input) {
		// TODO
		// lock for concurrent computation and evaluation requests
		synchronized (acc) {
			Input in = new Input(input);

			int hashCode = kryoR.readObject(in, Integer.class);
			int epoch = kryoR.readObject(in, Integer.class);
			System.out.println("Received epoch: " + epoch);

			if (hashCode != this.hashCode) {
				System.out.println("Gradients refer to different model. Dropping...");
				return;
			}

			byte[] g = kryoR.readObject(in, byte[].class);
			acc.add(new Pair<>(g, epoch));

			System.out.println("Read bytes: " + Helpers.humanReadableByteCount(in.total(), false));

			printParamsNative(g);
			
			/* M-soft sync */
			if (acc.size() < M) {
				System.out.println("Gradients left to update: " + (M - acc.size()));
				return;
			}
		}
	}

}
