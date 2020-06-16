/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.mlp;

import java.io.InputStream;

import org.jblas.DoubleMatrix;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import coreComponents.SGDUpdater;
import coreComponents.Sampler;

/**
 * Updater class implemented by the Service Provider
 * 
 */
public class MLPUpdater implements SGDUpdater {

	private Kryo kryo;
	private Kryo kryo2;
	private double momentum;
	private double numerator;
	private double pow;

	private int featureSize;
	private int hiddenNum;

	/**
	 * layer weights
	 */
	private DoubleMatrix w1, w2;
	/**
	 * bias parameters
	 */
	private DoubleMatrix b1, b2;
	/**
	 * delta parameters
	 */
	DoubleMatrix delta_w1, delta_w2, delta_b1, delta_b2;

	private long startTime;
	private int currEpoch;

	public void update(InputStream input) {

		Input in = new Input(input);
		MLPGradients gradients = kryo2.readObject(in, MLPGradients.class);

		double nu;

		nu = this.numerator / Math.pow(currEpoch + 1, this.pow);

		delta_w1 = gradients.delta_w1.mmul(-nu * (1 - this.momentum)).add(delta_w1.mmul(this.momentum));
		delta_b1 = gradients.delta_b1.mmul(-nu * (1 - this.momentum)).add(delta_b1.mmul(this.momentum));
		delta_w2 = gradients.delta_w2.mmul(-nu * (1 - this.momentum)).add(delta_w2.mmul(this.momentum));
		delta_b2 = gradients.delta_b2.mmul(-nu * (1 - this.momentum)).add(delta_b2.mmul(this.momentum));

		// lock for concurrent computation and evaluation requests
		synchronized (w1) {
			w1 = w1.add(delta_w1);
			b1 = b1.add(delta_b1);
			w2 = w2.add(delta_w2);
			b2 = b2.add(delta_b2);

			currEpoch++;
		}
	}

	public void getParameters(Output output, boolean isComputationRequest) {
		// lock for concurrent computation and evaluation requests
		synchronized (w1) {
			MLPModelParams params = new MLPModelParams(momentum, numerator, pow, -1, featureSize, hiddenNum, w1, w2, b1,
					b2, currEpoch, System.currentTimeMillis() - startTime);
			kryo.writeObject(output, params);
		}
	}

	public void initialize(InputStream input, Sampler sampler) {
		// create serialized model
		kryo = new Kryo();
		kryo.register(MLPModelParams.class);
		kryo2 = new Kryo();
		kryo2.register(MLPGradients.class);

		// Load the model
		Input in = new Input(input);
		MLPModelParams restored = kryo.readObject(in, MLPModelParams.class);
		w1 = restored.w1;
		w2 = restored.w2;
		b1 = restored.b1;
		b2 = restored.b2;
		momentum = restored.momentum;
		numerator = restored.numerator;
		pow = restored.pow;
		featureSize = restored.featureSize;
		hiddenNum = restored.hiddenNum;

		currEpoch = 0;
		startTime = System.currentTimeMillis();
		// System.out.println("Received w1: " + w1);

		hiddenNum = w1.getRows() / 2;
		featureSize = w1.getColumns();

		delta_w1 = DoubleMatrix.zeros(2 * hiddenNum, featureSize);
		delta_w2 = DoubleMatrix.zeros(hiddenNum, 1);

		delta_b1 = DoubleMatrix.zeros(2 * hiddenNum, 1);
		delta_b2 = DoubleMatrix.zeros(1, 1);
	}

}
