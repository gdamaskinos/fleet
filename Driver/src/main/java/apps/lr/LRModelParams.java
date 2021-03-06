/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.lr;

import org.jblas.DoubleMatrix;

public class LRModelParams {

	public LRModelParams() {

	}

	public LRModelParams(double rate, int featuresize, int numlabels, DoubleMatrix weights, DoubleMatrix biases, int currEpoch, long trainTime) {
		this.rate = rate;
		this.featuresize = featuresize;
		this.numlabels = numlabels;
		this.weights = weights;
		this.biases = biases;
		this.currEpoch = currEpoch;
		this.trainTime = trainTime;
	}

	public LRModelParams(double rate, int featuresize, int numlabels, DoubleMatrix weights, DoubleMatrix biases) {
		this.rate = rate;
		this.featuresize = featuresize;
		this.numlabels = numlabels;
		this.weights = weights;
		this.biases = biases;
	}

	
    public LRModelParams clone() {
    	LRModelParams cloneModel = new LRModelParams(this.rate,this.featuresize, this.numlabels, this.weights,this.biases,this.currEpoch,this.trainTime);
    	return cloneModel;
    }

	public void update(DoubleMatrix weights, DoubleMatrix biases){
		this.weights = weights;
		this.biases = biases;
	}

	public double rate;

	public int featuresize;
	public int numlabels;
	/**
	 * layer weights, biases
	 */
	public DoubleMatrix weights;
	public DoubleMatrix biases;
	/**
	 * number of outer iterations of the model (how many times the model was
	 * trained on the whole trainingSet).
	 */
	public int currEpoch;
	/**
	 * total training time
	 */
	public long trainTime;

}
