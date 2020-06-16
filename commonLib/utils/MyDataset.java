/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import org.jblas.DoubleMatrix;

/**
 * Dataset consists of an X input set and a target value set.
 *
 */
public class MyDataset {

	public MyDataset() {
		
	}

	/**
	 * Constructs dataset
	 * @param xset inputs
	 * @param tset labels
	 */
	public MyDataset(DoubleMatrix xset, DoubleMatrix tset) {
		this.Xset = xset;
		this.Tset = tset;
		numExamples = xset.rows;
		featureSize = xset.columns;
		numLabels = tset.columns;
	}

	/**
	 * Constructs dataset
	 * @param xset inputs
	 * @param tset labels
	 */
	public MyDataset(double[][] xset, double[][] tset) {
		this.Xset = new DoubleMatrix(xset);
		this.Tset = new DoubleMatrix(tset);
		numExamples = Xset.rows;
		featureSize = Xset.columns;
		numLabels = Tset.columns;
	}
	
	public int numExamples() {
		return numExamples;
	}
	
	public int featureSize() {
		return featureSize;
	}
	
	public int numLabels() {
		return numLabels;
	}
	
	protected int numExamples, featureSize, numLabels;
	public DoubleMatrix Xset, Tset;
}
