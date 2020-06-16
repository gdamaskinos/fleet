/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils.dl4j;

import org.nd4j.linalg.dataset.DataSet;
import utils.MyDataset;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

public class MyDl4jDataset extends MyDataset {
	
	public DataSet dataset;

	/**
	 * Constructs dataset
	 * @param xset inputs
	 * @param tset labels
	 */
	public MyDl4jDataset(double[][] xset, double[][] tset) {
		NDArray Xset = new NDArray(xset);
		NDArray Tset = new NDArray(tset);
		dataset = new DataSet(Xset, Tset);
		numExamples = Xset.rows();
		featureSize = Xset.columns();
		numLabels = Tset.columns();
	}
	
	public MyDl4jDataset(DataSet dataset) {
		this.dataset = dataset;
		numExamples = dataset.getFeatureMatrix().rows();
		featureSize = dataset.getFeatureMatrix().columns();
		numLabels = dataset.getLabels().columns();
	}
}
