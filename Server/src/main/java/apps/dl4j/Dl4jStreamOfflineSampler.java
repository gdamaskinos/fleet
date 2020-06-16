/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.dl4j;


import java.io.IOException;
import java.util.ArrayList;

import org.jblas.DoubleMatrix;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Output;

import coreComponents.Sampler;
import utils.MyDataset;
import utils.Parser;

/**
 * Streams csv input instead of loading it to memory.
 * Usefull for big datasets and small available memory.
 *
 */
public class Dl4jStreamOfflineSampler implements Sampler {

	private Kryo kryo;
	private Parser xparser;
	private Parser tparser;

	private int featureSize;
	private int numLabels;
	
	private int trainSize;
	
	/**
	 * mini-batch pool to implement "cyclic" rule
	 */
	private ArrayList<Integer> batchPool;
	
	/**
	 * Offline Sampler constructor
	 * @param prefix (e.g. /path/to/datasets/spambase_)
	 */
	public Dl4jStreamOfflineSampler(String prefix){
		
		kryo = new Kryo();
		kryo.register(MyDataset.class);
		
		String trainXPath = "training_features.csv";
		String trainTPath = "training_labels.csv";
	
		try {
			xparser = new Parser(prefix + trainXPath);
			tparser = new Parser(prefix + trainTPath);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		featureSize = xparser.cols;
		numLabels = tparser.cols;
		trainSize = xparser.rows;
	
		System.out.println("Training examples: " + trainSize);
		System.out.println("Input latencyFeatures: " + featureSize);
		System.out.println("Labels: " + numLabels);
		
		batchPool = new ArrayList<Integer>();
		for (int i=0; i<trainSize; i++)
			batchPool.add(i);
	}

	public void reset() {
		// refresh data pool 
		batchPool.clear();
			
		for (int i=0; i<trainSize; i++)
				batchPool.add(i);
		
	}
	
	private MyDataset getSample(int size) {
		Double temp;
		int k;

		if (size > trainSize) {
			System.out.println("MiniBatch size too large. Reducing to maximum possible: " + trainSize);
			size = trainSize;
		}
		
		// determine i
		DoubleMatrix miniTrainX = new DoubleMatrix(size, this.featureSize);
		DoubleMatrix miniTrainT = new DoubleMatrix(size, this.numLabels);
		
		// refresh data pool if needed
		if (size > batchPool.size()) {
			batchPool.clear();
			
			for (int i=0; i<trainSize; i++)
				batchPool.add(i);
		}
		
		for (int j = 0; j < size; j++) {
			System.out.println("NEXT");
			// k ~ U[0, poolSize)
			temp = Math.random() * batchPool.size();
			k = temp.intValue();
			try {
				miniTrainX.putRow(j, xparser.getSingleRow(batchPool.get(k)));
				miniTrainT.putRow(j, tparser.getSingleRow(batchPool.get(k)));
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			// delete from pool
			batchPool.remove(k);
		}
		
		return new MyDataset(miniTrainX, miniTrainT);
	}
	
	public void getSample(int size, Output output) {
//		MyDataset miniBatch = getSample(size);
//		NDArray outX = new NDArray(miniBatch.Xset.toArray2());
//		NDArray outT = new NDArray(miniBatch.Tset.toArray2());
//		DataSet outminiBatch = new DataSet(outX, outT);
//		System.out.println(outminiBatch.numExamples());
//		
//		System.out.println(outX.rows() + " " + outX.columns());
//		//System.out.println(miniBatch.Tset.columns);
//		//kryo.writeObject(output, outminiBatch);
//		kryo.writeObject(output, miniBatch.Xset);
//		kryo.writeObject(output, miniBatch.Tset);
		
		MyDataset miniBatch = getSample(size);
		kryo.writeObject(output, miniBatch);

	}


}
