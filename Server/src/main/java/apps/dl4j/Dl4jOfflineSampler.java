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

import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Output;

import coreComponents.Sampler;
import utils.Parser;
import utils.dl4j.MyDl4jDataset;
import utils.dl4j.Nd4jSerializer;

public class Dl4jOfflineSampler implements Sampler {

	private Kryo kryo;
	private MyDl4jDataset dataSet;
	private int featureSize;
	private int numLabels;
	
	private int trainSize;
	private double weight;
	
	/**
	 * mini-batch pool to implement "cyclic" rule
	 */
	private ArrayList<Integer> batchPool;
	
	/**
	 * Offline 	Sampler constructor
	 * @param prefix (e.g. /path/to/datasets/spambase_)
	 */
	public Dl4jOfflineSampler(String prefix){
		
		kryo = new Kryo();
        kryo.register(NDArray.class, new Nd4jSerializer());
        
		String trainXPath = "training_features.csv";
		String trainTPath = "training_labels.csv";
	
		try {
			dataSet = new MyDl4jDataset(new Parser(prefix + trainXPath).getValues(), new Parser(prefix + trainTPath).getValues());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		featureSize = dataSet.featureSize();
		numLabels = dataSet.numLabels();
		trainSize = dataSet.numExamples();
		
		batchPool = new ArrayList<Integer>();
		for (int i=0; i<trainSize; i++)
			batchPool.add(i);
		
//		weight = 1;
	}

	public void reset() {
		// refresh data pool 
		batchPool.clear();
			
		for (int i=0; i<trainSize; i++)
				batchPool.add(i);
		
	}
	
	public MyDl4jDataset getSample(int size) {
		Double temp;
		int k;

		if (size > trainSize) {
			System.out.println("MiniBatch size too large. Reducing to maximum possible: " + trainSize);
			size = trainSize;
		}
		
		// determine i
		NDArray miniTrainX = new NDArray(size, featureSize);
		NDArray miniTrainT = new NDArray(size, numLabels);
		
		// refresh data pool if needed
		if (size > batchPool.size()) {
			batchPool.clear();
			
			for (int i=0; i<trainSize; i++)
				batchPool.add(i);
		}
		// use the following only for debugging (fixed mini-batch)
//		   int[] arr = { 394,1072,949,1292,843,210,1240,723,742,158,1106,618,964,1434,900,1253,1349,490,1395,162,1181,1169,102,1140,1248,980,228,529,1028,234,730,1250,1171,746,58,850,1474,783,1057,372,156,1307,111,422,234,253,973,19,1361,166,1270,610,988,943,1047,591,201,718,1199,151,69,1072,1465,953,313,767,184,1374,40,1048,1224,396,1217,322,1025,229,1002,556,1399,792,920,730,191,882,280,1455,95,1165,91,542,843,13,84,327,150,1231,806,1435,685,347};	
			for (int j = 0; j < size; j++) {			
			// k ~ U[0, poolSize)
			temp = Math.random() * batchPool.size();
			k = temp.intValue();
//				k = arr[j];

			miniTrainX.putRow(j, dataSet.dataset.getFeatureMatrix().getRow(batchPool.get(k)));
			miniTrainT.putRow(j, dataSet.dataset.getLabels().getRow(batchPool.get(k)));
			
			// delete from pool
//			batchPool.remove(k);
		}
		
		return new MyDl4jDataset(new DataSet(miniTrainX, miniTrainT));
	}
	
	public void getSample(int size, Output output) {
		DataSet miniBatch = getSample(size).dataset;
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
//		kryo.register(NDArray.class, new Nd4jSerializer());
//		kryo.writeObject(output, outX);
//		kryo.writeObject(output, outT);
		

		kryo.writeObject(output, miniBatch.getFeatureMatrix());
		kryo.writeObject(output, miniBatch.getLabels());
		
	}

}
