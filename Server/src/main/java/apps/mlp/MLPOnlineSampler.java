/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.mlp;

import java.io.IOException;
import java.util.ArrayList;

import org.jblas.DoubleMatrix;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Output;

import coreComponents.Sampler;
import utils.MyDataset;
import utils.Parser;

public class MLPOnlineSampler implements Sampler {

	private Kryo kryo;
	private MyDataset dataSet;
	private int featureSize;
	private int trainSize;
	
	/**
	 * mini-batch pool to implement "cyclic" rule
	 */
	private ArrayList<Integer> batchPool;
	private ArrayList<Integer> onlinePool;
	private ArrayList<Integer> onlineClone;
	
	
	@SuppressWarnings("unchecked")
	public MLPOnlineSampler(String prefix){
		kryo = new Kryo();
		kryo.register(MyDataset.class);

		String trainXPath = "training_features.csv";
		String trainTPath = "training_labels.csv";
		
		try {
			dataSet = new MyDataset(new Parser(prefix + trainXPath).getValues(), new Parser(prefix + trainTPath).getValues());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		featureSize = dataSet.featureSize();
		trainSize = dataSet.numExamples();
		
		batchPool = new ArrayList<Integer>();
		for (int i=0; i<trainSize; i++)
			batchPool.add(i);
		
		/*
		 * initialize the samples pool for simulating online learning
		 */
		onlinePool = new ArrayList<Integer> ();
		int onlineSize = (int)(0.2*trainSize);
		
		for(int i=0; i<onlineSize; i++){
			int index = (int)(Math.random() * batchPool.size());
			onlinePool.add(batchPool.get(index));
			batchPool.remove(index);
		}
		
		onlineClone = (ArrayList<Integer>) onlinePool.clone();
		
	}

	public void reset() {
		// refresh data pool 
		batchPool.clear();
			
		for (int i=0; i<trainSize; i++)
				batchPool.add(i);
		
	}
	
	@SuppressWarnings("unchecked")
	public MyDataset getSample(int size) {
		Double temp;
		int k;

		if (size > trainSize) {
			System.out.println("MiniBatch size too large. Reducing to maximum possible: " + trainSize);
			size = trainSize;
		}
		
		// determine i
		DoubleMatrix miniTrainX = new DoubleMatrix(size, this.featureSize);
		DoubleMatrix miniTrainT = new DoubleMatrix(size, 1);
		
		// refresh data pool if needed
		if (size > onlineClone.size()) {
			onlineClone = (ArrayList<Integer>) onlinePool.clone();
		}
		
		for (int j = 0; j < size; j++) {
			
			// k ~ U[0, poolSize)
			temp = Math.random() * onlineClone.size();
			k = temp.intValue();
			miniTrainX.putRow(j, this.dataSet.Xset.getRow(onlineClone.get(k)));
			miniTrainT.put(j, oneHot2Value(this.dataSet.Tset.getRow(batchPool.get(k))));
			
			// delete from pool
			onlineClone.remove(k);
			
		}
		
		return new MyDataset(miniTrainX, miniTrainT);
	}

	public void getSample(int size, Output output) {
		MyDataset miniBatch = getSample(size);
		kryo.writeObject(output, miniBatch);
	}
	public MyDataset getTrainSet() {
		return this.dataSet;
	}

	public void updateOnlinePool() {
		if(batchPool.size()==0)
			return;
		
		int onlineSize = (int)(0.2*trainSize);
		
		for(int i=0; i<onlineSize; i++){
			
			if(batchPool.size()==0)
				return;
			
			int index = (int)(Math.random() * batchPool.size());
			
			onlinePool.add(batchPool.get(index));
			onlineClone.add(batchPool.get(index));
			
			batchPool.remove(index);
		}
		
	}
	
	/**
	 * Transforms one-hot vector to value corresponding to the labeled class
	 */
	private double oneHot2Value(DoubleMatrix v) {
		for (int i=0; i<v.length; i++) 
			if (v.get(i) == 1) {
				if (v.length == 2) // binary classification
					if (i==0)
						return -1;
				return i;
			}
		
		return -10;
	}

}
