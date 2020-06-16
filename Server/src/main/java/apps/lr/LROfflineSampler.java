/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.lr;

import java.io.IOException;
import java.util.ArrayList;

import org.jblas.DoubleMatrix;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Output;

import coreComponents.Sampler;
import utils.MyDataset;
import utils.Parser;


public class LROfflineSampler implements Sampler {

	private Kryo kryo;
	private MyDataset dataSet;
	private int featureSize;
	private int numlabels;
	
	private int trainSize;
	
	/**
	 * mini-batch pool to implement "cyclic" rule
	 */
	private ArrayList<Integer> batchPool;
	
	/**
	 * Offline 	Sampler constructor
	 * @param prefix (e.g. /path/to/datasets/spambase_)
	 */
	public LROfflineSampler(String prefix){
		
		kryo = new Kryo();
		kryo.register(MyDataset.class);
    //    kryo.register(Nd4j.getBackend().getNDArrayClass(), new Nd4jSerializer());
        
		String trainXPath = "training_features.csv";
		String trainTPath = "training_labels.csv";
	
		try {
			dataSet = new MyDataset(new Parser(prefix + trainXPath).getValueMatrix(), new Parser(prefix + trainTPath).getValueMatrix());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		featureSize = dataSet.featureSize();
		numlabels = dataSet.numLabels();
		trainSize = dataSet.numExamples();
		
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
	
	public MyDataset getSample(int size) {
		Double temp;
		int k;

		if (size > trainSize) {
			System.out.println("MiniBatch size too large. Reducing to maximum possible: " + trainSize);
			size = trainSize;
		}
		
		// determine i
		DoubleMatrix miniTrainX = new DoubleMatrix(size, this.featureSize);
		DoubleMatrix miniTrainT = new DoubleMatrix(size, this.numlabels);
		
		// refresh data pool if needed
		if (size > batchPool.size()) {
			batchPool.clear();
			
			for (int i=0; i<trainSize; i++)
				batchPool.add(i);
		}
		
		for (int j = 0; j < size; j++) {
			
			// k ~ U[0, poolSize)
			temp = Math.random() * batchPool.size();
			k = temp.intValue();
			miniTrainX.putRow(j, this.dataSet.Xset.getRow(batchPool.get(k)));
			miniTrainT.putRow(j, this.dataSet.Tset.getRow(batchPool.get(k)));
			
			// delete from pool
			batchPool.remove(k);
		}
		
		return new MyDataset(miniTrainX, miniTrainT);
	}

	
	public void getSample(int size, Output output) {
		MyDataset m = getSample(size);
		//System.out.println(miniBatch.Tset.columns);
		
//		NDArray outX = new NDArray(m.Xset.toArray2());
//		NDArray outT = new NDArray(m.Tset.toArray2());
//		
//		kryo.writeObject(output, outX);
//		kryo.writeObject(output, outT);
		
		kryo.writeObject(output, m);
	}

	public MyDataset getTrainSet() {
		return this.dataSet;
	}


}
