/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.dl4j;

import java.io.IOException;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import utils.MatrixOps;
import utils.MyDataset;
import utils.QuickSelect;
import utils.Result;
import utils.dl4j.MyDl4jDataset;

/**
 * Dl4j for Multi-Label classification (e.g. lastfm tagging)
 *
 */
public class Dl4jMultiLabelModel extends Dl4jModel {

	/**
	 * predicted values >= THRES will be considered as the output tags. 
	 * THRES regulates the precision VS recall trade-off.
	 */
	private double thres = -1;
	
	/**
	 * predicted values >= topN'th value will be considered as the output tags.
	 */
	private int N;
	
	/**
	 * holds the number of {@link Dl4jMultiLabelModel#evaluate(MyDataset) invokes}
	 */
	private int evalCount;
	
	/**
	 * 
	 * @param conf
	 * @param iterations
	 * @param base_lrate
	 * @param M M-softsync param
	 * @param size staleness range will be = [0, size - 1] 
	 * @param thres predicted values >= THRES will be considered as the output tags. THRES regulates the precision VS recall trade-off
	 */
	public Dl4jMultiLabelModel(MultiLayerConfiguration conf, int iterations, double base_lrate, int M, int size, double thres) {
		super(conf, iterations, base_lrate, M, size);
		this.thres = thres;
		evalCount = 0;
	}

	/**
	 * 
	 * @param conf
	 * @param iterations
	 * @param base_lrate
	 * @param M M-softsync param
	 * @param size staleness range will be = [0, size - 1] 
	 * @param N predicted values >= topN'th value will be considered as the output tags.
	 */
	public Dl4jMultiLabelModel(MultiLayerConfiguration conf, int iterations, double base_lrate, int M, int size, int N) {
		super(conf, iterations, base_lrate, M, size);
		this.N = N;
		evalCount = 0;
	}
	
	@Override
	public Result evaluate(MyDataset d) {
		NDArray setX, setT;		
		MyDl4jDataset dataset = (MyDl4jDataset) d;
		setX = (NDArray) dataset.dataset.getFeatures();
		setT = (NDArray) dataset.dataset.getLabels();
		
		INDArray output = net.output(setX);
		
		if (evalCount % 20 == 0 || evalCount == 0)
		try {
			MatrixOps.save2csv(output.data().asDouble(), output.columns(), System.getProperty("user.home") + "/data/data" + net.currEpoch + ".csv");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//System.out.println("Output: " + output.rows() + ", " + output.columns());
		//floatPrintArray(output);
		
		// get "recommended" i.e. round prediction values
		INDArray recom = new NDArray(output.rows(), output.columns());
		double top = -1;
		for (int i=0; i< output.rows(); i++) {
			// get top4 value
			if (thres == -1)
				top = QuickSelect.select(output.getRow(i).data().asDouble(), N);
			//System.out.println("Top4: " + top);
			for (int j=0; j<output.columns(); j++)
			if (thres != -1)
				recom.putScalar(i, j, FastMath.round(output.getDouble(i, j) + 0.5f - thres));
			else {
				if (output.getDouble(i, j) >= top)
					recom.putScalar(i, j, 1);
				else 
					recom.putScalar(i, j, 0);
			}
		}
		//System.out.println("Total recommendations: " + recom.sumNumber());
		
		// get perTag recommended
		INDArray perTag_recom = recom.sum(0);
		double total_recom = recom.sumNumber().doubleValue();
		System.out.println("Total recommended : " + total_recom);
		intPrintArray(perTag_recom);
		
		// get the "hits"
		INDArray hits = recom.mul(setT);	
		INDArray perTag_hits = hits.sum(0);
		double total_hits = hits.sumNumber().doubleValue();
		System.out.println("Total hits: " + total_hits);
		intPrintArray(perTag_hits);
		
		// get the "relevant"
		INDArray perTag_relevant = setT.sum(0);
		double total_relevant = setT.sumNumber().doubleValue();
		System.out.println("Total relevant : " + total_relevant);
		intPrintArray(perTag_relevant);
		//System.out.println(perTag_relevant.sumNumber());

		INDArray perTag_precision = perTag_hits.div(perTag_recom); //.meanNumber().doubleValue();
		double val, precision = 0;
		int count = 0;
		for (int i=0; i<perTag_precision.length(); i++) {
			val = perTag_precision.getDouble(i);
			if (!Double.isNaN(val)) {
				count++;
				precision += val;
			}	
		}
		//System.out.println("Precision count: " + count);
		precision /= count;
			
		INDArray perTag_recall = perTag_hits.div(perTag_relevant); //.meanNumber().doubleValue();
		double recall = 0;
		count = 0;
		for (int i=0; i<perTag_recall.length(); i++) {
			val = perTag_recall.getDouble(i);
			if (!Double.isNaN(val)) {
				count++;
				recall += val;
			}	
		}
		//System.out.println("Recall count: " + count);
		recall /= count;		
		
		System.out.println("Precision: " + precision);
		System.out.println("Recall: " + recall);

		// f1 score
		double accuracy = 2 * precision * recall / (precision + recall); 
		
		// calculate error
//		double error = LossFunctions.score(setT, LossFunction.NEGATIVELOGLIKELIHOOD, output, 0, 0, false) / setT.rows();
		
//		INDArray temp = Transforms.log(output);
//		NDArray ones = Nd4j.ones(shape);
		
//		double error = CustomCrossEntropy.computeScore(setT, output, true); // !affect output object
		
		double error = net.score(dataset.dataset);

		evalCount ++;
		return new Result(error, accuracy);
	}
	
	
	private void intPrintArray(INDArray x) {
		for (int i=0; i<x.rows(); i++) {
			for (int j=0; j<x.columns(); j++)
				System.out.print(x.getInt(i, j) + "\t");
			System.out.println();
		}
	}
	
	private void floatPrintArray(INDArray x) {
		for (int i=0; i<x.rows(); i++) {
			for (int j=0; j<x.columns(); j++)
				System.out.print(x.getFloat(i, j) + "\t");
			System.out.println();
		}
	}
	
}
