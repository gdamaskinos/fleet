/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.mlp;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;

import org.jblas.DoubleMatrix;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import coreComponents.Model;
import utils.MyDataset;
import utils.MatrixOps;
import utils.Result;

public class MLPModel implements Model {

	private Kryo kryo;
	private double momentum;
	private double numerator;
	private double pow;
	private double sigma;
	
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
	 * number of outer iterations of the model (how many times the model was trained on the whole trainingSet).
	 */
	int currEpoch;
	/**
	 * total training time
	 */
	long trainTime;
	
	public MLPModel(double momentum, double numerator, double pow, 
			double sigma, int featureSize, int hiddenNum){
		kryo = new Kryo();
		kryo.register(MLPModelParams.class);
		
		this.momentum = momentum;
		this.numerator = numerator;
		this.pow = pow;
		this.sigma = sigma;
		
		this.featureSize = featureSize;
		this.hiddenNum = hiddenNum;
		
		this.initialize();
		
	}

	public byte[] getParams() {
		//System.out.println("Sending w1: " + w1);
		MLPModelParams params = new MLPModelParams(momentum, numerator, pow, sigma, featureSize, hiddenNum, w1, w2, b1, b2, currEpoch, trainTime);
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		Output output = new Output(outputStream);

		kryo.writeObject(output, params);
		output.close();

		return outputStream.toByteArray();
	}
	
	
	public void initialize() {
		double s = this.sigma / Math.sqrt(this.featureSize);

		w1 = DoubleMatrix.randn(2 * this.hiddenNum, this.featureSize).mmul(s);
		b1 = DoubleMatrix.randn(2 * this.hiddenNum, 1).mmul(s);

		w2 = DoubleMatrix.randn(this.hiddenNum, 1).mmul(s);
		b2 = DoubleMatrix.randn(1, 1).mmul(s);

	}
	
	public Result evaluate(MyDataset dataset) {
		
		DoubleMatrix a1, a2, ae, ao, z1, a1t, results, setX, setT;
		int setN;

		setX = dataset.Xset;
		setT = new DoubleMatrix(dataset.Tset.rows, 1);
		// for each example cast one-hot vector to value
		for (int i=0; i<dataset.Tset.rows; i++)
			for (int j=0; j<dataset.Tset.columns; j++) {
				if (dataset.Tset.get(i, j) == 1) {// label-j
					if (dataset.Tset.columns == 2) // binary classification
						if (j == 0)
							setT.put(i, -1);
						else
							setT.put(i, j);
					break;
				}
			}
		setN = setX.rows;
		
		a1 = w1.mmul(setX.transpose()).add(b1.repmat(1, setN));
		a1t = a1.transpose();
		ae = new DoubleMatrix(setN, this.hiddenNum);
		ao = new DoubleMatrix(setN, this.hiddenNum);
		for (int row = 0; row < setN; row++) {
			ae.putRow(row, MatrixOps.getRowRange(a1t, row, 2, 2, 2 * this.hiddenNum));
			ao.putRow(row, MatrixOps.getRowRange(a1t, row, 1, 2, 2 * this.hiddenNum));
		}
		z1 = MatrixOps.gFunction(ao, ae).transpose();

		a2 = w2.transpose().mmul(z1).add(b2);

		results = a2.mul(setT);
		
		int count = 0;
		
		double error = 0;
		for (int i = 0; i < results.length; i++){
			error += Math.log(1 + Math.exp(-1 * results.get(i)));
			
			if(results.get(i)>0)
				count += 1;
		
		}
		
		error = error / (double) setN;
		double accuracy = count/(double)setN;	
		
		return new Result(error, accuracy);
	}

	public String getAllParametersIndices(){
		String indexString = "0,";
		String temp;
		temp = momentum + "";
		int currLength = temp.length();
		indexString += currLength+",";
		
		temp = numerator + "";
		currLength += temp.length();
		indexString += currLength+",";
		
		temp = pow + "";
		currLength += temp.length();
		indexString += currLength+",";
		
		temp = MatrixOps.matrixToString(w1);
		currLength += temp.length();
		indexString += currLength+",";
		
		temp = MatrixOps.matrixToString(b1);
		currLength += temp.length();
		indexString += currLength+",";
		
		temp = MatrixOps.matrixToString(w2);
		currLength += temp.length();
		indexString += currLength+",";
		
		temp = MatrixOps.matrixToString(b2);
		currLength += temp.length();
		indexString += currLength;
		
		return indexString;
	}
	
	public String getAllParametersInString(){
		String paramString = momentum + "";
		paramString += numerator +"";
		paramString += pow + "";
		paramString += MatrixOps.matrixToString(w1);
		paramString += MatrixOps.matrixToString(b1);
		paramString += MatrixOps.matrixToString(w2);
		paramString += MatrixOps.matrixToString(b2);
		
		return paramString;
	}

	@Override
	public void printParams() {
	}


	@Override
	public int getcurrEpoch() {
		return currEpoch;
	}

	@Override
	public long getTrainTime() {
		return trainTime;
	}
	
	@Override
	public int[][] predict() {
		return null;
	}

	@Override
	public void saveState() {
	}

	@Override
	public void restoreState() {
	}

	@Override
	public void cleanUp() {
	}

	public void fetchParams(InputStream input){
		
		Input in = new Input(input);
		MLPModelParams params = kryo.readObject(in, MLPModelParams.class);
		w1 = params.w1;
		w2 = params.w2;
		b1 = params.b1;
		b2 = params.b2;
		trainTime = params.trainTime;
		currEpoch = params.currEpoch;
		//System.out.println("Fetched w1: " + w1);
		
	}
	
}
