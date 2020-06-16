/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.lr;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.Arrays;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import coreComponents.Model;
import utils.MyDataset;
import utils.MatrixOps;
import utils.Result;

public class LRModel implements Model {

	private Kryo kryo;
	private double rate;
	private int featuresize;
	private int numlabels;
	
	/**
	 * layer weights,biases
	 */
	private DoubleMatrix weights;
	private DoubleMatrix biases;
	/**
	 * number of outer iterations of the model (how many times the model was trained on the whole trainingSet).
	 */
	int currEpoch;
	/**
	 * total training time
	 */
	long trainTime;
	
	public LRModel(double rate, int featuresize, int numlabels){
		kryo = new Kryo();
		kryo.register(LRModelParams.class);
		
		this.rate = rate;
		System.out.println("Learning rate: " + rate);
		this.featuresize = featuresize;
		this.numlabels = numlabels;
		
		this.initialize();
		
	}

	public byte[] getParams() {
		//System.out.println("Sending w1: " + w1);
		LRModelParams params = new LRModelParams(rate, featuresize, numlabels, weights, biases, currEpoch, trainTime);
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		Output output = new Output(outputStream);

		kryo.writeObject(output, params);
		output.close();

		return outputStream.toByteArray();
	}
	
	
	public void initialize() {
		weights = DoubleMatrix.zeros(numlabels,featuresize);
		biases = DoubleMatrix.zeros(numlabels);
	}
	public Integer[] returnNMaxInd(double[] arr, int N){
		Integer[] ind= new Integer[N];
		for(int i=0;i<N;i++){
			ind[i]=returnMaxInd(arr);
			arr[ind[i]]=0;
		}
		
		return ind;
	}
	
	
	public int returnMaxInd(double[] arr){
		int maxIndex = 0;
	    for (int i = 1; i < arr.length; i++) {
	        double newnumber = arr[i];
	        if ((newnumber > arr[maxIndex])) {
	            maxIndex = i;
	        }
	    }
	    return maxIndex;
	}
	public boolean contains(final Integer[] array, final int key) {
	    return Arrays.asList(array).contains(key);
	}
	
	public Result evaluate(MyDataset dataset) {
		
		//DoubleMatrix a1, a2, ae, ao, z1, a1t, results, setX, setT;
		DoubleMatrix CF, setX, setT;
		
		int setN;

		setX = dataset.Xset;
		setT = dataset.Tset;
		System.out.println(dataset.Xset.rows + " " + dataset.Xset.rows);
		setN = setX.rows;

		double error = 0;
		int count = 0;

		for (int i=0; i<setX.rows; i++) {

			
			DoubleMatrix x = setX.getRow(i);
			DoubleMatrix label = setT.getRow(i);
	        DoubleMatrix currentPredictY = x.mmul(weights.transpose()).addiRowVector(biases);
	        currentPredictY = Softmax(currentPredictY);

	        //System.out.println("Probabilities: "+currentPredictY);
	        
			Integer[] pred = returnNMaxInd(currentPredictY.toArray(),10);
			//System.out.println("pred: "+Arrays.toString(pred)+" label: "+predicted.argmax()+" results: "+contains(pred,predicted.argmax()));
			if (contains(pred,label.argmax()))
				count = count + 1;

			//weights update
		    //weights = weights.add((Tlabel.sub(predicted)).mmul(x).mul(rate)).sub(weights.mul(2*mu*rate));
		}
		
		double accuracy = count/(double)setN;	
		
		return new Result(getValue(setX,setT), accuracy);
	}
	
    public double getValue(DoubleMatrix myXSamples, DoubleMatrix myYSamples) {
    	DoubleMatrix currentPredictY;
        currentPredictY = myXSamples.mmul(weights.transpose()).addiRowVector(biases);
        currentPredictY = Softmax(currentPredictY);
        double loss = MatrixFunctions.powi(currentPredictY.sub(myYSamples), 2).sum() / myXSamples.rows;
        //loss += 0.5 * lambda* (MatrixFunctions.pow(weights, 2).sum() + MatrixFunctions.pow(biases, 2).sum());
        return loss;
    }

	public String getAllParametersIndices(){
		String indexString = "0,";
		String temp;
		temp = rate + "";
		int currLength = temp.length();
		indexString += currLength+",";
		
		temp = featuresize + "";
		currLength += temp.length();
		indexString += currLength+",";

		temp = numlabels + "";
		currLength += temp.length();
		indexString += currLength+",";

		
		temp = MatrixOps.matrixToString(weights);
		currLength += temp.length();
		indexString += currLength+",";

		temp = MatrixOps.matrixToString(biases);
		currLength += temp.length();
		indexString += currLength+",";

		return indexString;
	}
	
	public String getAllParametersInString(){
		String paramString = rate + "";
		paramString += featuresize +"";
		paramString += numlabels +"";
		paramString += MatrixOps.matrixToString(weights);
		paramString += MatrixOps.matrixToString(biases);
		return paramString;
	}

	
    public DoubleMatrix Softmax(DoubleMatrix y) {
        DoubleMatrix max = y.rowMaxs();
        MatrixFunctions.expi(y.subiColumnVector(max));
        DoubleMatrix sum = y.rowSums();
        y.diviColumnVector(sum);
        return y;
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
		LRModelParams params = kryo.readObject(in, LRModelParams.class);
		weights = params.weights;
		biases = params.biases;
		trainTime = params.trainTime;
		currEpoch = params.currEpoch;
		//System.out.println("Fetched w1: " + w1);
		
	}
	
}
