/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.lr;

import android.util.Log;

import coreComponents.GradientGenerator;
import utils.Helpers;
import utils.MatrixOperation;
import utils.MyDataset;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import org.jblas.DoubleMatrix;

import Jama.Matrix;

public class LRGradientGenerator implements GradientGenerator {

    private Kryo kryo;
    private Matrix trainX;
    private Matrix trainT;

    private Matrix weights,biases;

    private int trainN;
    private double fetchMiniBatchTime;
    private double fetchModelTime;
    private double computeGradientsTime;

    //model Fetch time: index 0, batch FetchTime: index 1, batch gradientComputeTime: index 2


    private int featuresize, numlabels;

    /**
     * number of outer iterations of the model (how many times the model was trained on the whole trainingSet).
     */
    int currEpoch;
    /**
     * total training time
     */
    long trainTime;

    private Matrix sE_delta_weights, sE_delta_biases;
    private Matrix delta_weights, delta_biases;

    public LRGradientGenerator(){
        kryo = new Kryo();
    }

    public void fetch(Input in) {

        double t = System.currentTimeMillis();

        MyDataset miniBatch = kryo.readObject(in, MyDataset.class);
        // Log.d("INFO", String.valueOf(miniBatch.Tset.columns));

        fetchMiniBatchTime = System.currentTimeMillis() - t;

        this.trainX = new Matrix(miniBatch.Xset.toArray2());
        this.trainT = new Matrix(miniBatch.Tset.toArray2());

        t = System.currentTimeMillis();

        LRModelParams params = kryo.readObject(in, LRModelParams.class);
        // Log.d("INFO", String.valueOf(params.w1.length));

        weights = new Matrix(params.weights.toArray2());
        biases = new Matrix(params.biases.toArray2());

        trainTime = params.trainTime;
        currEpoch = params.currEpoch;

        fetchModelTime = System.currentTimeMillis() - t;

        //System.out.println("Fetched w1: " + w1);
        System.out.println("Epoch: "+currEpoch);
        System.out.println("Mini-batch size: "+trainX.getRowDimension());

        Log.d("INFO", "Total Bytes read: " + Helpers.humanReadableByteCount(in.total(), false));
    }

    public void computeGradient(Output output){

        double t1 = System.currentTimeMillis();


        kryo.register(Integer.class);
        kryo.writeObject(output, currEpoch);

        this.trainN = trainX.getRowDimension();
        this.featuresize = trainX.getColumnDimension();
        this.numlabels = trainT.getColumnDimension();


        Matrix x, predicted, label, Tlabel, L;

        int t, i, j, k;
        double Fval;

        // initialize E_delta params
        sE_delta_weights = new Matrix(numlabels, featuresize);
        delta_weights = new Matrix(numlabels, featuresize);
        sE_delta_biases = new Matrix(numlabels,1);
        delta_biases = new Matrix(1,numlabels);

        //long total_time=0;
        double responseTime;
        //double eachLoopTime=0;
        // compute over all the inputs the parameters
        for (i = 0; i < trainN; i++) {

            responseTime = System.currentTimeMillis();

            // get the ith input example from the x_input vector of d dims

            x = MatrixOperation.getRowAt(trainX, i);
            label = MatrixOperation.getRowAt(trainT, i);
            Matrix currentPredictY = x.times(weights.transpose()).plus(biases.transpose());

            currentPredictY = Softmax(currentPredictY);


            sE_delta_biases = label.minus(currentPredictY);
            sE_delta_weights = sE_delta_biases.transpose().times(x);
            //System.out.println("deltaW1: "+deltaW);

            sE_delta_biases = colsum(sE_delta_biases);//.times(1/(double)trainN);
            //sE_delta_weights.times(1/(double)trainN);

            delta_weights = delta_weights.plus(sE_delta_weights);
            delta_biases = delta_biases.plus(sE_delta_biases);

            //eachLoopTime = (System.currentTimeMillis() - responseTime);
            //System.out.println("Each loop time: "+eachLoopTime);
            //total_time += eachLoopTime;
        }
        //System.out.println("Average time: "+ total_time/(double)trainN);

        kryo.register(LRGradients.class);
        LRGradients gradients = new LRGradients(new DoubleMatrix(delta_weights.getArray()),new DoubleMatrix(delta_biases.getArray()), trainN);
        kryo.writeObject(output, gradients);
        computeGradientsTime = System.currentTimeMillis() - t1;

        Log.d("INFO", "Number of gradients: " + String.valueOf(gradients.delta_biases.length + gradients.delta_weights.length));

        Log.d("INFO", "Total Bytes written: " + Helpers.humanReadableByteCount(output.total(), false));

    }

    public static Matrix colsum(Matrix m) {
        int numRows = m.getRowDimension();
        int numCols = m.getColumnDimension();
        Matrix sum = new Matrix(1, numCols);
        // loop through the rows and compute the sum
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                sum.set(0, j, sum.get(0, j) + m.get(i, j));
            }
        }
        return sum;
    }


    /**
     * The softmax method
     * @param z
     * @return
     */
    private static Matrix Softmax(Matrix z) {

        int zlength = z.getColumnDimension() * z.getRowDimension();

        Matrix exp = new Matrix(1,zlength);

        for(int i=0; i < zlength; i++){
            exp.set(0,i,Math.exp(z.get(0,i)));
        }

        Matrix softmax = exp.times(1/ (double) sum(exp));

        return softmax;

    }

    public static double sum(Matrix m) {
        int numRows = m.getRowDimension();
        int numCols = m.getColumnDimension();
        double sum = 0;
        // loop through the rows and compute the sum
        for (int i=0; i<numRows; i++) {
            for (int j=0; j<numCols; j++) {
                sum += m.get(i,j);
            }
        }
        return sum;
    }

    @Override
    public int getSize() {
        return trainN;
    }

    @Override
    public double getFetchMiniBatchTime() {
        return fetchMiniBatchTime;
    }

    @Override
    public double getFetchModelTime() {
        return fetchModelTime;
    }

    @Override
    public double getComputeGradientsTime() {
        return computeGradientsTime;
    }


}
