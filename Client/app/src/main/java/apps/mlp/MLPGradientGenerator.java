/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.mlp;

import coreComponents.GradientGenerator;
import utils.MatrixOperation;
import utils.MyDataset;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import org.jblas.DoubleMatrix;

import Jama.Matrix;

public class MLPGradientGenerator implements GradientGenerator {

    private Kryo kryo;
    private Matrix trainX;
    private Matrix trainT;

    private Matrix w1;
    private Matrix b1;
    private Matrix w2;
    private Matrix b2;

    private int trainN;

    private double fetchMiniBatchTime;
    private double fetchModelTime;
    private double computeGradientsTime;

    private int h1;
    private int d;

    /**
     * number of outer iterations of the model (how many times the model was trained on the whole trainingSet).
     */
    int currEpoch;
    /**
     * total training time
     */
    long trainTime;

    private Matrix sE_delta_w1, sE_delta_w2, sE_delta_b1, sE_delta_b2;

    public MLPGradientGenerator(){
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

        MLPModelParams params = kryo.readObject(in, MLPModelParams.class);
        // Log.d("INFO", String.valueOf(params.w1.length));


        w1 = new Matrix(params.w1.toArray2());
        w2 = new Matrix(params.w2.toArray2());
        b1 = new Matrix(params.b1.toArray2());
        b2 = new Matrix(params.b2.toArray2());

        trainTime = params.trainTime;
        currEpoch = params.currEpoch;

        fetchModelTime = System.currentTimeMillis() - t;

        //System.out.println("Fetched w1: " + w1);

    }

    public void computeGradient(Output output){

        double t1 = System.currentTimeMillis();


        this.trainN = trainX.getRowDimension();
        this.h1 = w1.getRowDimension()/2;
        this.d = w1.getColumnDimension();


        Matrix a1, a2, ae, ao, z1, x, a1t, temp;
        Matrix wnew = new Matrix(1, 2 * h1);
        Matrix r1 = new Matrix(1, 2 * h1);
        Matrix r2 = new Matrix(1, 1);
        int t, i, j, k;
        double Fval;

        // initialize E_delta params
        sE_delta_w1 = new Matrix(2*h1, d);
        sE_delta_w2 = new Matrix(h1, 1);
        sE_delta_b1 = new Matrix(2*h1, 1);
        sE_delta_b2 = new Matrix(1, 1);

        // compute over all the inputs the parameters
        for (i = 0; i < trainN; i++) {

            // get the ith input example from the x_input vector of d dims
            x = MatrixOperation.getRowAt(trainX, i);
            // get the corresponding target value
            t = (int) trainT.get(i, 0);

            // calculating the activations for layer 1
            // dims: 2h1 x 1
            a1 = w1.times(x.transpose()).plus(b1);

            // dims: 1 x 2h1
            a1t = a1.transpose();

            // dims: 1 x h1
            ae = MatrixOperation.getRowRange(a1t, 0, 2, 2, 2 * h1);
            ao = MatrixOperation.getRowRange(a1t, 0, 1, 2, 2 * h1);

            // dims: h1 x 1
            z1 = MatrixOperation.gFunction(ao, ae).transpose();

            a2 = w2.transpose().times(z1).plus(b2);

            // Equation 2
            // dims: 1 x 1
            for (k = 0; k < r2.getColumnDimension(); k++)
                r2.set(0, k, -t * Math.exp(-t * a2.get(k, 0)) / (1 + Math.exp(-t * a2.get(k, 0))));

            // Equation 3
            // dims: 1 x 2h1
            j = 0;
            for (k = 0; k < h1; k++) {
                wnew.set(0, j, w2.get(k, 0));
                j++;
                wnew.set(0, j, w2.get(k, 0));
                j++;
            }

            temp = r2.times(wnew);
            for (k = 0; k < r1.getColumnDimension(); k++) {
                // q % 2 == 0
                if ((k + 1) % 2 == 0)
                    Fval = ao.get(0, k / 2) * Math.exp(-ae.get(0, k / 2)) / (1 + Math.exp(-ae.get(0, k / 2)));
                else // q % 2 == 1
                    Fval = 1;
                r1.set(0, k, temp.get(0, k) * Fval / (1 + Math.exp(-ae.get(0, k / 2))));

            }

            // Equation 4
            // dims: h1 x 1
            sE_delta_w2 = r2.times(z1.transpose()).transpose().plus(sE_delta_w2);
            // Equation 5
            // dims: 2h1 x d
            sE_delta_w1 = r1.transpose().times(x).plus(sE_delta_w1);
            // Equation 6
            // dims: 1 x 1
            sE_delta_b2 = r2.plus(sE_delta_b2); // causes jblas INFO deleting ...
            // Equation 7
            // dims: 2h1 x 1
            sE_delta_b1 = r1.transpose().plus(sE_delta_b1);

        }
        kryo.register(MLPGradients.class);
        MLPGradients gradients = new MLPGradients(new DoubleMatrix(sE_delta_w1.getArray()), new DoubleMatrix(sE_delta_w2.getArray()), new DoubleMatrix(sE_delta_b1.getArray()), new DoubleMatrix(sE_delta_b2.getArray()), trainN);
        kryo.writeObject(output, gradients);
        computeGradientsTime += System.currentTimeMillis() - t1;

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
