/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.mlp;

import org.jblas.DoubleMatrix;

public class MLPModelParams {


    public MLPModelParams() {

    }

    public MLPModelParams(double momentum, double numerator, double pow, double sigma, int featureSize, int hiddenNum,
                          DoubleMatrix w1, DoubleMatrix w2, DoubleMatrix b1, DoubleMatrix b2, int currEpoch, long trainTime) {
        this.momentum = momentum;
        this.numerator = numerator;
        this.pow = pow;
        this.sigma = sigma;
        this.featureSize = featureSize;
        this.hiddenNum = hiddenNum;
        this.w1 = w1;
        this.w2 = w2;
        this.b1 = b1;
        this.b2 = b2;
        this.currEpoch = currEpoch;
        this.trainTime = trainTime;
    }

    public double momentum;
    public double numerator;
    public double pow;
    public double sigma;

    public int featureSize;
    public int hiddenNum;

    /**
     * layer weights
     */
    public DoubleMatrix w1, w2;
    /**
     * bias parameters
     */
    public DoubleMatrix b1, b2;
    /**
     * number of outer iterations of the model (how many times the model was
     * trained on the whole trainingSet).
     */
    public int currEpoch;
    /**
     * total training time
     */
    public long trainTime;

}
