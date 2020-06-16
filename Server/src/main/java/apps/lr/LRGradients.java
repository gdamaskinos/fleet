/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.lr;
import org.jblas.DoubleMatrix;

/**
 * Gradients computed on a mini-batch
 */
public class LRGradients {

    /**
     * layer weights, biases
     */
    public DoubleMatrix delta_weights;
    public DoubleMatrix delta_biases;
    /**
     * mini-batch size
     */
    public int batchSize;

    public LRGradients() {

    }
    public LRGradients(DoubleMatrix delta_weights, DoubleMatrix delta_biases, int batchSize) {
        this.delta_weights = delta_weights;
        this.delta_biases = delta_biases;
        this.batchSize = batchSize;

    }
}
