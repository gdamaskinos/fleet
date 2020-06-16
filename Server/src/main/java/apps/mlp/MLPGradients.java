/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.mlp;
import org.jblas.DoubleMatrix;

/**
 * Gradients computed on a mini-batch
 */
public class MLPGradients {

    /**
     * layer weights
     */
    public DoubleMatrix delta_w1, delta_w2;
    /**
     * bias parameters
     */
    public DoubleMatrix delta_b1, delta_b2;

    /**
     * mini-batch size
     */
    public int batchSize;

    public MLPGradients() {

    }
    public MLPGradients(DoubleMatrix delta_w1, DoubleMatrix delta_w2, DoubleMatrix delta_b1, DoubleMatrix delta_b2, int batchSize) {
        this.delta_w1 = delta_w1;
        this.delta_w2 = delta_w2;
        this.delta_b1 = delta_b1;
        this.delta_b2 = delta_b2;
        this.batchSize = batchSize;

    }
}
