/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.io.InputStream;

/**
 * Created by Mercury on 2016/11/10.
 */

public interface GradientGenerator {

    /**
     * Application specific method for computing gradients
     * @param output output stream for writting the serialized gradients
     */
    void computeGradient(Output output);

    /**
     * Application specific method for fetching parameters from the server
     * @param input stream which contains serialized model parameters and mini-batch (input, labels)
     **/
    void fetch(Input input);

    /**
     * Get the size of the mini-batch that the most recent gradients were computed on
     * @return
     */
    int getSize();


    double getFetchMiniBatchTime();

    double getFetchModelTime();

    double getComputeGradientsTime();

}
