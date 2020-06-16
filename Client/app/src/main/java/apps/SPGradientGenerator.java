/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps;


import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import apps.cppNN.CppNNGradientGenerator;
import apps.lr.LRGradientGenerator;
import coreComponents.GradientGenerator;


public class SPGradientGenerator implements GradientGenerator {

    // TODO: modify with the Application's SPGradientGenerator
    //SimpleSimpleCNNGradientGenerator gen = new SimpleSimpleCNNGradientGenerator();
    CppNNGradientGenerator gen = new CppNNGradientGenerator();
    //Dl4jGradientGenerator gen = new Dl4jGradientGenerator();
    //MLPGradientGenerator gen = new MLPGradientGenerator();
    //LRGradientGenerator gen = new LRGradientGenerator();

    public void fetch(Input input) {
        gen.fetch(input);
    }

    public void computeGradient(Output output){
        gen.computeGradient(output);
    }

    @Override
    public int getSize() {
        return gen.getSize();
    }

    @Override
    public double getFetchMiniBatchTime() {
        return gen.getFetchMiniBatchTime();
    }

    @Override
    public double getFetchModelTime() {
        return gen.getFetchModelTime();
    }

    @Override
    public double getComputeGradientsTime() {
        return gen.getComputeGradientsTime();
    }

}
