/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils.dl4j;

import java.util.Collection;

/*
*
*  * Copyright 2015 Skymind,Inc.
*  *
*  *    Licensed under the Apache License, Version 2.0 (the "License");
*  *    you may not use this file except in compliance with the License.
*  *    You may obtain a copy of the License at
*  *
*  *        http://www.apache.org/licenses/LICENSE-2.0
*  *
*  *    Unless required by applicable law or agreed to in writing, software
*  *    distributed under the License is distributed on an "AS IS" BASIS,
*  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  *    See the License for the specific language governing permissions and
*  *    limitations under the License.
*
*/

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TerminationCondition;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.solvers.BaseOptimizer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Stochastic Gradient Descent Standard fix step size No line search
 * 
 * @author Adam Gibson
 */
public class MyStochasticGradientDescent extends BaseOptimizer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public MyStochasticGradientDescent(NeuralNetConfiguration conf, StepFunction stepFunction,
			Collection<IterationListener> iterationListeners, Model model) {
		super(conf, stepFunction, iterationListeners, model);
	}

	public MyStochasticGradientDescent(NeuralNetConfiguration conf, StepFunction stepFunction,
			Collection<IterationListener> iterationListeners, Collection<TerminationCondition> terminationConditions,
			Model model) {
		super(conf, stepFunction, iterationListeners, terminationConditions, model);
	}


    public Pair<Gradient, Double> gradientAndScore() {
        oldScore = score;
        model.computeGradientAndScore();

        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (IterationListener l : iterationListeners) {
                if (l instanceof TrainingListener) {
                    ((TrainingListener) l).onGradientCalculation(model);
                }
            }
        }

        Pair<Gradient, Double> pair = model.gradientAndScore();
        score = pair.getSecond();
  //      updateGradientAccordingToParams(pair.getFirst(), model, model.batchSize()); // !! changes model state
        return pair;
    }
    
	public Gradient getGradients() {
		Pair<Gradient, Double> pair = gradientAndScore();
		Gradient gradient = pair.getFirst();
		return gradient;
	}

	public void descent(Gradient gradient) {
		updateGradientAccordingToParams(gradient, model, 1);

		INDArray params = model.params();
		/* Descent */
		stepFunction.step(params, gradient.gradient());
		// Note: model.params() is always in-place for MultiLayerNetwork and
		// ComputationGraph, hence no setParams is necessary there
		// However: for pretrain layers, params are NOT a view. Thus a
		// setParams call is necessary
		// But setParams should be a no-op for MLN and CG
		model.setParams(params);
		
//		for (IterationListener listener : iterationListeners)
//			listener.iterationDone(model, 0);
//
//		checkTerminalConditions(gradient.gradient(), oldScore, score, 0);
//
//		BaseOptimizer.incrementIterationCount(model, 1);
	}
	
	@Override
	public boolean optimize() {
		for (int i = 0; i < conf.getNumIterations(); i++) {
			/* Gradient */
			Pair<Gradient, Double> pair = gradientAndScore();
			Gradient gradient = pair.getFirst();
			INDArray params = model.params();

			/* Descent */
			stepFunction.step(params, gradient.gradient());
			// Note: model.params() is always in-place for MultiLayerNetwork and
			// ComputationGraph, hence no setParams is necessary there
			// However: for pretrain layers, params are NOT a view. Thus a
			// setParams call is necessary
			// But setParams should be a no-op for MLN and CG
			model.setParams(params);
			
			for (IterationListener listener : iterationListeners)
				listener.iterationDone(model, i);

			checkTerminalConditions(pair.getFirst().gradient(), oldScore, score, i);

			BaseOptimizer.incrementIterationCount(model, 1);
		}
		return true;
	}

	@Override
	public void preProcessLine() {
	}

	@Override
	public void postStep(INDArray gradient) {
	}
}
