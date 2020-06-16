/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import java.io.InputStream;
import com.esotericsoftware.kryo.io.Output;

import apps.SPSampler;

public interface SGDUpdater {
	
	/**
	 * Initializes-Resets Updater (e.g., set initial model params, get learning rate) 
	 * by using the information that the Driver sends with the initialization request
	 * @param input contains the serialized byte[] returned from XXXModel#getParams() of the Driver
	 * @param sampler reference to the sampler => updater - sampler communication
	 */
	void initialize(InputStream input, Sampler sampler);
	
	/**
	 * Fetches the current parameters of the model
	 * @param isComputationRequest true => computation request , false => evaluation request
	 * @return
	 */
	void getParameters(Output output, boolean isComputationRequest);

	/**
	 * Updates the current model with the serialized gradients
	 * @param input contains the serialized byte[] returned from GradientGenerator of the Client
	 */
	void update(InputStream input);
	
}
