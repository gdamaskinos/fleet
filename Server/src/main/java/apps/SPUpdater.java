/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps;

import java.io.InputStream;

import com.esotericsoftware.kryo.io.Output;

import apps.cppNN.CppNNUpdater;
import apps.dl4j.Dl4jUpdater;
import apps.lr.LRUpdater;
import apps.mlp.MLPUpdater;
import apps.simpleCNN.SimpleCNNUpdater;

import coreComponents.SGDUpdater;
import coreComponents.Sampler;

/**
 * Updater implemented by the Service Provider
 *
 */
public class SPUpdater implements SGDUpdater {

	// TODO: modify with the Application's Updater
	//public MLPUpdater upd = new MLPUpdater();
	//Dl4jUpdater upd = new Dl4jUpdater();
	//public LRUpdater upd = new LRUpdater();
	//SimpleCNNUpdater upd = new NativeCNNUpdater();
	public CppNNUpdater upd = new CppNNUpdater();
	
	public void initialize(InputStream input, Sampler sampler) {
		upd.initialize(input, sampler);
	}

	@Override
	public void getParameters(Output output, boolean isComputationRequest) {
		upd.getParameters(output, isComputationRequest);
	}

	@Override
	public void update(InputStream input) {
		upd.update(input);
	}

}
