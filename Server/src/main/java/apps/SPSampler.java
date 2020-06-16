/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps;

import com.esotericsoftware.kryo.io.Output;

import apps.cppNN.CppNNOfflineSampler;
import apps.dl4j.Dl4jOfflineSampler;
import apps.dl4j.Dl4jStreamOfflineSampler;
import apps.lr.LROfflineSampler;
import apps.mlp.MLPOfflineSampler;
import apps.simpleCNN.SimpleCNNOfflineSampler;

import coreComponents.Sampler;


public class SPSampler implements Sampler {

	// TODO: modify with the Application's Sampler
	//MLPOfflineSampler sampler;
	//Dl4jOfflineSampler sampler;
	//public LROfflineSampler sampler;
	//SimpleCNNOfflineSampler sampler;
	public CppNNOfflineSampler sampler;
		
	public SPSampler(String prefix) {
		//sampler = new LROfflineSampler(prefix);
		//sampler = new MLPOfflineSampler(prefix);
		//sampler = new Dl4jOfflineSampler(prefix);
		//sampler = new SimpleCNNOfflineSampler(prefix);
		sampler = new CppNNOfflineSampler(prefix);
	}
	
	@Override
	public void getSample(int size, Output output) {
		sampler.getSample(size, output);
	}

	@Override
	public void reset() {
		sampler.reset();
	}
}
