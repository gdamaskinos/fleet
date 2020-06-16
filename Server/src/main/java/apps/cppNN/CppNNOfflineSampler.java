/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.cppNN;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.DefaultArraySerializers.ByteArraySerializer;

import coreComponents.Sampler;
import utils.ByteVec;
import utils.Kardam;

public class CppNNOfflineSampler implements Sampler {

	private Kryo kryo;
	
	private native void initSampler(String s);
	private native byte[] getMiniBatch(int batchSize);

	
	/**
	 * Offline Sampler constructor
	 * @param prefix (e.g. /path/to/datasets/spambase_)
	 */
	public CppNNOfflineSampler(String prefix){
		
		// TODO prefix pass
		kryo = new Kryo();
		kryo.register(byte[].class, new ByteArraySerializer());
		initSampler(prefix);
	}

		
	public void getSample(int size, Output output) {
		synchronized(kryo) {
			kryo.writeObject(output, getMiniBatch(size));
		}
	}

	@Override
	public void reset() {
		// TODO 
		
	}

}
