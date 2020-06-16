/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;


import com.esotericsoftware.kryo.io.Output;

public interface Sampler {

	/**
	 * Get miniBatch with the specified size and serialize it to Kryo output
	 * @param size
	 * @param output
	 */
	void getSample(int size, Output output);

	/**
	 * Reset sampler (triggered by the Driver) (e.g., reset example pool)
	 */
	void reset();
}
