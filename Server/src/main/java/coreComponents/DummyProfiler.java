/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import utils.DeviceInfo;

/**
 * Dummy profiler
 * @author damaskin
 *
 */
public class DummyProfiler implements Profiler {

	private int fixedOutput;	

	public DummyProfiler(int fixedOutput) {
		this.fixedOutput = fixedOutput;
	}
	
	public void pushStats(String clientId, DeviceInfo stats) {
		System.out.println("Push stats for clientId " + clientId + ": " + stats.toString(-1, 1, true));
	}
	
	public int getMiniBatchSize(String clientId, DeviceInfo stats) {
		System.out.println("Get mini-batch size for clientId " + clientId + ": " + stats.toString(-1, 0, true));
		return fixedOutput;
	}
	
	public boolean continueRequests(String clientId) {
		return false;
	}

}
