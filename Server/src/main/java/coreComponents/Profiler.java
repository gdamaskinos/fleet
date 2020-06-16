/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import utils.DeviceInfo;

public interface Profiler {

	/**
	 * Sends stats for updating the model
	 */
	public void pushStats(String clientId, DeviceInfo stats);
	
	/**
	 * Sends stats (i.e., features) and receives the output mini-batch size
	 */
	public int getMiniBatchSize(String clientId, DeviceInfo stats); 
	
	/**
	 * Useful for implementing a dispatcher profiler
	 * Returns whether the client should send another request for the current profilers' epoch
	 * i.e., this will return true as many times as the profilers size is
	 * @param clientId
	 * @return
	 */
	public boolean continueRequests(String clientId); 
}
