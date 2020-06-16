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
public class SyncProfiler implements Profiler {

	final double latencySLO, energySLO;
	
	/**
	 * Number of currently connected clients
	 */
	int numConnected;
	
	int numClients;

	/**
	 * Starts accepting requests only after all the clients have been connected
	 * @param latencySLO
	 * @param energySLO
	 * @param numClients number of clients to wait for before serving the first request
	 */
	public SyncProfiler(double latencySLO, double energySLO, int numClients) {
		this.latencySLO = latencySLO;
		this.energySLO = energySLO;
		this.numClients = numClients;
	}
	

	public void pushStats(String clientId, DeviceInfo stats) {
		System.out.println("Push stats for clientId " + clientId + ": " + stats.toString(-1, 1, true));
	}
	
	public int getMiniBatchSize(String clientId, DeviceInfo stats) {
		System.out.println("Get mini-batch size for clientId " + clientId + ": " + stats.toString(-1, 0, true));
		numConnected++;
		
		if (numConnected < numClients)
			System.out.println(clientId + " waiting as numConnected= " + numConnected);

		while (numConnected < numClients) {
			try {
				Thread.sleep(3000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return 104;
	}
	
	public boolean continueRequests(String clientId) {
		return false;
	}

}
