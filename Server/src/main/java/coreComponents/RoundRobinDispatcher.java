/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import java.io.FileWriter;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jsat.classifiers.linear.PassiveAggressive.Mode;
import utils.DeviceInfo;

public class RoundRobinDispatcher implements Profiler {

	/**
	 * List with all to-be-compared profilers
	 */
	private List<Profiler> profilers;

	/**
	 * Request number used for dispatching the corresponding profiler (round robin manner)
	 */
	private HashMap<String, Integer> requestId;

	/**
	 * Number of currently connected clients
	 */
	int numConnected;
	
	int numClients;

	/**
	 * @param latencySLO
	 * @param energySLO
	 * @param numClients
	 */
	public RoundRobinDispatcher(double latencySLO, double energySLO, int numClients) {
		System.out.println("RR dispatcher with numClients: " + numClients);

		profilers = new ArrayList<>();
		
//		profilers.add(new DummyProfiler(latencySLO, energySLO, "DUMMY1"));
//		profilers.add(new DummyProfiler(latencySLO, energySLO, "DUMMY2"));
		
		profilers.add(new RPCProfiler(9991, latencySLO, energySLO, "src/main/resources/output/RPC1_output.csv"));
		profilers.add(new RPCProfiler(9992, latencySLO, energySLO, "src/main/resources/output/RPC2_output.csv"));
		
//		pa_1 = new PAProfiler("", latencySLO, soc1, Mode.PA, false, 1, 0.01, Math.pow(10.0, -3), "1", true, false, false);
//		pa_1 = new PAProfiler("", latencySLO, soc1, Mode.PA, false, 1, 0.01, Math.pow(10.0, -3), "1", true, false, true);
//		pa_2 = new PAProfiler("", latencySLO, soc2, Mode.PA, false, 1, 0.01, Math.pow(10.0, -3), "atStats", false, false, true);
//		pa_3 = new PAProfiler("", latencySLO, soc3, Mode.PA, false, 1, 0.01, Math.pow(10.0, -3), "scale", true, true, true);

//		lasso = new LASSOProfiler("", latencySLO, energySLO, soc2, 10, false);
//		lasso = new Profiler("", latencySLO, energySLO, soc2, 10, true);

		requestId = new HashMap<>(); 
		
		this.numClients = numClients;

	}
	
		
	public int getMiniBatchSize(String clientId, DeviceInfo stats) {
	
		System.out.println("Get mini-batch size for clientId " + clientId);
		numConnected++;
		
		if (numConnected < numClients)
			System.out.println(clientId + " waiting as numConnected = " + numConnected + " < " + numClients);

		while (numConnected < numClients) {
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		synchronized (requestId) {
			if (!requestId.containsKey(clientId))
				requestId.put(clientId, 0);
			else
				requestId.put(clientId, requestId.get(clientId)+1);
		
			int currProfiler = requestId.get(clientId) % profilers.size();
			return profilers.get(currProfiler).getMiniBatchSize(clientId, stats);
		}
		
	}

	public void pushStats(String clientId, DeviceInfo stats) {

		synchronized (requestId) {
			int currProfiler = requestId.get(clientId) % profilers.size();
			profilers.get(currProfiler).pushStats(clientId, stats);
		}
	}

	public boolean continueRequests(String clientId) {
		return false;
	}
}
