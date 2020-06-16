/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import au.com.bytecode.opencsv.CSVReader;
import jsat.classifiers.linear.PassiveAggressive.Mode;
import utils.DeviceInfo;

public class ScheduleDispatcher implements Profiler {

	/**
	 * List with all to-be-compared profilers
	 */
	private List<Profiler> profilers;

	/**
	 * Request number used for dispatching the corresponding profiler (round robin manner)
	 */
	private HashMap<String, Integer> clientIds;
	private int currId;

	private int scheduleIdx, counter;
	
	private Integer[] schedule;

	/**
	 * Read hardcoded schedule from csv file
	 * @param latencySLO
	 * @param energySLO
	 * @param path e.g., ```cat src/main/resources/schedule.csv``` 1,1,2,1,2
	 */
	public ScheduleDispatcher(double latencySLO, double energySLO, String path) {
		System.out.println("Schedule dispatcher with path: " + path);

		profilers = new ArrayList<>();
		
		profilers.add(new RPCProfiler(9991, latencySLO, energySLO, "src/main/resources/output/RPC1_output.csv"));
		profilers.add(new RPCProfiler(9992, latencySLO, energySLO, "src/main/resources/output/RPC2_output.csv"));
		

		clientIds = new HashMap<>();
		currId = 1;
		
		String[] vals = null;
		try (CSVReader csvReader = new CSVReader(new FileReader(path));) {
		    vals = csvReader.readNext();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		schedule = new Integer[vals.length];
		for (int i=0; i<vals.length; i++) 
			schedule[i] = Integer.parseInt(vals[i]);
		
		System.out.println("Request schedule: " + Arrays.toString(schedule));
		scheduleIdx = 0;
		counter = 0;
	}
	
	/**
	 * Creates a random schedule
	 * @param latencySLO
	 * @param energySLO
	 * @param numClients total number of clients
	 * @param requestsPerClient number of requests for each client
	 */
	public ScheduleDispatcher(double latencySLO, double energySLO, int numClients, int requestsPerClient) {

		profilers = new ArrayList<>();
		
		profilers.add(new DummyProfiler(104));
		profilers.add(new DummyProfiler(204));
		

		clientIds = new HashMap<>();
		currId = 1;
		
		ArrayList<Integer> vals = new ArrayList<>();
		
		for (int id=1; id<=numClients; id++) 
			for (int j=0; j<requestsPerClient; j++)
				vals.add(id);
		
		Collections.shuffle(vals, new Random(42));
		
		schedule = vals.toArray(new Integer[vals.size()]);
		
		System.out.println("Request schedule: " + Arrays.toString(schedule));
		scheduleIdx = 0;
		counter = 0;
	}
	
	
	public int getMiniBatchSize(String clientId, DeviceInfo stats) {
	
		if (!clientIds.containsKey(clientId))
			clientIds.put(clientId, currId++);
		
		if (schedule[scheduleIdx] != clientIds.get(clientId)) {
			System.out.println(clientId + " waits for clientId: " + schedule[scheduleIdx] + " ... Total connected: " + clientIds.size());
			while (schedule[scheduleIdx] != clientIds.get(clientId)) {
				try {
					Thread.sleep(1000);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		
		return profilers.get(counter).getMiniBatchSize(clientId, stats);
	}

	public void pushStats(String clientId, DeviceInfo stats) {

		if (schedule[scheduleIdx] != clientIds.get(clientId)) {
			System.err.println("Something went wrong!!");
		}
		
		profilers.get(counter++).pushStats(clientId, stats);
		
		if (counter == profilers.size()) { // reset and unlock
			counter = 0;
			scheduleIdx++;
		}
		
	}

	public boolean continueRequests(String clientId) {
		return false;
	}
}
