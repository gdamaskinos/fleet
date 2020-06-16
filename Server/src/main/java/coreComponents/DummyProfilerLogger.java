/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

import utils.DeviceInfo;

/**
 * Pseudo-profiler that log profiler info (device stats) in a csv file and outputs a fixed mini-batch size
 * Can be used for collected the PHT for CALOREE
 * @author damaskin
 *
 */
public class DummyProfilerLogger implements Profiler {

	private FileWriter writer;
	
	private int fixedOutput;

	public DummyProfilerLogger(int fixedOutput) {

		this.fixedOutput = fixedOutput;
		
		System.out.println("[WriteCsv \"Profiler\"]");
		System.out.println(DeviceInfo.header());

		//output file
		// FIXME eclipse issues (no such file) -> works on the server	            
		String outputCsvPath = "src/main/resources/output/threads_test_output.csv";
		File file = new File(outputCsvPath);

		try {
		writer = new FileWriter(file, false);
		writer.write(DeviceInfo.header() + "\n");
		writer.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}


	public void pushStats(String clientId, DeviceInfo stats) {
		try {
			writer.write(stats.toString(-1, 1, false) + "\n");
			writer.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	

	public int getMiniBatchSize(String clientId, DeviceInfo stats) {
		try {
			writer.write(stats.toString(-1, 0, false) + "\n");
			writer.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
			
		return fixedOutput;
	}
	

	public boolean continueRequests(String clientId) {
		return false;
	}

}
