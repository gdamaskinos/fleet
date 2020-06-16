/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import utils.DeviceInfo;
import java.io.*;
import java.net.Socket;
import org.json.JSONArray;
import org.json.JSONException;

public class RPCProfiler implements Profiler {

	private Socket clientSocket;
	private BufferedWriter buf;
	private BufferedReader in;
	private FileWriter writer;

	String hostName = "localhost";
	int portNumber = 9995;
	
	final double latencySLO, energySLO;


	public RPCProfiler(int portNumber, double latencySLO, double energySLO, String outputCsvPath) {
		this.portNumber = portNumber;
		this.latencySLO = latencySLO;
		this.energySLO = energySLO;
		
		System.out.println("[RPC Profiler] " + DeviceInfo.header());
		System.out.println(
				"IMPORTANT: must launch the python backend along with the Server: ```python src/main/python/profilerBackend.py & mvn tomcat7:run```");

		//output file
        // FIXME eclipse issues (no such file) -> works on the server
        File file = new File(outputCsvPath);
		try {
			writer = new FileWriter(file, false);
	        writer.write(DeviceInfo.header() + "\n");
	        writer.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}

		try {
			clientSocket = new Socket(hostName, portNumber);
			in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
			// PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
			OutputStreamWriter out = new OutputStreamWriter(clientSocket.getOutputStream());
			buf = new BufferedWriter(out);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("Connected to python profiler");
	}

	/**
	 * Sends stats for updating the model
	 */
	public synchronized void pushStats(String clientId, DeviceInfo stats) {
		
		//System.out.println("Pushing stats: " + stats.toString(-1, 1, false));
		String[] msg = stats.toString(-1, 1, false).split(",");
		send(msg);

		try {
			writer.write(stats.toString(-1, 1, false) + "\n");
			writer.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Sends stats (i.e., features) and receives the output mini-batch size
	 */
	public synchronized int getMiniBatchSize(String clientId, DeviceInfo stats) {
		int batchSize = -1;

		String[] msg = stats.toString(-1, 0, false).split(",");
		send(msg);
		double[] response = receive();
		batchSize = (int) response[0];
		System.out.println("real batch size for " + msg[3] + " is " + batchSize);

		try {
			writer.write(stats.toString(-1, 0, false) + "\n");
			writer.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return batchSize;
	}

	private void send(String[] arr) {
		String message = new JSONArray(arr).toString();
		try {
			buf.write(message + "\n");
			buf.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private double[] receive() {
		String temp, response = "";
		double[] res = null;

		try {
			while (response == "")
				while ((temp = in.readLine()) != null) {
					response += temp;
					if (response.length() > 1 && response.charAt(response.length() - 1) == ']')
						break;
				}

			//System.out.println("GOT" + response);

			try {

				JSONArray js = new JSONArray(response);

				res = new double[js.length()];
				for (int i = 0; i < js.length(); i++)
					res[i] = Double.parseDouble(js.getString(i));
				// System.out.println(js.getString(i));

				// System.out.println(Arrays.toString(res));

			} catch (JSONException e) {
				System.out.println("No array");
			}

		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		return res;

	}

	@Override
	public boolean continueRequests(String clientId) {
		return false;
	}

}
