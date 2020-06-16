/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import apps.SPSampler;
import apps.SPUpdater;
import apps.lr.LRModelParams;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import org.jblas.DoubleMatrix;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import utils.DeviceInfo;
import utils.Helpers;
import utils.JNITest;
import utils.MatrixOps;

import javax.servlet.Servlet;
import javax.servlet.ServletConfig;
import javax.servlet.ServletException;
import javax.servlet.annotation.MultipartConfig;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.Part;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.*;
import java.util.zip.GZIPOutputStream;

import jsat.classifiers.linear.PassiveAggressive;
import jsat.classifiers.linear.PassiveAggressive.Mode;

/**
 * Servlet implementation class Server
 */
@MultipartConfig
@WebServlet("/Server")
public class MasterOrchestrator extends HttpServlet {
	private static final long serialVersionUID = 1L;

	private Sampler sampler;
	private SGDUpdater updater;

	// target latency=1000ms
	private double latencySLO = 4000;
	private double energySLO = 0.04;

	private int iterations;

	private Kryo kryo;

	private int batchSizesIdx;
	
	/**
	 * Batch size profiler predictions for random size assignment
	 */
	private ArrayList<Integer> sizePred;
	private Random r;

	/**
	 * @see HttpServlet#HttpServlet()
	 */
	public MasterOrchestrator() {
		super();
	}

	private Profiler profiler;

	/**
	 * @see Servlet#init(ServletConfig)
	 */
	public void init(ServletConfig config) throws ServletException {

		new JNITest().hello();

		// profiler = new LASSOProfiler profiler;
		// profiler = new PAProfiler profiler;
		//profiler = new RPCProfiler(9995, latencySLO, energySLO, "src/main/resources/RPC.csv");
		profiler = new DummyProfiler(104);
		//profiler = new DummyProfilerLogger(104);
		//profiler = new SyncProfiler(latencySLO, energySLO, 5);
		//profiler = new RoundRobinDispatcher(latencySLO, energySLO, 0);
		//profiler = new ScheduleDispatcher(latencySLO, energySLO, "src/main/resources/schedule.csv");
		//profiler = new ScheduleDispatcher(latencySLO, energySLO, 2, 3);

		
		batchSizesIdx = 0;
		
		//sizePred = new ArrayList<Integer>();
		r = new Random(42);

		System.out.println("Request Type\tClientID\tTime");

	}

	@Override
	public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String clientType = null;

		Collection<Part> parts = request.getParts();
		for (Part part : parts) {
			if (part.getName().equals("clientType")) {
				clientType = getValue(part);
			}
		}

		if (clientType.equals("Initialize"))
			handleInitializeRequest(request, response);
		else if (clientType.equals("Eval"))
			handleEvaluationRequest(response);
		else if (clientType.equals("Gradient"))
			handleGradientResponse(request, response);
		else if (clientType.equals("Compute"))
			handleComputationRequest(request, response);
		else if (clientType.equals("Stats"))
			handleStatsResponse(request, response);

	}

	/**
	 * setup the updater, sampler and other parameters according to driver's request
	 *
	 * @param request
	 * @param response
	 * @throws IOException
	 * @throws NumberFormatException
	 * @throws ServletException
	 */
	private void handleInitializeRequest(HttpServletRequest request, HttpServletResponse response)
			throws NumberFormatException, IOException, ServletException {
		System.out.println("HTTP: Initialization POST request");
		String prefix = null;
		InputStream model = null;
		Collection<Part> parts = request.getParts();
		for (Part part : parts) {
			// get training iterations
			if (part.getName().equals("iterations")) {
				this.iterations = Integer.parseInt(getValue(part));
				System.out.println("Iterations: " + iterations);
			}
			// get prefix for datasets
			if (part.getName().equals("prefix")) {
				prefix = getValue(part);
				System.out.println("Prefix: " + prefix);
			}
			// get the serialized initialized model
			if (part.getName().equals("model")) {
				model = new Input(part.getInputStream());
				// System.out.println(IOUtils.toString(in, "UTF-8"));
			}
		}

		// initialize sampler
		// reload dataset if prefix is given
		if (prefix != null)
			sampler = new SPSampler(prefix);
		else { // reset it
			System.out.println("!No prefix given. A previous initialization request with prefix is crucial.");
			System.out.println("Resetting sampler...");
			sampler.reset();
		}

		// initialize updater
		updater = new SPUpdater();
		updater.initialize(model, sampler);

		try {
			// response.getWriter().append("Initialization success!");
			// response.getWriter().flush();
			OutputStream output = response.getOutputStream();
			String s = "Initialization success!";
			output.write(s.getBytes());
			output.flush();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Returns the text value of the given part.
	 */
	private String getValue(Part part) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(part.getInputStream(), "UTF-8"));
		StringBuilder value = new StringBuilder();
		char[] buffer = new char[10240];
		for (int length = 0; (length = reader.read(buffer)) > 0;) {
			value.append(buffer, 0, length);
		}
		return value.toString();
	}

	/**
	 * Handle an android client request for computing gradients on a mini-batch
	 * 
	 * @param request
	 * @param response
	 * @throws IOException
	 * @throws ServletException
	 */
	private void handleComputationRequest(HttpServletRequest request, HttpServletResponse response)
			throws IOException, ServletException {
		
		if (sampler == null) {
			System.out.println("Rejecting computation request due to unitialized server!");
			return;
		}
		
		long t = System.currentTimeMillis();
		String clientId = null, stats = null, androidInfo = null;
		Collection<Part> parts = request.getParts();
		for (Part part : parts) {
			// get training iterations
			if (part.getName().equals("clientID"))
				clientId = getValue(part);
			// get statistics
			if (part.getName().equals("stats")) {
				stats = getValue(part);
			}
			if (part.getName().equals("androidInfo")) {
				androidInfo = getValue(part);
			}
		}

		/**
		 * See Profiler#continueRequests()
		 */
		boolean continueRequests = false;
		int batchSize = -1;


		// profiler invoke
		if (stats != null) {
			DoubleMatrix deviceMatrix = MatrixOps.readMatrix(new BufferedReader(new StringReader(stats)));
			DeviceInfo deviceInfo = new DeviceInfo(clientId, deviceMatrix, androidInfo);
			batchSize = profiler.getMiniBatchSize(clientId, deviceInfo);
			continueRequests = profiler.continueRequests(clientId);
		}

		// hardcode the batch size
	    //batchSize = 104;

		// random size from profiles sizes (in size_predictions.csv file)
		//int index = r.nextInt(sizePred.size());
		//int index = batchSizesIdx;
		//batchSizesIdx = (batchSizesIdx + 1) % (sizePred.size() - 1);
		//batchSize = sizePred.get(index);

		// random size~Gaussian(mean-3sigma=1, mean+3sigma=max_size)
		// double max = sizePred.get(sizePred.size()-1);
		//double max = 199;
		//double mean = (max + 1) / 2.0;
		//double sigma = (mean - 1) / 3.0;
		//batchSize = (int) Math.round(Math.max(1, r.nextGaussian() * sigma + mean));

		// Minimum batch size value
		if (batchSize <= 0)
			batchSize = 1;
		// Maximum batch size value
		if (batchSize > 10000) {
			System.out.println("MO: MiniBatch size too large. Reducing from " + batchSize + " to maximum possible.");
			batchSize = 10000;
		}

		//round up the batch size to a multiple of 8
		//most Android device have 4 or 8 CPUs
		//batchSize = batchSize + 8 - (batchSize % 8);

		//System.out.println("Batch size: " + batchSize);
				
		// write response (miniBatch + model)
		OutputStream output = new GZIPOutputStream(response.getOutputStream());
		// OutputStream output = response.getOutputStream();
		Output out = new Output(output);

		kryo = new Kryo();
		kryo.register(Boolean.class);
		kryo.writeObject(out, continueRequests);

		// send miniBatch first to avoid buffer overflow due to model serialization issues
		this.sampler.getSample(batchSize, out);
		this.updater.getParameters(out, true);

		System.out.println("Written: " + Helpers.humanReadableByteCount(out.total(), false));
		out.close();
		System.out.println("HTTP: Computation\t" + clientId + "\t" + (System.currentTimeMillis() - t));

	}

	/**
	 * Driver evaluation request handler. send the current parameters to a driver
	 * 
	 * @param response
	 * @throws IOException
	 */
	private void handleEvaluationRequest(HttpServletResponse response) throws IOException {
		long t = System.currentTimeMillis();

		OutputStream output = response.getOutputStream();
		Output out = new Output(output);
		updater.getParameters(out, false);
		out.close();
		// output.write(Files.readAllBytes(Paths.get("test.txt")));
		System.out.println("HTTP: Evaluation\t\t" + (System.currentTimeMillis() - t));
	}

	/**
	 * Handle a reply with stats from a client
	 * 
	 * @param request
	 * @throws ServletException
	 * @throws IOException
	 */
	private void handleStatsResponse(HttpServletRequest request, HttpServletResponse response)
			throws IOException, ServletException {
		long t = System.currentTimeMillis();
		String clientId = null, stats = null, androidInfo = null;
		Collection<Part> parts = request.getParts();
		for (Part part : parts) {
			if (part.getName().equals("clientID")) {
				clientId = getValue(part);
			}
			// get statistics
			if (part.getName().equals("stats")) {
				stats = getValue(part);
			}
			if (part.getName().equals("androidInfo")) {
				androidInfo = getValue(part);
			}
		}


		// forward stats
		if (stats != null) {

			//System.out.println("Ticket: "+ticket);

			DoubleMatrix deviceMatrix = MatrixOps.readMatrix(new BufferedReader(new StringReader(stats)));
			DeviceInfo deviceInfo = new DeviceInfo(clientId, deviceMatrix, androidInfo);
			profiler.pushStats(clientId, deviceInfo);
		}

		System.out.println("HTTP: Stats\t" + clientId + "\t" + (System.currentTimeMillis() - t));
	}

	/**
	 * Handle a reply with gradients from a client
	 * 
	 * @param request
	 * @throws ServletException
	 * @throws IOException
	 */
	private void handleGradientResponse(HttpServletRequest request, HttpServletResponse response)
			throws IOException, ServletException {
		long t = System.currentTimeMillis();
		String clientId = null, stats = null;
		Collection<Part> parts = request.getParts();
		for (Part part : parts) {
			if (part.getName().equals("clientID")) {
				clientId = getValue(part);
			}
			// get statistics
			if (part.getName().equals("stats")) {
				stats = getValue(part);
			}
			// get the serialized gradients
			if (part.getName().equals("gradients")) {
				InputStream in = new Input(part.getInputStream()); // input for getting the model

				// write output
				try {
					OutputStream output = response.getOutputStream();
					String s = "HTTP: Gradients Received!";
					output.write(s.getBytes());
					output.flush();

				} catch (IOException e) {
					e.printStackTrace();
				}

				this.updater.update(in);
				// currEpoch ++;
			}
		}

		// System.out.println("CLIENT1: "+batchSize);
		System.out.println("HTTP: Gradient\t" + clientId + "\t" + (System.currentTimeMillis() - t));
	}

}
