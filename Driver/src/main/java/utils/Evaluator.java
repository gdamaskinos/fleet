/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import java.io.IOException;

import org.apache.http.HttpEntity;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;

import coreComponents.Model;

/**
 * Communicates with the server for the updated version of the model
 * Evaluates the model with the specified manner
 */
public class Evaluator {
	public Evaluator(String serverPath, CsvWriter csvWriter) {
		this.serverPath = serverPath;
		this.csvWriter = csvWriter;
	}

	public Evaluator(String serverPath) {
		this.serverPath = serverPath;
	}

	double res;
	CsvWriter csvWriter;
	String serverPath;

	/**
	 * Cross validation for a model
	 * 
	 * @param m
	 *            model to validate
	 * @param rounds
	 *            number of times to evaluate the model
	 * @param timeStep
	 *            time interval (in sec) between the evaluations of the model. Set 0 to evaluate in every epoch.
	 * @param thres
	 *            threshold for early stopping at kth evaluation: Error{k} >
	 *            {@linkplain thres} * Error{optimal}
	 * @param evalTrain
	 * 			 whether or not to evaluate on Training set. Set to false when training set is big to avoid OoM errors
	 * @throws IOException
	 * @throws InterruptedException 
	 */
	public void crossValidation(MyDataset trainingSet, MyDataset validationSet, MyDataset testSet, Model m, int rounds, double timeStep, double thres, boolean evalTrain) throws IOException, InterruptedException {
		if (rounds == 0)
			return;
		
		double minError; // used for early stopping
		int optimalRound; // epoch of the optimal model
		int currRound = 1; // current test round
		String out;
		Result res;
		int prevEpoch=-1;

		m.initialize();
		m.printParams();

		minError = Double.MAX_VALUE;
		optimalRound = -1;
		System.out.println("round,epoch,trE,trA,valE,valA,Time(ms)");
		while (currRound <= rounds) {
			Thread.sleep((long) (timeStep * 1000));
			// new test round
			receiveFromServer(serverPath, m);
			if (m.getcurrEpoch() <= prevEpoch) { // server send older model (on-demand staleness)
				System.out.println("Got epoch : " + m.getcurrEpoch() + " prevEpoch: " + prevEpoch + ". Skipping evaluation...");
				continue;
			}
			prevEpoch = m.getcurrEpoch();
			// evaluate for this test round
			out = "eval:" + currRound + "," + m.getcurrEpoch() + ",";
			if (evalTrain) {
				res = m.evaluate(trainingSet);
				out += String.format("%.4f,", res.error);
				out += String.format("%.4f,", res.accuracy);
			}
			else {
				out += String.format("%.4f,", -1.0);
				out += String.format("%.4f,", -1.0);
			}
			res = m.evaluate(validationSet);
			out += String.format("%.4f,", res.error);
			out += String.format("%.4f,", res.accuracy);
			out += m.getTrainTime();

			System.out.println(out);
			
			// early stopping
			if (res.error < minError) {
				minError = res.error;
			//	m.saveState();
				optimalRound = currRound;
			} else if (res.error > thres * minError) {
				System.out.println("Early stopped!");
				//break;
			}
			currRound++;
		}

		System.out.println("Optimal Test Round:," + optimalRound);
		//m.restoreState();
		res = m.evaluate(validationSet);
		System.out.println("Validation Error:," + res.error);
		System.out.println("Validation Accuracy:," + res.accuracy);

		res = m.evaluate(testSet);
		System.out.println("Test Error:," + res.error);
		System.out.println("Test Accuracy:," + res.accuracy);
	}
	
	/**
	 * ask the server for the updated model
	 * @return string form of all parameters
	 * @throws ClientProtocolException
	 * @throws IOException
	 */
	private void receiveFromServer(String serverPath, Model m) throws ClientProtocolException, IOException {
		CloseableHttpClient httpClient = HttpClients.createDefault();
		HttpPost uploadFile = new HttpPost(serverPath);
		MultipartEntityBuilder builder = MultipartEntityBuilder.create(); // http://mvnrepository.com/artifact/org.apache.httpcomponents/httpmime/4.3.1
		builder.addTextBody("clientType", "Eval", ContentType.TEXT_PLAIN);

		HttpEntity multipart = builder.build();
		uploadFile.setEntity(multipart);
		CloseableHttpResponse response = httpClient.execute(uploadFile);
		HttpEntity responseEntity = response.getEntity();

//		System.out.println("Post Status" + response.getStatusLine());
		// System.out.println(EntityUtils.toString(response.getEntity()));

		m.fetchParams(responseEntity.getContent());
	}
	
}
