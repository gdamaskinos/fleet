/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.lr;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import org.apache.commons.math3.util.Pair;
import org.jblas.DoubleMatrix;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import coreComponents.SGDUpdater;
import coreComponents.Sampler;
import utils.Helpers;

/**
 * Updater class implemented by the Service Provider
 * 
 */
public class LRUpdater implements SGDUpdater {

	private Kryo kryo;
	private Kryo kryo2;
	//private double rate;

	//private int featuresize;
	//private int numlabels;

	private int M;
	private int priority;


	private int size;

	ArrayList<LRModelParams> models;

	private ArrayList<Pair<LRGradients, Integer>> acc;

	private long startTime;
	private Random r;

	private double rate;

	public void update(InputStream input) {
		
		synchronized (models) {

		DoubleMatrix delta_weights;
		DoubleMatrix delta_biases;

		DoubleMatrix weights;
		DoubleMatrix biases;
		int epoch;
		double lrate = rate;
		
		Input in = new Input(input);
		epoch = kryo2.readObject(in, Integer.class);
		LRGradients gradient = kryo2.readObject(in, LRGradients.class);
		ArrayList<Pair<LRGradients, Integer>> aggregated; // M aggregated gradients for updating

		//delta_weights= (Tlabel.sub(predicted)).mmul(x).mul(rate)
		//weights = weights.add().sub(weights.mul(2*mu*rate));

		//nu = this.numerator / Math.pow(currEpoch + 1, this.pow);


		//delta_weights = gradients.delta_weights;
		//delta_biases = gradients.delta_biases;

		System.out.println("Read bytes: " + Helpers.humanReadableByteCount(in.total(), false));

		acc.add(new Pair<>(gradient, epoch));
		/* M-soft sync */
		if (acc.size() < M) {
			System.out.println("Gradients left to update: " + (M - acc.size()));
			return;
		}

		//System.out.println("Size: "+acc.size());
		//System.out.println("Acc: "+acc);
		//System.out.println("Priority: "+priority);
		// lock for concurrent computation and evaluation requests
		synchronized (models) {

			synchronized (acc){
				aggregated = stalenessSim(acc, size - 1);
				if (aggregated == null){
					// not possible to update with the available gradients
					//System.out.println("Here1");
					return;
				}
			}

			LRModelParams model = models.get(models.size() - 1).clone();

			weights = model.weights;
			biases = model.biases;

			//LRGradients g = average(aggregated,model.currEpoch);
			//LRGradients g = inverse(aggregated,model.currEpoch);
			LRGradients g = exp(aggregated,model.currEpoch);
			
//			if (lrate > rate / 2) // learning rate schedule
//				lrate = rate / Math.pow(1 + model.currEpoch, 0.1);
			
			weights.addi(g.delta_weights.mmuli(rate));
			biases.addi(g.delta_biases.mmul(rate));


			//System.out.println("Here2");
			//System.out.println("Epoch before: "+model.currEpoch);
			//System.out.println("Models count: "+models.size());            
			model.currEpoch++;
			//System.out.println("Epoch after: "+model.currEpoch);


			models.add(model);
			// remove older model
			if (models.size() > size) {
				System.out.println("Removing model as size = " + models.size());
				models.remove(0);
			}

			aggregated.clear();
		}
		}
	}

	/**
	 * Processes the accumulated gradients in such a way that staleness follows a predefined distribution.
	 * @param gradients list of accumulated gradients. The returned gradients are removed from this list.
	 * @param range range of possible staleness values to be drawn from the predefined distribution. Set to 0 for no staleness.
	 * @return ArrayList of {@link LRUpdater#M} gradients or null if not possible
	 */
	private ArrayList<Pair<LRGradients, Integer>> stalenessSim(ArrayList<Pair<LRGradients, Integer>> gradients, int range) {
		// process accumulated gradients
		String out;
		double mean, sigma=0;
		int currEpoch = models.get(models.size()-1).currEpoch;
		int pick, tau, minSt = Integer.MAX_VALUE, maxSt = -1;
		HashMap<Integer, Integer> tauMap = new HashMap<Integer, Integer>();
		Pair<LRGradients, Integer> g;

		out = "Current staleness values: [";
		for (int i=0; i<gradients.size(); i++) {
			g = gradients.get(i);
			tau = currEpoch - g.getSecond();
			if (tauMap.containsKey(tau))
				tauMap.put(tau, tauMap.get(tau) + 1);
			else
				tauMap.put(tau, 1);

			// discard gradient for being too old to be selected from gaussian
			if (tau > range) {
				gradients.remove(i);
				i--;
			}
			// check to update min, max available staleness
			if (tau > maxSt)
				maxSt = tau;
			if (tau < minSt)
				minSt = tau;
			out += tau + " ";
		}
		System.out.println(out + "]");


		// gaussian distribution 
		sigma = range / 6.0;
		mean = 3 * sigma;

		ArrayList<Pair<LRGradients, Integer>> res = new ArrayList<>();

		// if not enough model versions yet
		if (models.size() < this.size) { 
			// randomly pick M gradients
			for (int i=0; i<M; i++) {
				g = gradients.get((int) (Math.random() * gradients.size()));
				res.add(g);
				gradients.remove(g);
			}
			//System.out.println("HEREEE1111");
			priority = models.size() -1 + 1; // models size is going to be incremented
			System.out.println("Gradients"+res.get(0));
			return res;
		}

		// check if possible to get result <=?> if 'gradients' contains all possible staleness values at least M times
		for (int i=0; i<=range; i++) 
			if (!tauMap.containsKey(i) || tauMap.get(i) < M) {
				// missing possible staleness value || not enough occurrences
				priority = Math.max(0, models.size() - 1 - i); // on-demand
				//System.out.println("HEREEE2222");
				return null;				
			}

		System.out.printf("Drawing from gaussian(%f, %f)...\n", mean, sigma);

		//		pick gradients (drawn from Gaussian distribution)
		for (int i=0; i<M; i++) {
			pick = (int) Math.round(Math.max(0, r.nextGaussian() * sigma + mean));
			System.out.println("Picking staleness: " + pick);
			for (int j=0; j<gradients.size(); j++) {
				g = gradients.get(j);
				tau = currEpoch - g.getSecond();
				if (tau == pick) {
					res.add(g);
					gradients.remove(j);
					priority = Math.max(0, models.size() - 1 - tau); // on-demand
					break;
				}
			}
		}
		//System.out.println("HEREEE3333");
		return res;

	}


	/**
	 * Simple averaging policy
	 * @param gradients
	 * @param currEpoch current version of the model (to compute the staleness from)
	 * @return
	 */
	@SuppressWarnings("unused")
	private LRGradients average(ArrayList<Pair<LRGradients, Integer>> gradients, int currEpoch) {
		int tau;
		LRGradients res = gradients.get(0).getFirst();
		tau = currEpoch - gradients.get(0).getSecond();
		String out = "\tresponse: client|epoch|staleness|time:," + -1 + "," + currEpoch + "," + tau + 
				"," + (System.currentTimeMillis() - startTime) + "\n";
		System.out.print(out);

		
		for (int i = 1; i < gradients.size(); i++) {
			LRGradients g = gradients.get(i).getFirst();

			res.delta_weights.addi(g.delta_weights);
			res.delta_biases.addi(g.delta_biases);

			//weights.addi(delta_weights.mmuli(rate));
			//biases.addi(delta_biases.mmul(rate));
			tau = currEpoch - gradients.get(i).getSecond();
			out = "\tresponse: client|epoch|staleness|time:," + -1 + "," + currEpoch + "," + tau + 
					"," + (System.currentTimeMillis() - startTime) + "\n";
			System.out.print(out);
		}

		res.delta_weights.divi(gradients.size());
		res.delta_biases.divi(gradients.size());


		return res;
	}

	/**
	 * Inverse policy
	 * @param gradients
	 * @param currEpoch current version of the model (to compute the staleness from)
	 * @return
	 */
	@SuppressWarnings("unused")
	private LRGradients inverse(ArrayList<Pair<LRGradients, Integer>> gradients, int currEpoch) {
		int tau;
		double l_tau;
		LRGradients res = null;
		String out;
		
		for (int i = 0; i < gradients.size(); i++) {
			tau = currEpoch - gradients.get(i).getSecond();
			
			//if (tau < size / 5.0)
				l_tau = 1 / (double) (tau + 1);
			//else 
			//	l_tau = (size / 5.0) / (double) (tau + 1);
			
			if (i == 0) {
				res = gradients.get(i).getFirst();
				res.delta_weights.muli(l_tau);
				res.delta_biases.muli(l_tau);;
			}
			else {
				LRGradients g = gradients.get(i).getFirst();
				res.delta_weights.addi(g.delta_weights.muli(l_tau));
				res.delta_biases.addi(g.delta_biases.muli(l_tau));
			}
			
			out = "\tresponse: client|epoch|staleness|time:," + -1 + "," + currEpoch + "," + tau + 
					"," + (System.currentTimeMillis() - startTime) + "\n";
			System.out.print(out);
		}

		res.delta_weights.divi(gradients.size());
		res.delta_biases.divi(gradients.size());

		return res;
	}


	/**
	 * Exponential policy
	 * @param gradients
	 * @param currEpoch current version of the model (to compute the staleness from)
	 * @return
	 */
	@SuppressWarnings("unused")
	private LRGradients exp(ArrayList<Pair<LRGradients, Integer>> gradients, int currEpoch) {
		int tau;
		double l_tau;
		LRGradients res = null;
		String out;
		
		for (int i = 0; i < gradients.size(); i++) {
			tau = currEpoch - gradients.get(i).getSecond();
			
			l_tau = Math.exp(-0.5 * tau);
			
			if (i == 0) {
				res = gradients.get(i).getFirst();
				res.delta_weights.muli(l_tau);
				res.delta_biases.muli(l_tau);;
			}
			else {
				LRGradients g = gradients.get(i).getFirst();
				res.delta_weights.addi(g.delta_weights.muli(l_tau));
				res.delta_biases.addi(g.delta_biases.muli(l_tau));
			}
			
			out = "\tresponse: client|epoch|staleness|time:," + -1 + "," + currEpoch + "," + tau + 
					"," + (System.currentTimeMillis() - startTime) + "\n";
			System.out.print(out);
		}

		res.delta_weights.divi(gradients.size());
		res.delta_biases.divi(gradients.size());

		return res;
	}

	

	public void getParameters(Output output, boolean isComputationRequest) {
		// lock for concurrent computation and evaluation requests
		synchronized (models) {
			models.get(priority).trainTime = System.currentTimeMillis() - startTime;

			LRModelParams model = models.get(priority);
			//System.out.println("Epoch: "+model.currEpoch);
			//LRModelParams params = new LRModelParams(rate, featuresize, numlabels, weights, biases, currEpoch, System.currentTimeMillis() - startTime);
			kryo.writeObject(output, model);
		}
	}



	public void initialize(InputStream input, Sampler sampler) {
		// create serialized model
		kryo = new Kryo();
		kryo.register(LRModelParams.class);
		kryo2 = new Kryo();
		kryo2.register(LRModelParams.class);

		// Load the model
		acc = new ArrayList<>();
		models = new ArrayList<>();

		// staleness range will be = [0, size - 1] 
		size = 1;
		System.out.println("Staleness size: " + size);

		M = 1;
		System.out.println("M: " + M);

		Input in = new Input(input);
		LRModelParams restored = kryo.readObject(in, LRModelParams.class);
		rate = restored.rate;
		System.out.println("Learning rate: " + rate);
		
		//featuresize = restored.featuresize;
		//numlabels = restored.numlabels;
		//weights = restored.weights;
		//biases = restored.biases;

		models.add(new LRModelParams(restored.rate,restored.featuresize,restored.numlabels,restored.weights,restored.biases));
		//models.add(new LRCompleteModel(restored));
		priority = 0;

		//currEpoch = 0;
		models.get(0).currEpoch = 0;

		startTime = System.currentTimeMillis();
		r = new Random();
		// System.out.println("Received w1: " + w1);

		//weights = DoubleMatrix.rand(numlabels,featuresize);	
	}

}
