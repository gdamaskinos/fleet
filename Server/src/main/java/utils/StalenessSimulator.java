/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import utils.Helpers.*;

public class StalenessSimulator<Tgrad> {
	private Random r;
	
	public StalenessSimulator() {
		r = new Random();
	}

	
	/**
	 * Processes the accumulated gradients in such a way that staleness follows a predefined distribution.
	 * @param gradients list of accumulated gradients (deltas, class dist, epoch, client id). The returned gradients are removed from this list.
	 * @param range possible staleness values to be drawn from the predefined distribution are [0, range]. Set to 0 for no staleness.
	 * @param maxRange useful only for having a fix number of cold-start (i.e., \tau=0) updates; for example for measuring impact of different staleness distributions; set maxRange=range by default; else maxRange >= range.
	 * @param modelsSize size of stored models
	 * @param M M-softasync param
	 * @param outlierClass gradients computed on this class will be given staleness = outlierStaleness. Set to -1 to deactivate.
	 * @param outlierStaleness set to -1 to deactivate
	 * @return [Priority, ArrayList of M gradients or null if not possible] 
	 * Priority: next model id \in [0,range] models to be sent
	 */
	public Tuple<Integer, ArrayList<Quadruple<Tgrad, int[], Integer, Integer>>> stalenessSim(ArrayList<Quadruple<Tgrad, int[], Integer, Integer>> gradients, int currEpoch, int range, int maxRange, int modelsSize, int M, int outlierClass, int outlierStaleness) {
		// process accumulated gradients
		String out;
		double mean, sigma=0;
		int priority=-1, pick, tau, minSt = Integer.MAX_VALUE, maxSt = -1;
		HashMap<Integer, Integer> tauMap = new HashMap<Integer, Integer>();
		Quadruple<Tgrad, int[], Integer, Integer> g;
		int[] classDist;
		boolean isOutlier;
		
		ArrayList<Quadruple<Tgrad, int[], Integer, Integer>> res = new ArrayList<>();
		ArrayList<Quadruple<Tgrad, int[], Integer, Integer>> outlierGradients = new ArrayList<>();
		
		out = "<Staleness-isOutlier> values for accumulated gradients: [";
		for (int i=0; i<gradients.size(); i++) {
			g = gradients.get(i);
			tau = currEpoch - g.getThird();
			classDist = g.getSecond();
			
			// handle outlier gradients
			isOutlier = false;
			for (int j=0; j<classDist.length; j++)
				if (classDist[j] > 0 && j == outlierClass) {
					isOutlier = true;
					if (tau >= outlierStaleness) {
						System.out.println("Picking staleness: " + tau);
						res.add(g);
						M--;
						priority=0;
					} else
						// store it and push it back before returning
						outlierGradients.add(g);
					gradients.remove(i);
					i--;
					break;
				}
			if (isOutlier) {
				out += tau + "-T ";
				continue;
			}
			
			if (tauMap.containsKey(tau))
				tauMap.put(tau, tauMap.get(tau) + 1);
			else
				tauMap.put(tau, 1);
					
			// discard gradient for being too old to be selected from gaussian
			if (tau > maxRange) {
				gradients.remove(i);
				i--;
			}
			// check to update min, max available staleness
			if (tau > maxSt)
				maxSt = tau;
			if (tau < minSt)
				minSt = tau;
			out += tau + "-F ";

		}
		System.out.println(out + "]");

		// check if gradient of outlierClass exists with outlierStaleness and return it
		
		
		// gaussian distribution 
		sigma = range / 6.0;
		mean = 3 * sigma;
				
		// if not enough model versions yet
		if (modelsSize < maxRange+1) { 
			// randomly pick M gradients
			if (gradients.size() < M) {
				System.out.println("Not enough model versions yet. Not enough gradients to pick (some are outliers).");
				priority = modelsSize - 1; // models size is not going to be incremented
				// push back outlier gradients and return
				for (int j=0; j<outlierGradients.size(); j++)
					gradients.add(outlierGradients.get(j));
				for (int j=0; j<res.size(); j++)
					gradients.add(res.get(j));
				return new Tuple<Integer, ArrayList<Quadruple<Tgrad, int[], Integer, Integer>>>(priority, null);
			}
			for (int i=0; i<M; i++) {
				g = gradients.get((int) (Math.random() * gradients.size()));
				res.add(g);
				gradients.remove(g);
			}
			priority = modelsSize - 1 + 1; // models size is going to be incremented
			System.out.println("Not enough model versions yet.\nPicking staleness: 0");
					
			// push back outlier gradients and return
			for (int j=0; j<outlierGradients.size(); j++)
				gradients.add(outlierGradients.get(j));
			return new Tuple<Integer, ArrayList<Quadruple<Tgrad, int[], Integer, Integer>>>(priority, res);
		}
		
		// TODO optimization: continuously check and pick till return 
		// check if possible to get result <=> if 'gradients' contains all possible staleness values at least M times
		for (int i=0; i<=range; i++) 
			if (!tauMap.containsKey(i) || tauMap.get(i) < M) {
				// missing possible staleness value || not enough occurrences
				priority = Math.max(0, modelsSize - 1 - i); // on-demand
				System.out.println("Missing staleness value: " + i);
				
				// push back outlier gradients and return
				for (int j=0; j<outlierGradients.size(); j++)
					gradients.add(outlierGradients.get(j));
				for (int j=0; j<res.size(); j++)
					gradients.add(res.get(j));
				return new Tuple<Integer, ArrayList<Quadruple<Tgrad, int[], Integer, Integer>>>(priority, null);
			}
		
		
		//	pick gradients (drawn from Gaussian distribution)
		if (M > 0)
			System.out.printf("Drawing from gaussian(%f, %f)...\n", mean, sigma);
		for (int i=0; i<M; i++) {
			pick = (int) Math.round(Math.max(0, r.nextGaussian() * sigma + mean));
			if (pick > maxSt) {// avoid out-of-range due to rounding
				System.out.println("Rounding: " + pick);
				pick = maxSt;
			}
			System.out.println("Picking staleness: " + pick);
			for (int j=0; j<gradients.size(); j++) {
				g = gradients.get(j);
				tau = currEpoch - g.getThird();
				if (tau == pick) {
					res.add(g);
					gradients.remove(j);
					priority = Math.max(0, modelsSize - 1 - tau); // on-demand
					break;
				}
			}
		}
		
		// push back outlier gradients and return
		for (int j=0; j<outlierGradients.size(); j++)
			gradients.add(outlierGradients.get(j));
		return new Tuple<Integer, ArrayList<Quadruple<Tgrad, int[], Integer, Integer>>>(priority, res);
	}
}
