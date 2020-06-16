/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;

/**
 * Various helpers
 *
 */
public class Helpers {
    public static String humanReadableByteCount(long bytes, boolean si) {
        int unit = si ? 1000 : 1024;
        if (bytes < unit) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(unit));
        String pre = (si ? "kMGTPE" : "KMGTPE").charAt(exp-1) + (si ? "" : "i");
        return String.format("%.1f %sB", bytes / Math.pow(unit, exp), pre);
    }

//    public static double round(double origin, double thres) {
//        return FastMath.round(origin + 0.5f - thres);
//    }
//
//
//    public static float round(float origin, double thres) {
//        return FastMath.round(origin + 0.5f - thres);
//    }
    
    
    public static double avg(ArrayList<Double> list) {
    	  double sum = 0;
    	  if(!list.isEmpty()) {
    		Iterator<Double> it = list.iterator();
    		while (it.hasNext()) 
    			sum += it.next();
    	    return sum / (double) list.size();
    	  }
    	  return sum;
    	}
    
    public static double percentile(ArrayList<Double> list, double p) {
    		Collections.sort(list);
		double[] target = new double[list.size()];
		for (int i = 0; i < target.length; i++) 
		    target[i] = list.get(i);	
		
		return (new Percentile()).evaluate(target, p);
    }
    
    public static ArrayList<Double> toDouble(ArrayList<Integer> list) {
    	ArrayList<Double> temp = new ArrayList<>();
    	for (int val : list)
    		temp.add((double) val);
    	return temp;
    }
    
    public static class Tuple<X, Y> { 
    	  public X x; 
    	  public Y y; 
    	  public Tuple(X x, Y y) { 
    	    this.x = x; 
    	    this.y = y; 
    	  } 
    	  public X getFirst() {
    		  return x;
    	  }
    	  public Y getSecond() {
    		  return y;
    	  }
    	} 
    
    public static class Triple<X, Y, Z> { 
  	  public X x; 
  	  public Y y;
  	  public Z z;
  	  public Triple(X x, Y y, Z z) { 
  	    this.x = x; 
  	    this.y = y; 
  	    this.z = z;
  	  } 
  	  public X getFirst() {
  		  return x;
  	  }
  	  public Y getSecond() {
  		  return y;
  	  }
  	  public Z getThird() {
  		  return z;
  	  }
  	} 
    
    public static class Quadruple<X, Y, Z, W> { 
    	  public X x; 
    	  public Y y;
    	  public Z z;
    	  public W w;
    	  public Quadruple(X x, Y y, Z z, W w) { 
    	    this.x = x; 
    	    this.y = y; 
    	    this.z = z;
    	    this.w = w;
    	  } 
    	  public X getFirst() {
    		  return x;
    	  }
    	  public Y getSecond() {
    		  return y;
    	  }
    	  public Z getThird() {
    		  return z;
    	  }
    	  public W getFourth() {
    		  return w;
    	  }
    	} 
    
    public static double[] normalizeVec(int[] v){
        int s = 0;
        for (int i=0; i<v.length; i++)
            s += v[i];
        double[] ret = new double[v.length];

        for (int i=0; i<v.length; i++) 
            ret[i] = (double) v[i] / s;

        return ret;
    }

    public static double similarity(int[] x, int[] y) {
        double[] normx = normalizeVec(x);
        double[] normy = normalizeVec(y);
        
        return bhattacharyya_coefficient(normx, normy);
    }

    /**
     * Bhattacharyya distance between two normalized histograms (https://github.com/DiegoCatalano/Catalano-Framework/blob/master/Catalano.Math/src/Catalano/Math/Distances/Distance.java#L62)
     * @param histogram1 Normalized histogram.
     * @param histogram2 Normalized histogram.
     * @return The Bhattacharyya distance between the two histograms.
     */
    public static double bhattacharyya_coefficient(double[] histogram1, double[] histogram2){
        int bins = histogram1.length; // histogram bins
        double b = 0; // Bhattacharyya's coefficient

        for (int i = 0; i < bins; i++)
            b += Math.sqrt(histogram1[i]) * Math.sqrt(histogram2[i]);

        return b;
    }
}
