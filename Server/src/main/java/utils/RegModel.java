/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import java.util.Arrays;

import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.RegressionResults;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.commons.math3.stat.regression.UpdatingMultipleLinearRegression;
import org.apache.commons.math3.util.FastMath;
import org.jblas.DoubleMatrix;

import smile.regression.LASSO;

public class RegModel {

	private double[][] features;
	private double[] targets;
	
	//private double[] means;
	//sprivate double[] vars;
	
	private double[] betas;
	//private RegressionResults regressResults;
	
	//private OLSMultipleLinearRegression regression;
	//private UpdatingMultipleLinearRegression regression;
	private LASSO regression;
	private LASSO.Trainer trainer;
	private double lasso_lambda=0.01;
	
	public RegModel(double[][] features, double[] targets){
		
		//this.means = new double[features[0].length];
		//this.vars = new double[features[0].length];
		
		
		this.features = features;
		//this.features = normalize(features);
		this.targets = targets;
		
		//this.regression = new OLSMultipleLinearRegression();
		//this.regression.setNoIntercept(true);
		
		this.trainer = new LASSO.Trainer(lasso_lambda);//new LASSO(features, targets);
		
//		System.out.println("Features");
//		for(int ii=0;ii<targets.length;ii++){
//			System.out.println("feat: "+Arrays.toString(features[ii])+" target: "+targets[ii]);
//		}
		
		trainModel();
		System.out.println("Model trained!");
	}
	
	public LASSO getModel(){
		return this.regression;
	}
	
	private void trainModel(){
		
		// singular matrix exception can occurs
		//this.regression.newSampleData(targets, features);
		//this.betas = regression.estimateRegressionParameters();
		this.regression = trainer.train(features,targets);
		//System.out.println(Arrays.toString(targets));
		this.betas = this.regression.coefficients();
	}

	private double[][] normalize(double[][] matrix) {
		
		DoubleMatrix mat = new DoubleMatrix(matrix);
		
		for(int i=0; i<mat.getColumns(); i++){
			DoubleMatrix column = mat.getColumn(i);
			double[] data = column.transpose().toArray();
			
			double mean = StatUtils.mean(data);
			double var = FastMath.sqrt(StatUtils.variance(data));
			
			DoubleMatrix norColumn = (column.sub(mean)).div(var);
			
			//this.means[i] = mean;
			//this.vars[i] = var;
			
			mat.putColumn(i, norColumn);
			
		}
		
		return mat.toArray2();
	}
}
