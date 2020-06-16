/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import java.util.List;
import java.util.Set;
import java.util.ArrayList;
import java.util.HashSet;

import jsat.linear.DenseVector;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.CategoricalData;
import jsat.regression.RegressionDataSet;
import jsat.classifiers.linear.PassiveAggressive;
import jsat.classifiers.linear.PassiveAggressive.Mode;


public class PAModel {
	
	private final int WINDOW_SIZE = 10;
	
	// Hyperparameters
	private PassiveAggressive.Mode mode;
	private final boolean hp_tuning;
	private final double default_C;
	private double profilerEpoch;
	private double epsilon;
	private int epochs;
	private double C;
	
	// Passive-Aggressive linear regression model
	private PassiveAggressive model;

	// Data
	private RegressionDataSet dataset;
	private Set<Integer> login_events;

	// Internal state
	private boolean setup = false;

	public PAModel(int epochs, double epsilon, double C, PassiveAggressive.Mode mode, boolean hp) {
		this.login_events = new HashSet<>();
		this.profilerEpoch = 0;
		this.epsilon = epsilon;
		this.epochs = epochs;
		this.hp_tuning = hp;
		this.default_C = C;
		this.mode = mode;

		
		model = new PassiveAggressive(this.epochs, this.mode);
		model.setEps(this.epsilon);
		model.setEpochs(epochs);
		if (mode.equals(Mode.PA1) || mode.equals(Mode.PA2)) model.setC(C);
		
		System.err.println("Mode: " + model.getMode() + ", Epsilon: " + model.getEps() + " , C_value: " + model.getC());

	}
	
	public PAModel(PassiveAggressive.Mode mode) {
		this(10, 0.01, 0.0001, mode, true);
	}

	public PAModel() {
		this(10, 0.01, 0.0001, PassiveAggressive.Mode.PA, true);
	}

	public PassiveAggressive.Mode getMode() {
		return this.mode;
	}
	
	public boolean isUp() {
		return this.setup;
	}
	
	/**
	 * Initialize the model with the first client query when no initial dataset is used
	 * 
	 * @param features
	 * @param targets
	 */
	public void initOnTheFly(double[][] features, double[] targets) {
		// If model has already been ran or dataset is inconsistent
		if (setup || features.length < 1 || features.length != targets.length|| features[0].length < 1) {
			System.out.println("Failed to train profiler: model has already been ran or the input dataset is inconsistent.");
			return;
		}

		dataset = buildDataSet(features, targets);
		setup = true;
	}
	
	/**
	 * Initialize the model when initial dataset is given
	 * 
	 * @param features
	 * @param targets
	 */
	public void initialize(double[][] features, double[] targets) {
		// If model has already been ran or dataset is inconsistent
		if (setup || features.length < 1 || features.length != targets.length|| features[0].length < 1) {
			System.out.println("Failed to train profiler: model has already been ran or the input dataset is inconsistent.");
			return;
		}

		dataset = buildDataSet(features, targets);
		model.train(dataset);
		setup = true;
	}

	public void update(double[] features, double target) {
		model.update(buildDataPoint(features), target);
	}

	public double predict(double[] features) {
		return model.regress(buildDataPoint(features));
	}
	
	/**
	 * Reverses the regression model for the given feature: 
	 * targetPrediction = W.dot(features) =>
	 * features[index] = (targetPrediction - w[0]*features[0] - ...) / w[index]
	 * @param targetPrediction
	 * @param features
	 * @param index
	 * @return
	 */
	public double reverse(double targetPrediction, double[] features, int index) {
		double numer = targetPrediction;
		double denom = model.getRawWeight().get(index);
		
		for (int i=0; i<features.length; i++)
			if (i != index)
				numer -= model.getRawWeight().get(i) * features[i];
		
		return (double) numer / denom;
	}
	
	/**
	 * Tunes aggressiveness parameter C when multiple logins occur simultaneously
	 * @deprecated C value tuning was not performed correctly
	 * 
	 * @param epoch
	 */
	public void newLoginAtEpoch(int epoch) {
		if (hp_tuning) {
			login_events.add(epoch);
			int last_epoch = (int) Math.max(epoch, profilerEpoch);
			//test code
//			setC(Math.max(1.0, login_events.stream().filter(e -> last_epoch - WINDOW_SIZE < e).count()) * default_C);
//			model.setC(default_C);
		}
	}
	
	public void setC(double C) {
		if (0 < C && Double.isFinite(C) && hp_tuning) {
//			System.out.println(mode + " switching aggressiveness to " + C);
//			model.setC(C);
			//test code
			System.out.println(mode + " switching aggressiveness to " + default_C);
//			model.setC(default_C);
		}
	}
	
	/**
	 * Tunes aggressiveness parameter C on each epoch
	 * @deprecated C_value tuning was not performed correctly
	 * 
	 * @param epoch
	 */
	public void tick(int epch) {
		profilerEpoch = epch;
//		if (hp_tuning) setC(Math.max(1.0, login_events.stream().filter(e -> profilerEpoch - WINDOW_SIZE < e).count()) * default_C);
	}
	
	public double getC() { return C; }

	private RegressionDataSet buildDataSet(double[][] features, double[] targets) {
		List<DataPointPair<Double>> dataList = new ArrayList<>();
		model.setUp(new CategoricalData[] {}, features[0].length);
		for (int i = 0; i < features.length; ++i) {
			DataPointPair<Double> dp = new DataPointPair<Double>(buildDataPoint(features[i]), targets[i]);
			dataList.add(dp);
		}

		return new RegressionDataSet(dataList);
	}

	private DataPoint buildDataPoint(double[] features) {
		return new DataPoint(new DenseVector(features));
	}
}
