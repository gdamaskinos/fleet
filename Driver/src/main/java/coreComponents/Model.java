/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;
import java.io.InputStream;

import utils.MyDataset;
import utils.Result;
/**
 * Interface for single neural network model
 * 
 */
public interface Model {
	/**
	 * Initializes the required parameters for the model.
	 */
	public void initialize();

	/**
	 * Evaluates the model on the given dataset
	 */
	public Result evaluate(MyDataset set);

	public void printParams();
	
	public int getcurrEpoch();
	
	/**
	 * Gets total time (in milliseconds) that the model has been trained till the last update.
	 * @return
	 */
	public long getTrainTime();

	/**
	 * Fetch the model from the serialized representation sent by the Server
	 * @param result
	 */
	public void fetchParams(InputStream input);
	
	/**
	 * Predict t-values for the test set.
	 * 
	 * @return 2D (single column) matrix with predicted classes (0 or 1) for
	 *         each test x_input
	 */
	public int[][] predict();

	/**
	 * Saves the current state of the model for early stopping.
	 */
	public void saveState();

	/**
	 * Restore the saved state for future evaluation.
	 */
	public void restoreState();
	
	/**
	 * Stop any activities
	 */
	public void cleanUp();

	/**
	 * Get the parameters for the initialization of the Server
	 * @return
	 */
	public byte[] getParams();
}
