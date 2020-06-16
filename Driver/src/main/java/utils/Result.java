/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

public class Result {

	public double error;
	public double accuracy;
	
	public Result(double error, double accuracy){
		this.error = error;
		this.accuracy = accuracy;
	}
}
