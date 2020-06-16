/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.dl4j;

public class Dl4jExtraParams {

	public Dl4jExtraParams() {
		
	}
	public Dl4jExtraParams(long trainTime, int currEpoch, int hashCode) {
		this.trainTime = trainTime;
		this.currEpoch = currEpoch;
		this.hashCode = hashCode;
	}
	public long trainTime;
	public int currEpoch;
	public int hashCode;
}
