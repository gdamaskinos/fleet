/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils.dl4j;
import java.io.IOException;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

public class MultiLayerNetworkSerializer extends Serializer<Model>{

	@Override
	public void write(Kryo kryo, Output output, Model model) {
		try {
			ModelSerializer.writeModel(model, output, false);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public Model read(Kryo kryo, Input input, Class<Model> type) {
		try {
			return ModelSerializer.restoreMultiLayerNetwork(input, false);
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	
}
