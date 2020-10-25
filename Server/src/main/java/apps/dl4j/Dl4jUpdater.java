/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.dl4j;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.CollectionSerializer;
import com.esotericsoftware.kryo.serializers.DefaultSerializers;
import com.esotericsoftware.kryo.serializers.DeflateSerializer;
import com.esotericsoftware.kryo.serializers.MapSerializer;

import coreComponents.SGDUpdater;
import coreComponents.Sampler;
import utils.Helpers;
import utils.Helpers.*;
import utils.dl4j.MultiLayerNetworkSerializer;
import utils.dl4j.MyDefaultGradient;
import utils.dl4j.MyMultiLayerNetwork;
import utils.dl4j.Nd4jSerializer;
import utils.StalenessSimulator;


public class Dl4jUpdater implements SGDUpdater {

	private Kryo kryoR, kryo1, kryo2, kryo3, kryo4, kryo5;
	/**
	 * Number of required gradients for each model update (M-softsync)
	 */
	private int M;
	/**
	 * Collected (gradients, epoch, id) since the last model update
	 */
	private ArrayList<Quadruple<Gradient, int[], Integer, Integer>> acc;
	
	private StalenessSimulator<Gradient> staleSim;

	private boolean firstInvoke;
	
	/**
	 * List of models for on-demand staleness updates
	 * stale[list.size() -1] (i.e. end of the list) is the most recent version of the model
	 * stale[list.size() -2] is the model before one update
	 * ...
	 */
	ArrayList<MyMultiLayerNetwork> models;
	/**  
	 * Maximum size for {@link Dl4jUpdater#models}
	 */
	private int size;
	/**
	 * determines the version of the model that is going to be sent to the next request
	 */
	private int priority;
	
	/**
	 * Learning model hashCode
	 */
	private int hashCode;
	
	ArrayList<Double> lrates;
	
	BufferedWriter bw;
	private long startTime;
	
	private MapSerializer mapser;
	
	@SuppressWarnings("unchecked")
	@Override
	public void initialize(InputStream input, Sampler sampler) {

		System.out.println("Checking nd4j...");
		INDArray arr1 = Nd4j.create(new float[] { 1, 2, 3, 4 }, new int[] { 2, 2 });
		System.out.println(arr1);
		System.out.println("Check OK");
		
		// create serialized model
		kryoR = new Kryo();
		kryo1 = new Kryo();
		kryo1.register(MyMultiLayerNetwork.class, new MultiLayerNetworkSerializer());

		kryo2 = new Kryo();
		kryo2.register(Dl4jExtraParams.class);

		kryo3 = new Kryo();
		DeflateSerializer deflser1 = new DeflateSerializer(new DefaultSerializers.StringSerializer());
		deflser1.setCompressionLevel(9);
		kryo3.register(String.class, deflser1);

		kryo4 = new Kryo();
		DeflateSerializer deflser2 = new DeflateSerializer(new Nd4jSerializer());
		deflser2.setCompressionLevel(9);
		kryo4.register(NDArray.class, deflser2);
//		kryo4.register(NDArray.class, new Nd4jSerializer());

		mapser = new MapSerializer();
        mapser.setValueClass(NDArray.class, new Nd4jSerializer());
        
		kryo5 = new Kryo();
		kryo5.register(HashMap.class, new MapSerializer());
		
		Input in = new Input(input);
		
		// load learning rate schedule
		lrates = kryoR.readObject(in, ArrayList.class, new CollectionSerializer());
		System.out.println("Initial learning rate: " + lrates.get(0));

		// load staleness size for simulator
		size = kryoR.readObject(in, Integer.class);
		acc = new ArrayList<>();
		models = new ArrayList<>();
		System.out.println("Staleness size: " + size);
		
		// load M-softsync param
		M = kryoR.readObject(in, Integer.class);
		System.out.println("M: " + M);
		// load model
		MultiLayerNetwork restored = kryo1.readObject(in, MyMultiLayerNetwork.class);
		
		models.add(new MyMultiLayerNetwork(restored.getLayerWiseConfigurations(), restored.params()));
		priority = 0;
		hashCode = models.get(0).hashCode();
		
		models.get(0).currEpoch = 0;
		startTime = System.currentTimeMillis();
		// System.out.println(model.params());
		
		String path = System.getProperty("user.home") + "/ServerOut";
		System.out.println("Logging to: " + path);
		FileOutputStream fos = null;
		try {
			fos = new FileOutputStream(path);
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		
		staleSim = new StalenessSimulator<>();
		bw = new BufferedWriter(new OutputStreamWriter(fos));
		
 		firstInvoke = true; // for printing schedule
	}

	@Override
	public void getParameters(Output output, boolean isComputationRequest) {


		// lock for concurrent computation and evaluation requests
		synchronized (models) {
			// ! If send model first => buffer overflow
			// write additional params
			System.out.println("Sending priority: " + priority);
			MyMultiLayerNetwork model = models.get(priority);
			
			model.trainTime = System.currentTimeMillis() - startTime;

			kryo2.writeObject(output, new Dl4jExtraParams(model.trainTime, model.currEpoch, hashCode));
			// write model configuration
			kryo3.writeObject(output, model.getLayerWiseConfigurations().toJson());
			// write model params
		//	Nd4j.getCompressor().setDefaultCompression("FLOAT16");
		//	kryo4.writeObject(output, Nd4j.getCompressor().compress(model.params()));
			kryo4.writeObject(output, model.params());
			// System.out.println("Sent: " + model.params());
		}
		
		//for (int i=0; i<2; i++)
		//	update(null);
		
	}

	@Override
	public void update(InputStream input) {
		// lock for concurrent computation and evaluation requests
		synchronized (models) {
			
		Input in = new Input(input);
		ArrayList<Quadruple<Gradient, int[], Integer, Integer>> aggregated; // M aggregated gradients for updating
		
		// Dl4jGradient gradient = kryo2.readObject(in, Dl4jGradient.class);
		int hashCode = kryoR.readObject(in, Integer.class);
		int epoch = kryoR.readObject(in, Integer.class);
		System.out.println("Received epoch: " + epoch);

		if (hashCode != this.hashCode) {
			System.out.println("Gradients refer to different model. Dropping...");
			return;
		}

		// Get gradients in as a flattened NDArray
		//NDArray gradient = kryoR.readObject(in, NDArray.class, new Nd4jSerializer());
		
		// Get gradients as a HashMap (String -> NDArray)
		HashMap<String, NDArray> gradientsMap = kryo5.readObject(in, HashMap.class, mapser);
		
		// Get the ordered gradient key list
		ArrayList<String> orderedKeys = kryoR.readObject(in, ArrayList.class, new CollectionSerializer());
		
		// Get flattening info as a HashMap(String -> Character)
		HashMap<String, Character> flattenInfoMap = kryo5.readObject(in, HashMap.class);
		
		Gradient g = new MyDefaultGradient(); 
		for (String key : orderedKeys)
			g.setGradientFor(key, gradientsMap.get(key), flattenInfoMap.get(key)); // IMPORTANT: put order determines flattenedGradient 
		
		g.gradient(); // call to trigger the flattened gradient creation
		//NDArray gradient = (NDArray) g.gradient();
		acc.add(new Quadruple<>(g, new int[0], epoch, -1));

		System.out.println("Read bytes: " + Helpers.humanReadableByteCount(in.total(), false));
		/* M-soft sync */
		if (acc.size() < M) {
			System.out.println("Gradients left to update: " + (M - acc.size()));
			return;
		}

				// initialize lrate
//				double lrate;
//				for (int l = 0; l < model.getnLayers(); l++) {
//					for (String key : model.getLayer(l).conf().getLearningRateByParam().keySet()) {
//						// Reset batch normalization learning rate
//						if (key.equals("gamma") || (key.equals("beta"))) {
//							model.getLayer(l).conf().setLearningRateByParam(key, 1);
//						}
//						lrate = model.getLayer(l).conf().getLearningRateByParam(key);
//						//System.out.println("Current Learning rate for " + key + ": " + lrate);
//					}
//				}

				synchronized (acc) {
					//aggregated = acc; // FIXME evaluation priority issues
					//size = 1; 
					// 	simulate staleness
					Tuple<Integer, ArrayList<Quadruple<Gradient, int[], Integer, Integer>>> temp = staleSim.stalenessSim(
							acc, models.get(models.size()-1).currEpoch, size-1, size-1, models.size(), M, -1, -1);
					aggregated = temp.getSecond();
					priority = temp.getFirst();	
					if (aggregated == null) // not possible to update with the available gradients
						return;
				}
				System.out.println("Updating...");
				// update latest model
				
				MyMultiLayerNetwork model = models.get(models.size() - 1).clone();
				System.out.println("Model version: " + model.currEpoch);
				//System.out.println(aggregated.get(0).getFirst().toString());
				model.applyGradients(average(aggregated, model.currEpoch));
				//model.applyGradients(inverse(aggregated, model.currEpoch));
				//model.applyGradients(exp(aggregated, model.currEpoch));
				//model.applyGradients(custom(aggregated, model.currEpoch));
				
				//model.applyGradients(g);
				model.currEpoch++;
				
				if (model.currEpoch == lrates.size())
					System.out.println("Iterations reached. Keeping static learning rate.");
				// manually update learning rate
				if (model.currEpoch < lrates.size())
					for (int l = 0; l < model.getnLayers(); l++) {
						for (String key : model.getLayer(l).conf().getLearningRateByParam().keySet()) {
							//lrate = model.getLayer(l).conf().getLearningRateByParam(key);
							// learning rate update
							if (!((key.equals("gamma") || (key.equals("beta"))))) {
								// System.out.println("New Learning rate for " +
								// key + ": " + lrate);
								model.getLayer(l).conf().setLearningRateByParam(key, lrates.get(model.currEpoch));
							}
						}
					}
//                
//                models.clear();
//                models.add(model);
//				
				models.add(model);
				// remove older model
				if (models.size() > size) {
					System.out.println("Removing model as size = " + models.size());
					models.remove(0);
				}
				
//				String out = "Epochs: ";
//				for (int i=0; i<models.size(); i++)
//					out += models.get(i).currEpoch + " ";
//				System.out.println(out);
				
				// flush accumulator in case simulation is not used 
				aggregated.clear();
		}
	}

	/**
	 * Custom decay policy of OP-SGD (theoretically faster than inverse)
	 * @param gradients
	 * @param currEpoch current version of the model (to compute the staleness from)
	 * @return
	 */
	private Gradient custom(ArrayList<Quadruple<Gradient, int[], Integer, Integer>> gradients, int currEpoch) {
		if (firstInvoke) {
			System.out.println("schedule: custom");
			firstInvoke = false;
		}
		Gradient res = new MyDefaultGradient();
		int tau;
		double l_tau;

		for (int i=0; i<gradients.size(); i++) {
			Gradient g = gradients.get(i).getFirst();
			// get \lambda(\tau_{km})
			tau = currEpoch - gradients.get(i).getThird();
			String out = "\tresponse: client|epoch|staleness|time:," + -1 + "," + currEpoch + "," + tau + 
					"," + (System.currentTimeMillis() - startTime) + "\n";
			System.out.print(out);
			try {
				bw.write(out);
				bw.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
			l_tau = Math.exp(-0.685 * Math.pow(tau, 1/1.85));
			
			if (i == 0)
				for (String key : g.gradientForVariable().keySet())
					res.setGradientFor(key, g.getGradientFor(key).muli(l_tau), g.flatteningOrderForVariable(key));
			else
				for (String key : g.gradientForVariable().keySet())
					res.getGradientFor(key).addi(g.getGradientFor(key).muli(l_tau));
		}
		
		for (String key : res.gradientForVariable().keySet())
			res.getGradientFor(key).divi(gradients.size());
		
		return res;
	}
	
	/**
	 * Exponential decay policy of OP-SGD
	 * @param gradients
	 * @param currEpoch current version of the model (to compute the staleness from)
	 * @return
	 */
	private Gradient exp(ArrayList<Quadruple<Gradient, int[], Integer, Integer>> gradients, int currEpoch) {
		double a = 1.475;
		if (firstInvoke) {
			System.out.println("schedule: exp, a=" + a);
			firstInvoke = false;
		}
		Gradient res = new MyDefaultGradient();
		int tau;
		double l_tau;

		for (int i=0; i<gradients.size(); i++) {
			Gradient g = gradients.get(i).getFirst();
			// get \lambda(\tau_{km})
			tau = currEpoch - gradients.get(i).getThird();
			String out = "\tresponse: client|epoch|staleness|time:," + -1 + "," + currEpoch + "," + tau + 
					"," + (System.currentTimeMillis() - startTime) + "\n";
			System.out.print(out);
			try {
				bw.write(out);
				bw.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
			l_tau = Math.exp(-a * Math.pow(tau, 0.2));
			//l_tau = Math.exp(-a * tau);
			
			if (i == 0)
				for (String key : g.gradientForVariable().keySet())
					res.setGradientFor(key, g.getGradientFor(key).muli(l_tau), g.flatteningOrderForVariable(key));
			else
				for (String key : g.gradientForVariable().keySet())
					res.getGradientFor(key).addi(g.getGradientFor(key).muli(l_tau));
		}
		
		for (String key : res.gradientForVariable().keySet())
			res.getGradientFor(key).divi(gradients.size());
		
		return res;
	}

	/**
	 * Inverse policy (http://www.ijcai.org/Proceedings/16/Papers/335.pdf)
	 * @param gradients
	 * @param currEpoch current version of the model (to compute the staleness from)
	 * @return
	 */
	private Gradient inverse(ArrayList<Quadruple<Gradient, int[], Integer, Integer>> gradients, int currEpoch) {
		if (firstInvoke) {
			System.out.println("schedule: inverse");
			firstInvoke = false;
		}
		Gradient res = new MyDefaultGradient();
		int tau;
		double l_tau;

		for (int i=0; i<gradients.size(); i++) {
			Gradient g = gradients.get(i).getFirst();
			// get \lambda(\tau_{km})
			tau = currEpoch - gradients.get(i).getThird();
			String out = "\tresponse: client|epoch|staleness|time:," + -1 + "," + currEpoch + "," + tau + 
					"," + (System.currentTimeMillis() - startTime) + "\n";
			System.out.print(out);
			try {
				bw.write(out);
				bw.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
			l_tau = 1 / (double) (tau + 1);
			
			if (i == 0)
				for (String key : g.gradientForVariable().keySet())
					res.setGradientFor(key, g.getGradientFor(key).muli(l_tau), g.flatteningOrderForVariable(key));
			else
				for (String key : g.gradientForVariable().keySet())
					res.getGradientFor(key).addi(g.getGradientFor(key).muli(l_tau));
		}
		
		for (String key : res.gradientForVariable().keySet())
			res.getGradientFor(key).divi(gradients.size());
		
		return res;
	}
	
	/**
	 * Simple averaging policy
	 * @param gradients
	 * @param currEpoch current version of the model (to compute the staleness from)
	 * @return
	 */
	private Gradient average(ArrayList<Quadruple<Gradient, int[], Integer, Integer>> gradients, int currEpoch) {
		if (firstInvoke) {
			System.out.println("schedule: average");
			firstInvoke = false;
		}
		Gradient res = new MyDefaultGradient();
		int tau;

		for (int i = 0; i < gradients.size(); i++) {
			Gradient g = gradients.get(i).getFirst();
			if (i == 0)
				for (String key : g.gradientForVariable().keySet())
					res.setGradientFor(key, g.getGradientFor(key), g.flatteningOrderForVariable(key));
			else
				for (String key : g.gradientForVariable().keySet())
					res.getGradientFor(key).addi(g.getGradientFor(key)) ;
			
			tau = currEpoch - gradients.get(i).getThird();
			String out = "\tresponse: client|epoch|staleness|time:," + -1 + "," + currEpoch + "," + tau + 
					"," + (System.currentTimeMillis() - startTime) + "\n";
			System.out.print(out);
			try {
				bw.write(out);
				bw.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
			

		}
		for (String key : res.gradientForVariable().keySet())
			res.getGradientFor(key).divi(gradients.size());
		
		res.gradient(); // call to trigger the flattened gradient creation
		
		return res;
	}

}
