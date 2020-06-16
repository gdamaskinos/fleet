/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.dl4j;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.ArrayList;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.CollectionSerializer;
import com.esotericsoftware.kryo.serializers.DefaultSerializers;
import com.esotericsoftware.kryo.serializers.DeflateSerializer;

import coreComponents.Model;
import utils.Helpers;
import utils.MyDataset;
import utils.Result;
import utils.dl4j.MultiLayerNetworkSerializer;
import utils.dl4j.MyDl4jDataset;
import utils.dl4j.MyMultiLayerNetwork;
import utils.dl4j.Nd4jSerializer;

/**
 * Dl4j based model used for Single Label classification (e.g. mnist, spambase)
 *
 */
public class Dl4jModel implements Model {

	public MyMultiLayerNetwork net;
	public int iterations;
	public double base_lrate;
	public int M, size;
	protected Kryo kryo;

	/**
	 * 
	 * @param conf MultiLayer configuration
	 * @param iterations number of iterations to create a learning rate
	 * @param base_lrate initial learning rate value
	 * @param M number of gradients to aggregate before updating the model (M-softsync)
	 * @param size staleness range will be = [0, size - 1] 
	 */
	public Dl4jModel(MultiLayerConfiguration conf, int iterations, double base_lrate, int M, int size) {
		kryo = new Kryo();
		kryo.register(MyMultiLayerNetwork.class, new MultiLayerNetworkSerializer());
		net = new MyMultiLayerNetwork(conf);
		this.iterations = iterations;
		this.base_lrate = base_lrate;
		this.M = M;
		this.size = size;
		this.initialize();
	}

	@Override
	public void initialize() {
		net.init();
	}

	
	public void train(MyDataset miniBatch) {
	}
	
	public byte[] getParams() {
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		Output output = new Output(outputStream);
		
		// Save the model to file
		// Output output = new Output(new
		// FileOutputStream("MLPModel"));

		// Save the model to buffer
		// Output output = new Output(2048, -1); // problems with big files
		
		// write learning rate schedule 
		ArrayList<Double> lrates = new ArrayList<Double>();
		double lrate;
		//System.out.println("Learning rate schedule: if (lrate > base_lrate / 2) lrate = base_lrate / Math.pow(1 + i, 0.1);");
		for (int i=0; i<iterations; i++) {
			lrate = base_lrate / Math.pow(1 + i, 0.1);
			if (lrate > base_lrate / 2)
				//lrates.add(lrate);
				lrates.add(base_lrate);
			else {
				System.out.println("Total lrates: " + lrates.size());
				break;
			}
		}
		System.out.println("Total lrates: " + lrates.size());
		System.out.println("Last lrate: " + lrates.get(lrates.size() - 1));
		
		kryo.register(ArrayList.class, new CollectionSerializer());
		kryo.writeObject(output, lrates);	
		
		// write staleness size for distribution
		kryo.register(Integer.class);
		kryo.writeObject(output, size);
		
		// write M-softsync param
		kryo.register(Integer.class);
		kryo.writeObject(output, M);
		
		// ! write the model in the end to avoid kryo overflow issues
		kryo.register(MyMultiLayerNetwork.class, new MultiLayerNetworkSerializer());
		kryo.writeObject(output, net);

		output.close();
		return outputStream.toByteArray();
	}

	/**
	 * Net output of the dataset in steps 
	 * to avoid Integer.MAX_VALUE exception of Nd4jAvoid
	 * @param dataset
	 * @return
	 */
	private INDArray stepOutput(NDArray setX) {
		// evaluate the dataset in steps to avoid Integer.MAX_VALUE exception of Nd4j
		int sampleSize; 
		int[] ids;
		
		// evaluation on the entire dataset at once
//		sampleSize = setX.rows();
//		for (int i=0; i< Math.ceil(setX.rows() / (float) sampleSize); i++) {
//			// get ids range
//			int start = i * sampleSize;
//			int fin = Math.min((i + 1) * sampleSize, setX.rows());
//			ids = new int[fin - start];
//			for (int j=0; j< fin-start; j++)
//				ids[j] = i * sampleSize + j;
//			
//			// concatenate outputs
//			if (i==0)
//				output = net.output(setX.getRows(ids));
//			else
//				output = Nd4j.concat(0, output, net.output(setX.getRows(ids)));
//		}
		
		sampleSize = 1000; // FIXME bogus
		INDArray output2 = null;
		for (int i=0; i< Math.ceil(setX.rows() / (float) sampleSize); i++) {
			// get ids range
			int start = i * sampleSize;
			int fin = Math.min((i + 1) * sampleSize, setX.rows());
			ids = new int[fin - start];
			for (int j=0; j< fin-start; j++)
				ids[j] = i * sampleSize + j;
			
			// concatenate outputs
			if (i==0)
				output2 = net.output(setX.getRows(ids));
			else
				output2 = Nd4j.concat(0, output2, net.output(setX.getRows(ids)));
		}
//		System.out.println("CHECK: " + output.equals(output2));
		
		return output2;
	}
	
	/**
	 * Net score of the dataset in steps 
	 * to avoid Integer.MAX_VALUE exception of Nd4jAvoid
	 * @param dataset
	 * @return
	 */
	private double stepScore(DataSet dataset) {
		// evaluate the dataset in steps to avoid Integer.MAX_VALUE exception of Nd4j
		int sampleSize, i; 
		int[] ids;
		double score = 0;
		
//		// evaluation on the entire dataset at once
//		sampleSize = dataset.numExamples();
//		
//		for (i=0; i< Math.ceil(dataset.numExamples() / (float) sampleSize); i++) {
//			// get ids range
//			int start = i * sampleSize;
//			int fin = Math.min((i + 1) * sampleSize, dataset.numExamples());
//			ids = new int[fin - start];
//			for (int j=0; j< fin-start; j++)
//				ids[j] = i * sampleSize + j;
//			
//			score += net.score(dataset.get(ids));
//		}
//		System.out.println(score / (float) i);
//		
//		System.out.println(net.score(dataset));
		
		sampleSize = 1000; // FIXME bogus
		score = 0;
		int count = 0;
		for (i=0; i< Math.ceil(dataset.numExamples() / (float) sampleSize); i++) {
			// get ids range
			int start = i * sampleSize;
			int fin = Math.min((i + 1) * sampleSize, dataset.numExamples());
			ids = new int[fin - start];
			for (int j=0; j< fin-start; j++) {
				ids[j] = i * sampleSize + j;
			}
			
			score += net.score(dataset.get(ids)) * ids.length;
			count += ids.length;
		}
		
		System.out.println(score / (float) count);

//		System.out.println("CHECK: " + output.equals(output2));
		
		return score;
	}
	
	@Override
	public Result evaluate(MyDataset d) {
		NDArray setX, setT;	
		MyDl4jDataset dataset = (MyDl4jDataset) d;
		setX = (NDArray) dataset.dataset.getFeatures();
		setT = (NDArray) dataset.dataset.getLabels();
		
//		double lrate;
//		System.out.println("Gamma: " net.getLayer(0));
//		for (int l = 0; l < net.getnLayers(); l++) {
//			System.out.println("Layer: " + l);
//			for (String key : net.getLayer(l).conf().getLearningRateByParam().keySet()) {
//				System.out.println("Learning rate key: " + key);
//				lrate = net.getLayer(l).conf().getLearningRateByParam(key);
//				System.out.println("Learning rate: " + lrate);
//			}
//		}
		
		// TODO set depending on test set size
//		INDArray output = net.output(setX);
		INDArray output = stepOutput(setX);

		// get predictions before getting the error not to affect output object
		// compute accuracy based on argmax
		INDArray argmaxOut = Nd4j.getExecutioner().exec(new IAMax(output), 1);
		INDArray argmaxLabels = Nd4j.getExecutioner().exec(new IAMax(setT), 1);

		int hits = 0;
		
		// following commented out code is slower for computing the accuracy
//		final Integer[] sum = new Integer[1];
//		sum[0] = 0;
//	      Consumer<Double> myConsumer = x -> 
//	      {
//	          if (x==0)
//	        	  sum[0]++;
//	      };
//	    
//	    NDArray res = (NDArray) argmaxOut.sub(argmaxLabels);
//		System.out.println(res.sumNumber());
//
//		res.forEach(myConsumer);
//		System.out.println(sum[0]);
		
		for (int i = 0; i < setT.rows(); i++) {
			if (argmaxOut.getDouble(i) == argmaxLabels.getDouble(i))
				hits++;
		}

		double accuracy = 100 * hits / (double) setT.rows();
		
//		double accuracy = net.f1Score(dataset.dataset); 

		// calculate error
//		double error = LossFunctions.score(setT, LossFunction.MCXENT, output, 0, 0, false);
//		INDArray temp = Transforms.log(output);
//		NDArray ones = Nd4j.ones(shape)
		
//		double error = CustomCrossEntropy.computeScore(setT, output, true); // !affects output object
		
		// TODO set depending on test set size
//		double error = net.score(dataset.dataset);
		double error = stepScore(dataset.dataset);
		
		return new Result(error, accuracy);
	}

	@Override
	public void printParams() {
		// TODO Auto-generated method stub

	}

	@Override
	public int getcurrEpoch() {
		return net.currEpoch;
	}

	@Override
	public long getTrainTime() {
		return net.trainTime;
	}

	@Override
	public void fetchParams(InputStream input) {
		Input in = new Input(input);

		try {
		// get extra params
		Dl4jExtraParams extras = kryo.readObject(in, Dl4jExtraParams.class);
		// get model configuration
		DeflateSerializer deflser1 = new DeflateSerializer(new DefaultSerializers.StringSerializer());
		deflser1.setCompressionLevel(9);
		MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(
				kryo.readObject(in, String.class, deflser1));
		// get model params
		DeflateSerializer deflser2 = new DeflateSerializer(new Nd4jSerializer());
		deflser2.setCompressionLevel(9);
//		Nd4j.getCompressor().setDefaultCompression("FLOAT16");
 //       NDArray params = (NDArray) Nd4j.getCompressor().decompress(
 //       		kryo.readObject(in, NDArray.class, deflser2));
		NDArray params = kryo.readObject(in, NDArray.class, deflser2);
        //NDArray params = kryo.readObject(in, NDArray.class, new Nd4jSerializer());
		net = new MyMultiLayerNetwork(conf, params);
		net.currEpoch = extras.currEpoch;
		net.trainTime = extras.trainTime;
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("Read Bytes: " + Helpers.humanReadableByteCount(in.total(), false));

	}

	@Override
	public int[][] predict() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void saveState() {
		// TODO Auto-generated method stub

	}

	@Override
	public void restoreState() {
		// TODO Auto-generated method stub

	}

	@Override
	public void cleanUp() {
		// TODO Auto-generated method stub

	}

}
