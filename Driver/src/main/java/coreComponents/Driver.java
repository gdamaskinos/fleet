/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.util.FastMath;
import org.apache.http.HttpEntity;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.conn.HttpHostConnectException;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import apps.cppNN.CppNNModel;
import apps.dl4j.Dl4jModel;
import apps.dl4j.Dl4jMultiLabelModel;
import apps.lr.LRModel;
import apps.mlp.MLPModel;
import apps.simpleCNN.SimpleCNNModel;
import au.com.bytecode.opencsv.CSVReader;
import utils.MyDataset;
import utils.Evaluator;
import utils.Helpers;
import utils.JNITest;
import utils.Parser;
import utils.dl4j.MyDl4jDataset;

/**
 * Initializes model on Server and evaluates periodically
 * 
 *
 */
public class Driver {

	
	public static void main(String[] args) throws ClientProtocolException, IOException, InterruptedException {

//		double[][] data = new double[10][10];
//		System.out.println(data[2][2]);
//		Nd4j.getCompressor().setDefaultCompression("FLOAT8");
//		NDArray array = (NDArray) Nd4j.getCompressor().compress(Nd4j.create(data));
//		NDArray out = (NDArray) Nd4j.getCompressor().decompress(array);
//		System.out.println(out);
//		System.exit(0);
		
		
	
		if (args.length < 9) {
			System.out.println(
					"Example Usage is: mvn exec:java -Dexec.mainClass=\"coreComponents.Driver\" -Dexec.args=\"http://server.ip:port/Server/Server\" /local/path/to/dataset/ /server's/path/to/dataset/ <batch size percentile threshold> <similarity percentile threshold> <number of client requests> <eval rounds> <lrate> <M> <E> <size> <policy> <alpha>");
			System.out.println("Check constructor of CppNNModel.java for info about extra arguments (M, size, etc)");
			System.exit(0);
		}

		new JNITest().hello();
	
		int i=0;
		String serverPath = args[i++];
		String localPathPrefix = args[i++];
		String serverPathPrefix = args[i++];
		int batch_size_threshold = Integer.parseInt(args[i++]);
		int similarity_threshold = Integer.parseInt(args[i++]);
		int clientRequestsNum = Integer.parseInt(args[i++]);
		int evalRounds = Integer.parseInt(args[i++]);
		double lrate = Double.parseDouble(args[i++]);
		int M = Integer.parseInt(args[i++]);
		int E = Integer.parseInt(args[i++]);
		double sigma = Double.parseDouble(args[i++]);
		double C = Double.parseDouble(args[i++]);
		int size = Integer.parseInt(args[i++]);
		int policy = Integer.parseInt(args[i++]);
		double alpha = Double.parseDouble(args[i++]);
		
		System.out.println("Server path: " + serverPath);
		System.out.println("local path prefix: " + localPathPrefix);
		System.out.println("Server path prefix: " + serverPathPrefix);
		System.out.println("Batch size threshold: " + batch_size_threshold);
		System.out.println("Similarity threshold: " + similarity_threshold);
		System.out.println("Evaluation rounds: " + evalRounds);
		System.out.println("Number of client requests: " + clientRequestsNum);
		System.out.println("Learning rate: " + lrate);
		System.out.println("M: " + M);
		System.out.println("E: " + E);
		System.out.println("Sigma: " + sigma);
		System.out.println("C: " + C);
		System.out.println("Staleness size: " + size);
		System.out.println("Policy: " + policy);
		System.out.println("alpha: " + alpha);
		
		String trainXPath = "training_features.csv";
		String trainTPath = "training_labels.csv";
		String validXPath = "validation_features.csv";
		String validTPath = "validation_labels.csv";
		String testXPath = "test_features.csv";
		String testTPath = "test_labels.csv";

		MyDataset trainSet = null, validSet = null, testSet = null;
		// TODO uncomment if dataset parsed from Java (and not JNI) 
//		System.out.println("Dataset/Examples/Features/Labels");
//		trainSet = createDataSet(localPathPrefix + trainXPath, localPathPrefix + trainTPath);
//		System.out.println("Training  \t" + trainSet.numExamples() + "\t" + trainSet.featureSize() + "\t" + trainSet.numLabels());
//		validSet = createDataSet(localPathPrefix + validXPath, localPathPrefix + validTPath);
//		System.out.println("Validation\t" + validSet.numExamples() + "\t" + validSet.featureSize() + "\t" + validSet.numLabels());
//		testSet = createDataSet(localPathPrefix + testXPath, localPathPrefix + testTPath);
//		System.out.println("Test      \t" + testSet.numExamples() + "\t" + testSet.featureSize() + "\t" + testSet.numLabels());
//		int featureSize = validSet.featureSize(); // number of rows and columns in the input
//		int numLabels = validSet.numLabels(); // number of output classes
		
		int rngSeed = 123; // random number seed for reproducibility
		
		Evaluator eval = new Evaluator(serverPath);


		/* MLP */
//		// spambase (z-score standardization, ratio1=0.8, ratio2=0.9) -> 92% @ 500+ epochs (not converged)
//		// mnist (X = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(X), ratio1=60000, ratio2=54000) -> epoch:48 Error:1.7392 Acc:0.6517 (not converged)
//        int batchSize = 128; // batch size for each epoch
//		lrate = 0.006 / 128; // constant
//		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed) 
//				// use stochastic gradient descent as an optimization algorithm
//				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//				.iterations(1)
//				.learningRate(lrate) 
//				.updater(Updater.NESTEROVS).momentum(0.9) 
//				.regularization(true).l2(1e-4)
//				.list()
//				.layer(0, new DenseLayer.Builder() 
//						.nIn(featureSize)
//						.nOut(1000)
//						.activation(Activation.RELU)
//						.weightInit(WeightInit.XAVIER)
//						.build())
//				.layer(1, new DenseLayer.Builder().activation(Activation.RELU)
//	                		.nOut(numLabels).build())
//				.layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) 
//						.nOut(numLabels)
//						.activation(Activation.SOFTMAX)
//						.weightInit(WeightInit.XAVIER)
//						.build())
//				.pretrain(false).backprop(true) 
//				.build();
	 
		
		/* CNN */
		// lastfm dataset 
//		int height = 28;
//		int width = featureSize / 28;
//		int nChannels = 1;
//		int lid = 0;
//      int batchSize = 50; // batch size for each epoch
//		lrate = 0.00005 / 50;
//		System.out.println("Lrate: " + lrate);
//		System.out.println("BatchSize: " + batchSize);
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(rngSeed) //include a random seed for reproducibility
//                .regularization(true).l2(5 * 1e-4) 
//                .activation(Activation.RELU)
//                .learningRate(lrate)
//               // .biasLearningRate(lrate * 0.1)
//                .weightInit(WeightInit.XAVIER)
//                //.weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) 
//                .updater(Updater.NESTEROVS) //.momentum(0.9) 
//                .list()
//               //  .weightInit(WeightInit.XAVIER)
//             /* !Only set initial learning rate here. Do not set LearningRatePolicy -> must change learning rate manually on the Dl4jUpdater.java */
//                // Layer 0
//            //    .layer(lid++, new BatchNormalization.Builder(false) // keep learning rate to 1 on Updater. !If lock gamma and beta -> errors on the client
//            //   		.nOut(featureSize)
//             //   		.name("Batch " + String.valueOf(lid)).build())
//                .layer(lid++, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
//                        /* nIn and nOut specify depth. nIn here is the nChannels (input depth) and nOut is the number of filters to be applied (output depth) */
//                        .nIn(nChannels)
//                        .nOut(32)
//                        //.weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                        .biasInit(Math.pow(10,-3))
//                        //.convolutionMode(ConvolutionMode.Same)
//                        //.activation(Activation.RELU)
//                        .name("Conv2d " + String.valueOf(lid)).build())
//           //     .layer(lid++, new BatchNormalization.Builder(false)
//           //     		.name("Batch " + String.valueOf(lid)).build())
//                .layer(lid++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 4}, new int[]{2, 4})
//                		//.convolutionMode(ConvolutionMode.Same)
//                		.name("Pooling " + String.valueOf(lid)).build())
//                // Layer 1
//                .layer(lid++, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
//                        /* nIn and nOut specify depth. nIn here is the nChannels (input depth) and nOut is the number of filters to be applied (output depth) */
//                        .nIn(32)
//                        .nOut(128)
//                        //.weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                        .biasInit(Math.pow(10,-3))
//                        //.convolutionMode(ConvolutionMode.Same)
//                        //.activation(Activation.RELU)
//                        .name("Conv2d " + String.valueOf(lid)).build())
//            //    .layer(lid++, new BatchNormalization.Builder(false)
//            //    		.name("Batch " + String.valueOf(lid)).build())
//                .layer(lid++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 4}, new int[]{2, 4})
//                		//.convolutionMode(ConvolutionMode.Same)
//                		.name("Pooling " + String.valueOf(lid)).build())
//                // Output Layer
//                .layer(lid++, new DenseLayer.Builder()
//                		.nOut(numLabels)
//                		//.activation(Activation.IDENTITY)
//                		//.l2(5e-4)
//                        //.weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                        //.biasInit(0.1)
//                        .name("Dense " + String.valueOf(lid)).build())
//                .layer(lid++, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) 
//                		.activation(Activation.SIGMOID)
//                        .nOut(numLabels)
//                        .name("Output " + String.valueOf(lid)).build())
//                .setInputType(InputType.convolutionalFlat(height, width, nChannels)) 
//                .pretrain(false).backprop(true).build();
        
//		int height = 28;
//		int width = featureSize / 28;
//		int nChannels = 1;
//		int lid = 0;
//		// lrate = 0.006;
//		lrate = 0.006;
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(rngSeed) //include a random seed for reproducibility
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) 
//              //  .weightInit(WeightInit.XAVIER)
//             /* !Only set initial learning rate here. Do not set LearningRatePolicy -> must change learning rate manually on the Dl4jUpdater.java */
//                .learningRate(lrate) 
//                .updater(Updater.NESTEROVS).momentum(0.9) 
//                .regularization(true).l2(5e-2) 
//                .list()
//                // Layer 0
//            //    .layer(lid++, new BatchNormalization.Builder(false) // keep learning rate to 1 on Updater. !If lock gamma and beta -> errors on the client
//            //   		.nOut(featureSize)
//             //   		.name("Batch " + String.valueOf(lid)).build())
//                .layer(lid++, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
//                        /* nIn and nOut specify depth. nIn here is the nChannels (input depth) and nOut is the number of filters to be applied (output depth) */
//                        .nIn(nChannels)
//                        .nOut(32)
//                        .weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                        .biasInit(0.1)
//                        .convolutionMode(ConvolutionMode.Same)
//                        .activation(Activation.ELU)
//                        .name("Conv2d " + String.valueOf(lid)).build())
//           //     .layer(lid++, new BatchNormalization.Builder(false)
//           //     		.name("Batch " + String.valueOf(lid)).build())
//                .layer(lid++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 4}, new int[]{2, 4})
//                		.convolutionMode(ConvolutionMode.Same)
//                		.name("Pooling " + String.valueOf(lid)).build())
//                // Layer 1
//                .layer(lid++, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
//                        /* nIn and nOut specify depth. nIn here is the nChannels (input depth) and nOut is the number of filters to be applied (output depth) */
//                        .nIn(32)
//                        .nOut(128)
//                        .weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                        .biasInit(0.1)
//                        .convolutionMode(ConvolutionMode.Same)
//                        .activation(Activation.ELU)
//                        .name("Conv2d " + String.valueOf(lid)).build())
//            //    .layer(lid++, new BatchNormalization.Builder(false)
//            //    		.name("Batch " + String.valueOf(lid)).build())
//                .layer(lid++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 4}, new int[]{2, 4})
//                		.convolutionMode(ConvolutionMode.Same)
//                		.name("Pooling " + String.valueOf(lid)).build())
//                // Output Layer
//                .layer(lid++, new DenseLayer.Builder()
//                		.nOut(numLabels)
//                		.activation(Activation.IDENTITY)
//                		//.l2(5e-4)
//                        .weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                        .biasInit(0.1)
//                        .name("Dense " + String.valueOf(lid)).build())
//                .layer(lid++, new OutputLayer.Builder(LossFunction.MCXENT) 
//                		.activation(Activation.SOFTMAX)
//                        .nOut(numLabels)
//                        .name("Output " + String.valueOf(lid)).build())
//                .setInputType(InputType.convolutionalFlat(height, width, nChannels)) 
//                .pretrain(false).backprop(true).build();

		
		// TODO comment back in the following block of lines for setting the DL4J model
		// mnist (X = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(X), ratio1=60000, ratio2=54000)  eval:112 938 -1.0000 -1.0000 0.1727  0.9533  968468
//		int height = 28;
//		int width = featureSize / 28;
//		int nChannels = 1;
//		int lid = 0;
//		int batchSize = 128;
//		lrate = 0.05 / 128;
//		System.out.println("Lrate: " + lrate);
//		System.out.println("BatchSize: " + batchSize);
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(rngSeed) //include a random seed for reproducibility
//                .iterations(1)
//                .regularization(true).l2(0.0005) 
//                .activation(Activation.RELU)
//                .learningRate(lrate)
//               // .biasLearningRate(lrate * 0.1)
//                .weightInit(WeightInit.XAVIER)
//                //.weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) 
//                .updater(Updater.NESTEROVS).momentum(0.9) 
//                .list()
//               //  .weightInit(WeightInit.XAVIER)
//             /* !Only set initial learning rate here. Do not set LearningRatePolicy -> must change learning rate manually on the Dl4jUpdater.java */
//                // Layer 0
//            //    .layer(lid++, new BatchNormalization.Builder(false) // keep learning rate to 1 on Updater. !If lock gamma and beta -> errors on the client
//            //   		.nOut(featureSize)
//             //   		.name("Batch " + String.valueOf(lid)).build())
//                .layer(lid++, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
//                        /* nIn and nOut specify depth. nIn here is the nChannels (input depth) and nOut is the number of filters to be applied (output depth) */
//                        .nIn(nChannels)
//                        .nOut(32)     
//                        //.stride(1, 1)
//                        //.activation(Activation.IDENTITY)
//                        //.weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                        //.biasInit(0.01)
//                        //.convolutionMode(ConvolutionMode.Same)
//                        //.activation(Activation.RELU)
//                        .name("Conv2d " + String.valueOf(lid)).build())
//           //     .layer(lid++, new BatchNormalization.Builder(false)
//           //     		.name("Batch " + String.valueOf(lid)).build())
//                .layer(lid++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 4}, new int[]{2, 4})
//                		//.kernelSize(2,2) 
//                		//.stride(2,2)
//                		//.convolutionMode(ConvolutionMode.Same)
//                		.name("Pooling " + String.valueOf(lid)).build())
//                // Layer 1
//                .layer(lid++, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
//                        /* nIn and nOut specify depth. nIn here is the nChannels (input depth) and nOut is the number of filters to be applied (output depth) */
//                		//.stride(1, 1)
//                		.nIn(32)
//                        .nOut(128)
//                        //.activation(Activation.IDENTITY)
//                        //.weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                        //.biasInit(0.01)
//                        //.convolutionMode(ConvolutionMode.Same)
//                        //.activation(Activation.RELU)
//                        .name("Conv2d " + String.valueOf(lid)).build())
//            //    .layer(lid++, new BatchNormalization.Builder(false)
//            //    		.name("Batch " + String.valueOf(lid)).build())
//                .layer(lid++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 4}, new int[]{2, 4})
//                		//.kernelSize(2,2) 
//                		//.stride(2,2)
//                		//.convolutionMode(ConvolutionMode.Same)
//                		.name("Pooling " + String.valueOf(lid)).build())
//                // Output Layer
//                .layer(lid++, new DenseLayer.Builder()
//                		.nOut(numLabels)
//                		//.activation(Activation.IDENTITY)
//                		//.l2(5e-4)
//                        //.weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.1))
//                        //.biasInit(0.1)
//                        .name("Dense " + String.valueOf(lid)).build())
//                .layer(lid++, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) 
//                		.activation(Activation.SOFTMAX)
//                        .nOut(numLabels)
//                        .name("Output " + String.valueOf(lid)).build())
//                .setInputType(InputType.convolutionalFlat(height, width, nChannels)) 
//                .backprop(true).pretrain(false).build();
	
		// mnist (X = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(X), ratio1=60000, ratio2=54000) -> epoch:99 Error:0.6663 Acc:0.8398 (not converged)
//		int height = 28;
//		int width = featureSize / 28;
//		int nChannels = 1;
//	    //int batchSize = 64; // batch size for each epoch
//		lrate = 0.08 / 100;
//		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//	                .seed(rngSeed)
//	                .regularization(true).l2(0.0005)
//	                /*
//	                    Uncomment the following for learning decay and bias
//	                 */
//	                .learningRate(lrate)//.biasLearningRate(0.02)
//	                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
//	                .weightInit(WeightInit.XAVIER)
//	                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//	                .updater(Updater.NESTEROVS).momentum(0.9)
//	                .list()
//	                .layer(0, new ConvolutionLayer.Builder(5, 5)
//	                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
//	                        .nIn(nChannels)
//	                        .stride(1, 1)
//	                        .nOut(10)
//	                        .activation(Activation.IDENTITY)
//	                        .build())
//	                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//	                        .kernelSize(2,2)
//	                        .stride(2,2)
//	                        .build())
//	                .layer(2, new ConvolutionLayer.Builder(5, 5)
//	                        //Note that nIn need not be specified in later layers
//	                        .stride(1, 1)
//	                        .nOut(10)
//	                        .activation(Activation.IDENTITY)
//	                        .build())
//	                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//	                        .kernelSize(2,2)
//	                        .stride(2,2)
//	                        .build())
//	                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
//	                        .nOut(15).build())
//	                .layer(5, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
//	                        .nOut(numLabels)
//	                        .activation(Activation.SOFTMAX)
//	                        .build())
//	                .setInputType(InputType.convolutionalFlat(height, width, nChannels)) 
//	                .backprop(true).pretrain(false).build();
		
		// TODO change here with XModel
		//Dl4jModel model = new Dl4jModel(conf, clientRequestsNum, lrate, 1, 25);
		//Dl4jMultiLabelModel model = new Dl4jMultiLabelModel(conf, clientRequestsNum, lrate, 1, 1, 0.001);
		//Dl4jMultiLabelModel model = new Dl4jMultiLabelModel(conf, clientRequestsNum, lrate, 1, 1, 1);

		//SimpleCNNModel model = new SimpleCNNModel(iterations, 0.0102, 2, 1, 28, 28, 1);
//		MojoCNNModel model = new MojoCNNModel(iterations, 0.0008, 1, 1, localPathPrefix, false, null); // MNIST
//		CppNNModel model = new CppNNModel(iterations, 0.0015, 1, 25, localPathPrefix, false, null); // CIFAR-100 + MNIST (sgd)
//        CppNNModel model = new CppNNModel(iterations, 0.01, 1, 1, localPathPrefix, false, null); // CIFAR-100 (rmsprop)
        
		CppNNModel model = new CppNNModel(clientRequestsNum, lrate, M, E, sigma, C, batch_size_threshold, similarity_threshold, size, policy, alpha, localPathPrefix, false, null);


		//MLPModel model = new MLPModel(0.5, 0.02, 0.5, 0.001, featureSize, 60);
	 	//LRModel model = new LRModel(8 / 100.0, 8, 770); // ! get data in DoubleMatrix
		//MLPModel model = new MLPModel(0.5, 0.02, 0.5, 0.001, featureSize, 60); // !!! Binary MNIST
		
		initializeServer(serverPath, serverPathPrefix, model);
		//initializeServer(serverPath, null, model);

		eval.crossValidation(trainSet, validSet, testSet, model, evalRounds, 1, 100, false);
		//eval.crossValidation(trainSet, validSet, testSet, model, 1, 1, 100, false);

	}

	/**
	 * Create a data set according to the paths
	 * @param inputX:
	 *            path to training features
	 * @param labelsT:
	 *            path to training labels
	 * @return DataSet
	 * @throws IOException
	 */
	private static MyDataset createDataSet(String inputX, String labelsT) throws IOException {
		Parser inputP = new Parser(inputX);
		Parser labelsP = new Parser(labelsT);
		// TODO change here with the preferred dataset parser 
		//MyDataset data = new MyDl4jDataset(inputP.getValues(), labelsP.getValues()); // get data in NDArray
		MyDataset data = new MyDataset(inputP.getValueMatrix(), labelsP.getValueMatrix()); // get data in DoubleMatrix
		return data;
	}

	/**
	 * Initialize with Server with Post HTTP request
	 * @param serverPath server servlet url
	 * @param remotePrefix optional prefix for datasets on the server. Put null to trigger only model initialization
	 * @param model 
	 * @throws ClientProtocolException
	 * @throws IOException
	 */
	private static void initializeServer(String serverPath, String remotePrefix, Model model)
			throws ClientProtocolException, IOException {
		CloseableHttpClient httpClient = HttpClients.createDefault();
		HttpPost uploadFile = new HttpPost(serverPath);
		MultipartEntityBuilder builder = MultipartEntityBuilder.create(); // http://mvnrepository.com/artifact/org.apache.httpcomponents/httpmime/4.3.1
		builder.addTextBody("clientType", "Initialize", ContentType.TEXT_PLAIN);
		if (remotePrefix != null) // this will cause the server to reload the dataset in memory
			builder.addTextBody("prefix", remotePrefix, ContentType.TEXT_PLAIN);
		
		byte[] bytes = model.getParams();
		System.out.println("Serialized model size: " + Helpers.humanReadableByteCount(bytes.length, false));

		// This attaches the file to the POST:
		builder.addBinaryBody("model", bytes);

//		 File f = new File("MLPModel");
//		 builder.addBinaryBody(
//		 "model",
//		 f,
//		 ContentType.APPLICATION_OCTET_STREAM,
//		 "MLPModel"
//		 );

		HttpEntity multipart = builder.build();
		uploadFile.setEntity(multipart);
		
		boolean succeed;
		CloseableHttpResponse response = null;
		do {
			try {
				response = httpClient.execute(uploadFile);
				succeed = true;
			}
			catch (HttpHostConnectException e1) {
				System.out.println("Cannot reach server...");
				try {
					Thread.sleep(1000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				System.out.println("Retrying...");
				succeed = false;
			}
		}
		while (!succeed);
			
		HttpEntity responseEntity = response.getEntity();

		// System.out.println("Post Status " + response.getStatusLine());
		// System.out.println(EntityUtils.toString(response.getEntity()));

		BufferedReader rd = new BufferedReader(new InputStreamReader(responseEntity.getContent()));

		StringBuffer result = new StringBuffer();
		String line = "";
		while ((line = rd.readLine()) != null) {
			result.append(line);
		}

		System.out.println(result.toString());

	}

}
