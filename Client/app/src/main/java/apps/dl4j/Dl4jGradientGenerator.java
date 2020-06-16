/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.dl4j;

import android.util.Log;

import coreComponents.GradientGenerator;
import utils.Helpers;
import utils.dl4j.MyMultiLayerNetwork;
import utils.dl4j.Nd4jSerializer;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.DefaultSerializers;
import com.esotericsoftware.kryo.serializers.DeflateSerializer;
import com.esotericsoftware.kryo.serializers.MapSerializer;
import com.esotericsoftware.kryo.serializers.CollectionSerializer;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.HashMap;

//import org.deeplearning4j.nn.conf.layers.DenseLayer;

/**
 * MLP with DL4J library
 */
public class Dl4jGradientGenerator implements GradientGenerator {

    private MyMultiLayerNetwork net;
    private Kryo kryo;
    private MapSerializer mapser;

    private DataSet miniBatch;
    private int trainN;
    private double fetchMiniBatchTime;
    private double fetchModelTime;
    private double computeGradientsTime;

    /**
     * Received model extra parameters (i.e. epoch, trainTime, hashCode)
     */
    private Dl4jExtraParams extras;

    public Dl4jGradientGenerator() {
        kryo = new Kryo();
        mapser = new MapSerializer();
        mapser.setValueClass(NDArray.class, new Nd4jSerializer());
    }

    @Override
    public void computeGradient(Output output) {

        trainN = miniBatch.getFeatures().rows();
        double t = System.currentTimeMillis();
        double t1 = System.currentTimeMillis();
        Gradient gradient = net.getGradients(miniBatch);

        //Log.d("INFO", "Number of gradients: " + Arrays.toString(gradient.gradient().shape())); // !! may cause problems to the gradient object. NEEDS CHECK!!

        //Log.d("INFO", "Gradient computing time: " + String.valueOf(System.currentTimeMillis() - t) + " ms");
        //Log.d("INFO", String.valueOf(gradient.gradient().length()));
        //Log.d("INFO", String.valueOf(gradient.gradient()));
        t = System.currentTimeMillis();
        // send gradient extra params (model info)
        kryo.register(Integer.class);
        kryo.writeObject(output, extras.hashCode);
        kryo.writeObject(output, extras.currEpoch);


        // send gradients as a flattened NDArray
        //  kryo.register(NDArray.class, new Nd4jSerializer());
        // kryo.writeObject(output, gradient.gradient());

        // send gradients as a map (String -> NDArray)
        HashMap<String, NDArray> gradientsMap = new HashMap<>();
        for (String key : gradient.gradientForVariable().keySet()) {
            //temp.put(key, (NDArray) gradient.gradientForVariable().get(key));
            gradientsMap.put(key, (NDArray) gradient.getGradientFor(key));
        }
        kryo.register(HashMap.class, mapser);
        kryo.writeObject(output, gradientsMap);
        gradientsMap.clear();
        System.gc();

        // send flattening info and ordered list of keys useful for parsing the gradients on the server
        HashMap<String, Character> flattenInfoMap = new HashMap<>();
        ArrayList<String> orderedKeys = new ArrayList<>();
        for (String key : gradient.gradientForVariable().keySet()) {
            //temp.put(key, (NDArray) gradient.gradientForVariable().get(key));
            flattenInfoMap.put(key, gradient.flatteningOrderForVariable(key));
            orderedKeys.add(key);
        }

        kryo.register(ArrayList.class, new CollectionSerializer());
        kryo.writeObject(output, orderedKeys);
        kryo.register(HashMap.class, new MapSerializer());
        kryo.writeObject(output, flattenInfoMap);
        orderedKeys.clear();
        System.gc();
        flattenInfoMap.clear();
        System.gc();


//        HashMap<String, Integer> temp = new HashMap();
//        temp.put("tsa", 1);
//        temp.put("pou", 0);
//
//        kryo.register(HashMap.class, new MapSerializer());
//        kryo.writeObject(output, temp);

        output.flush();
        computeGradientsTime = System.currentTimeMillis() - t1;
        //Log.d("INFO", "Gradient serialization time: " + String.valueOf(System.currentTimeMillis() - t) + " ms");
        Log.d("INFO", "Total Bytes written: " + Helpers.humanReadableByteCount(output.total(), false));
    }

    @Override
    public int getSize() {
        return trainN;
    }

    @Override
    public double getFetchMiniBatchTime() {
        return fetchMiniBatchTime;
    }

    @Override
    public double getFetchModelTime() {
        return fetchModelTime;
    }

    @Override
    public double getComputeGradientsTime() {
        return computeGradientsTime;
    }

    public void fetch(Input in) {
        double t = System.currentTimeMillis();

//        MyDataset m = kryo.readObject(in, MyDataset.class);

//		NDArray outX = new NDArray(m.Xset.toArray2());
//		NDArray outT = new NDArray(m.Tset.toArray2());

//        NDArray outX = (NDArray) kryo.readObject(in, Nd4j.getBackend().getNDArrayClass(), new Nd4jSerializer());
//        NDArray outT = (NDArray) kryo.readObject(in, Nd4j.getBackend().getNDArrayClass(), new Nd4jSerializer());

        NDArray outX = kryo.readObject(in, NDArray.class, new Nd4jSerializer());
        NDArray outT = kryo.readObject(in, NDArray.class, new Nd4jSerializer());

        //Log.d("INFO", "MiniBatch Fetch time: " + String.valueOf(System.currentTimeMillis() - t) + " ms");
        fetchMiniBatchTime = System.currentTimeMillis() - t;

        miniBatch = new DataSet(outX, outT);
        outX.cleanup();
        outT.cleanup();
        System.gc();

        Log.d("INFO", "MiniBatch Size: " + String.valueOf(miniBatch.numExamples()));

        // get extra params
        extras = kryo.readObject(in, Dl4jExtraParams.class);
        Log.d("INFO", "Received model epoch: " + extras.currEpoch);
        // get model configuration
        t = System.currentTimeMillis();
//        MultiLayerNetwork restored = kryo.readObject(in, MyMultiLayerNetwork.class, new MultiLayerNetworkSerializer());

        DeflateSerializer deflser1 = new DeflateSerializer(new DefaultSerializers.StringSerializer());
        deflser1.setCompressionLevel(9);
        MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(kryo.readObject(in, String.class, deflser1));
        // MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(kryo.readObject(in, String.class));
        //Log.d("INFO", "Model conf fetch time: " + String.valueOf(System.currentTimeMillis() - t) + " ms");

        // get model params
        deflser1 = new DeflateSerializer(new Nd4jSerializer());
        deflser1.setCompressionLevel(9);
        NDArray params = kryo.readObject(in, NDArray.class, deflser1);
        Log.d("INFO", "Number of model params: " + params.length());
        //  Nd4j.getCompressor().setDefaultCompression("FLOAT16");
        //  NDArray params = (NDArray) Nd4j.getCompressor().decompress(
        //          kryo.readObject(in, NDArray.class, deflser2));
        //NDArray params = kryo.readObject(in, NDArray.class, new Nd4jSerializer());
        //Log.d("INFO", "Model params fetch time: " + String.valueOf(System.currentTimeMillis() - t) + " ms");
        fetchModelTime = System.currentTimeMillis() - t;

        net = new MyMultiLayerNetwork(conf, params);

        Log.d("INFO", "Total Bytes read: " + Helpers.humanReadableByteCount(in.total(), false));

    }
}
