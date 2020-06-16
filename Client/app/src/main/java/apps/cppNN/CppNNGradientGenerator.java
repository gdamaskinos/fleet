/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package apps.cppNN;

import android.util.Log;

import coreComponents.GradientGenerator;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.DefaultArraySerializers;

import java.io.FileReader;

import coreComponents.R;
import utils.Helpers;


public class CppNNGradientGenerator implements GradientGenerator {

    private Kryo kryo;

    private int epoch;
    private int trainN;

    private double fetchMiniBatchTime;
    private double fetchModelTime;
    private double computeGradientsTime;

    /**
     * total training time
     */
    private long trainTime;
    /**
     * Learning model hashCode
     */
    private int hashCode;

    /**
     * id assigned by the server
     */
    private int id;

    private byte[] miniBatch;

    static {
        System.loadLibrary("cppNN-lib");
    }
    private native byte[] getGradients();
    private native void printParamsNative(byte[] inBuffer);
    private native void fetchNative(byte[] inBuffer);
    private native int fetchMiniBatch(byte[] inBuffer);
    private native int[] getClassDist();

    public CppNNGradientGenerator(){
        kryo = new Kryo();
    }

    public void fetch(Input in) {

        double t = System.currentTimeMillis();

        // fetch miniBatch
        kryo.register(byte[].class, new DefaultArraySerializers.ByteArraySerializer());
        miniBatch = kryo.readObject(in, byte[].class);
        trainN = fetchMiniBatch(miniBatch);

        fetchMiniBatchTime = System.currentTimeMillis() - t;

        t = System.currentTimeMillis();

        // fetch Extras
        kryo.register(Integer.class);
        epoch = kryo.readObject(in, Integer.class);
        Log.d("INFO", "Received model epoch: " + epoch);

        hashCode = kryo.readObject(in, Integer.class);
        id = kryo.readObject(in, Integer.class);
        long trainTime = kryo.readObject(in, Long.class);

        // fetch Model
        kryo.register(byte[].class, new DefaultArraySerializers.ByteArraySerializer());
        byte[] nativeOutput = kryo.readObject(in, byte[].class);

        //System.out.println("Received: " + new String(nativeOutput, StandardCharsets.UTF_8));
        fetchNative(nativeOutput);
 //       trainTime = params.trainTime;
 //       currEpoch = params.currEpoch;
        fetchModelTime = System.currentTimeMillis() - t;

        //System.out.println("Fetched w1: " + w1);

    }

    public void computeGradient(Output output){

        kryo.register(Integer.class);
        kryo.writeObject(output, hashCode);
        kryo.writeObject(output, id);
        kryo.writeObject(output, epoch);
        kryo.writeObject(output, trainN);

        kryo.register(byte[].class, new DefaultArraySerializers.ByteArraySerializer());
        //Log.d("INFO", "Sending: SimpleCNNGradientGenerator#ComputeGradientCheck!");
        //kryo.writeObject(output, "SimpleCNNGradientGenerator#ComputeGradientCheck!".getBytes());

        double begin = System.currentTimeMillis();
        byte[] gradients = getGradients();
        computeGradientsTime = System.currentTimeMillis() - begin;

        kryo.writeObject(output, gradients);
//        this.trainN = trainX.getRowDimension();

//        kryo.register(MLPGradients.class);
//        kryo.writeObject(output, gradients);

        kryo.register(int[].class);
        kryo.writeObject(output, getClassDist());

        Log.d("INFO", "Total Bytes written: " +
                Helpers.humanReadableByteCount(output.total(), false));
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


}

