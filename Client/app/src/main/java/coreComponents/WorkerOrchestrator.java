/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import android.content.pm.ApplicationInfo;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Debug;
import android.util.Log;
import android.widget.TextView;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.facebook.network.connectionclass.ConnectionClassManager;
import com.facebook.network.connectionclass.DeviceBandwidthSampler;
import com.jaredrummler.android.device.DeviceName;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.params.HttpConnectionParams;
import org.apache.http.params.HttpParams;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipException;

import apps.SPGradientGenerator;
import utils.DeviceInfo;
import utils.Helpers;

//import android.app.ActivityManager.MemoryInfo;

public class WorkerOrchestrator extends AsyncTask<Void, Void, DeviceInfo> {
    private final ApplicationInfo applicationInfo;
    private String dstAddress;
    private int dstPort;
    private boolean requestMade;
    private boolean stop;
    private boolean continueRequests;

    private String clientName;
    private int sleepTime;
    private TextView stat1;
    private MainActivity activity;

    private GradientGenerator calculator = new SPGradientGenerator();
    private double totalLatency = 0;
    private double networkLatency = 0;
    private int numRequests = 0;

    private HttpClient httpClient;
    private HttpPost uploadFile;


    private boolean executeRequest = false;

    private double deviceEnergy;
    private double deviceLatency;
    private double sizeEnergy;
    private double sizeLatency = 0;
    private double pastEnergy = 0;
    private double nowEnergy = 0;

    private Kryo kryo;

    public WorkerOrchestrator(String addr, int port, String clientName, int sleepTime, TextView stat1, MainActivity activity, ApplicationInfo applicationInfo) {
        this.dstAddress = addr;
        this.dstPort = port;
        this.clientName = clientName;
        this.sleepTime = sleepTime;
        this.stat1 = stat1;
        this.activity = activity;
        this.stop = false;
        this.applicationInfo = applicationInfo;
    }

    MultipartEntityBuilder builder;
    HttpEntity multipart;
    HttpResponse response = null;
    HttpEntity responseEntity;
    Input input;
    Output out;
    BufferedReader rd;


    private double downloadModel(String postUrl, DeviceInfo deviceInfo, double tempTime) throws IOException {
        // Computation POST request
        httpClient = new DefaultHttpClient();
        HttpParams httpParameters = httpClient.getParams();
        HttpConnectionParams.setConnectionTimeout(httpParameters, 5 * 1000);
        HttpConnectionParams.setSoTimeout        (httpParameters, 5 * 1000);

        uploadFile = new HttpPost(postUrl);
        builder = MultipartEntityBuilder.create(); // http://mvnrepository.com/artifact/org.apache.httpcomponents/httpmime/4.3.1
        builder.addTextBody("clientType", "Compute", ContentType.TEXT_PLAIN);
        builder.addTextBody("clientID", clientName, ContentType.TEXT_PLAIN);
        String deviceName = DeviceName.getDeviceName();
        //String deviceName = Build.MODEL;
        //System.out.println("Device name: "+deviceName);
        String AndroidVersion = android.os.Build.VERSION.RELEASE;
        //System.out.println("Android version: "+AndroidVersion);
        String serialNumber = android.os.Build.SERIAL;

        String AndroidInfo = deviceName + "," + AndroidVersion + "," + serialNumber;

        builder.addTextBody("androidInfo", AndroidInfo, ContentType.TEXT_PLAIN);
        builder.addTextBody("stats", deviceInfo.getSerializableInfo());

        multipart = builder.build();
        uploadFile.setEntity(multipart);

        executeRequest = true;

        //deviceInfo.deviceEnergy = DeviceInfo.dumpBatteryStats(this.applicationInfo);
        deviceInfo.deviceLatency = System.currentTimeMillis() - tempTime;

        Log.d("INFO", "...Download model and mini-batch "+ postUrl + "..." );
        long startDownloadTime = System.currentTimeMillis();

        try {
            response = httpClient.execute(uploadFile);
            executeRequest = false;

            responseEntity = response.getEntity();

            InputStream inStream = new GZIPInputStream(responseEntity.getContent()); // input for getting the model
            input = new Input(inStream);

            kryo = new Kryo();
            continueRequests = kryo.readObject(input, Boolean.class);
            Log.d("INFO", "continueRequests: " + continueRequests);

            calculator.fetch(input);
            Log.d("INFO", "Read: " + Helpers.humanReadableByteCount(input.total(), false));
            double downloadLatency = System.currentTimeMillis() - startDownloadTime;
            Log.d("INFO", "Download latency: " + downloadLatency + " ms");

            return downloadLatency;
        }
        catch (Exception e) {
            e.printStackTrace();
            return -1;
        }

    }

    private void computeGradient(DeviceBandwidthSampler mDeviceBandwidthSampler, DeviceInfo deviceInfo) {
        Log.d("INFO", "...Processing assigned task...");
        double startProcessTaskTime = System.currentTimeMillis();
        //double startProcessTaskEnergy = DeviceInfo.dumpBatteryStats(this.applicationInfo);
        mDeviceBandwidthSampler.stopSampling();
        deviceInfo.bandwidth = ConnectionClassManager.getInstance().getDownloadKBitsPerSecond();

        out = new Output(2048, -1); // output for gradient post request

        deviceInfo.deviceLatency += System.currentTimeMillis() - startProcessTaskTime;

        double beforeComputeGradientEnergy = DeviceInfo.dumpBatteryStats(this.applicationInfo);
        //deviceInfo.deviceEnergy += beforeComputeGradientEnergy - startProcessTaskEnergy;

        //InputStream is = activity.getResources().openRawResource(R.raw.pht_s8);
        calculator.computeGradient(out); // part of deviceLatency and sizeLatency
        Log.d("INFO", "Written: " + Helpers.humanReadableByteCount(out.total(), false));

        deviceInfo.batchSize = calculator.getSize();
        deviceInfo.sizeLatency = calculator.getComputeGradientsTime();
        deviceInfo.meanSizeLatency = deviceInfo.sizeLatency / (double) calculator.getSize();

        deviceInfo.sizeEnergy = DeviceInfo.dumpBatteryStats(this.applicationInfo) - beforeComputeGradientEnergy;
        Log.d("devSizeEnergy","Value: " + String.valueOf(deviceInfo.sizeEnergy));

        out.close();

        try {
            Log.d("INFO", "...Sleeping for " + sleepTime + " secs...");
            TimeUnit.SECONDS.sleep(sleepTime);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * Continuously make computation requests to the server
     *
     * @param arg0
     * @return
     */
    @Override
    protected DeviceInfo doInBackground(Void... arg0) {
        final String postUrl = "http://" + dstAddress + ":" + dstPort + "/Server/Server";
        double tempTime, startTime;
        double downloadLatency, uploadLatency; // time to download model and mini-batch, upload gradients

        DeviceInfo deviceInfo = new DeviceInfo(activity);

        // uncomment this for the PHT collection
        //int configurations[][] = { {1,0}, {2,0}, {3,0}, {4,0}, {0,1}, {0,2}, {0,3}, {0,4}, {1,1},
        //        {1,2}, {1,3}, {1,4}, {2,1}, {2,2}, {2,3}, {2,4}, {3,1}, {3,2}, {3,3}, {3,4}, {4,1},
        //        {4,2}, {4,3}, {4,4} };

        do {
            tempTime = System.currentTimeMillis();

            deviceInfo.deviceAvailableRam = deviceInfo.getAvailableMem();
            deviceInfo.storeBatteryLevel();
            deviceInfo.storeTemperature();
            deviceInfo.storeVolt();

            //deviceInfo.threadNumberLittle = configurations[numRequests / 10][0];
            //deviceInfo.threadNumberBig = configurations[numRequests / 10][1];

            DeviceBandwidthSampler mDeviceBandwidthSampler = DeviceBandwidthSampler.getInstance();
            mDeviceBandwidthSampler.startSampling();

            try {

                //deviceInfo.deviceCpuUsage = deviceInfo.getTotalCpuUsage();
                //nowEnergy = DeviceInfo.dumpBatteryStats(this.applicationInfo);
                deviceInfo.idleEnergy =  nowEnergy - pastEnergy;
                DeviceInfo.resetBatteryStats();
                startTime = System.currentTimeMillis();

                downloadLatency = this.downloadModel(postUrl, deviceInfo, tempTime);

                if (downloadLatency<0) {
                    Log.d("INFO", "Downloading the model failed! Pausing for 1 sec...");
                    TimeUnit.MILLISECONDS.sleep(1000);
                    continue;
                }

                computeGradient(mDeviceBandwidthSampler, deviceInfo);

                downloadLatency += calculator.getFetchModelTime() + calculator.getFetchMiniBatchTime();

                tempTime = System.currentTimeMillis();
                //double afterComputeGradientEnergy = DeviceInfo.dumpBatteryStats(this.applicationInfo);

                // Gradient POST request
                httpClient = new DefaultHttpClient();
                HttpParams httpParameters = httpClient.getParams();
                HttpConnectionParams.setConnectionTimeout(httpParameters, 5 * 1000);
                HttpConnectionParams.setSoTimeout        (httpParameters, 5 * 1000);
                uploadFile = new HttpPost(postUrl);
                builder = MultipartEntityBuilder.create(); // http://mvnrepository.com/artifact/org.apache.httpcomponents/httpmime/4.3.1
                builder.addTextBody("clientType", "Gradient", ContentType.TEXT_PLAIN);
                builder.addTextBody("clientID", clientName, ContentType.TEXT_PLAIN);

                // Gradient Post request
                builder.addBinaryBody("gradients", out.getBuffer());
                //TimeUnit.MILLISECONDS.sleep(1000); //for LR application since the memory usage was much faster than the memory release which was leading it to hang
                multipart = builder.build();
                uploadFile.setEntity(multipart);
                executeRequest = true;

                deviceInfo.deviceLatency += System.currentTimeMillis() - tempTime;
                //deviceInfo.deviceEnergy += DeviceInfo.dumpBatteryStats(this.applicationInfo) - afterComputeGradientEnergy;

                Log.d("INFO", "Device Latency: " + deviceInfo.deviceLatency + " ms");
                Log.d("INFO", "Size Latency: " + deviceInfo.sizeLatency + " ms");

                Log.d("INFO", "...Upload gradients...");
                tempTime = System.currentTimeMillis();

                response = httpClient.execute(uploadFile);
                executeRequest = false;
                responseEntity = response.getEntity();

                uploadLatency = System.currentTimeMillis() - tempTime;
                Log.d("INFO", "Upload latency: " + uploadLatency + " ms");
                networkLatency = downloadLatency + uploadLatency;
                deviceInfo.networkLatency = networkLatency;
                Log.d("INFO", "Network latency: " + networkLatency + " ms");

                rd = new BufferedReader(new InputStreamReader(responseEntity.getContent()));

                // Stats POST request <- NOT INCLUDING IT TO THE LATENCY MEASURES
                tempTime = System.currentTimeMillis();
                httpClient = new DefaultHttpClient();
                httpParameters = httpClient.getParams();
                HttpConnectionParams.setConnectionTimeout(httpParameters, 5 * 1000);
                HttpConnectionParams.setSoTimeout        (httpParameters, 5 * 1000);
                uploadFile = new HttpPost(postUrl);
                builder = MultipartEntityBuilder.create(); // http://mvnrepository.com/artifact/org.apache.httpcomponents/httpmime/4.3.1
                builder.addTextBody("clientType", "Stats", ContentType.TEXT_PLAIN);
                builder.addTextBody("clientID", clientName, ContentType.TEXT_PLAIN);

                String deviceName = DeviceName.getDeviceName();
                //String deviceName = Build.MODEL;
                //System.out.println("Device name: "+deviceName);
                String AndroidVersion = android.os.Build.VERSION.RELEASE;
                //System.out.println("Android version: "+AndroidVersion);
                String serialNumber = android.os.Build.SERIAL;

                String AndroidInfo = deviceName + "," + AndroidVersion + "," + serialNumber;

                builder.addTextBody("androidInfo", AndroidInfo, ContentType.TEXT_PLAIN);
                builder.addTextBody("stats", deviceInfo.getSerializableInfo());
                builder.addTextBody("batchSize", String.valueOf(calculator.getSize()));

                multipart = builder.build();
                uploadFile.setEntity(multipart);
                executeRequest = true;
                response = httpClient.execute(uploadFile);
                executeRequest = false;
                responseEntity = response.getEntity();

                //deviceInfo.deviceCpuUsage = deviceInfo.getTotalCpuUsage();
                //pastEnergy = DeviceInfo.dumpBatteryStats(this.applicationInfo);
                totalLatency = System.currentTimeMillis() - startTime;

                //some background load
                /*Thread[] tt = new Thread[deviceInfo.coreNumber];
                for(int i=0; i<deviceInfo.coreNumber; i++) {
                    tt[i] = new Thread(new Runnable() {
                        @Override
                        public void run() {
                            long x = 1;
                            for(long i = 0; i < 10000; i++)
                                for(long j = 0; j < 1000; j++)
                                    x = x*i*j;
                            Log.d("INFO", "Bla: " + x);
                        }
                    });
                    tt[i].start();
                }
                for(int i=0; i<deviceInfo.coreNumber; i++) {
                    tt[i].join();
                }*/

                StringBuffer result = new StringBuffer();
                String line = "";
                while ((line = rd.readLine()) != null) {
                    result.append(line);
                }
                Log.d("INFO", result.toString());
                Log.d("INFO", "Stats upload latency: " + String.valueOf(System.currentTimeMillis() - tempTime) + " ms");
            } catch (Exception e) {
                e.printStackTrace();
            }

            memInfo();
            Log.d("INFO", "Total latency: " + totalLatency + " ms");

            this.deviceEnergy = deviceInfo.deviceEnergy;
            this.sizeEnergy = deviceInfo.sizeEnergy;
            this.sizeLatency = deviceInfo.sizeLatency;
            this.deviceLatency = deviceInfo.deviceLatency;
            this.sizeLatency = deviceInfo.sizeLatency;

            //System.gc();
            //startTime = System.currentTimeMillis();
            numRequests++;
            Log.d("INFO", "Number of requests: " + numRequests);
            publishProgress();

        } while (!stop);

        Log.d("INFO", "STOPPED");
        return deviceInfo;
    }

    @Override
    protected void onPostExecute(DeviceInfo deviceInfo) {
        activity.receiveWorkerOrchestratorOutput(deviceInfo);
    }

    public void stop() {
        if (continueRequests) {
            Log.d("INFO", "CANNOT stop requests!");
            return;
        }
        // Make sure we clean up if the task is killed
        if (executeRequest) {
            uploadFile.abort();
        }
        stop = true;
    }

    public boolean hasStopped() {
        return stop;
    }

    @Override
    protected void onProgressUpdate(Void... values) {
        super.onProgressUpdate(values);
        stat1.setText("Latency: " + totalLatency + " ms" +
                //"\nDevice: " + deviceLatency + " ms" +
                //"\nNetwork Latency: " + networkLatency + " ms" +
                "\nComputation Latency: " + sizeLatency + " ms" +
                "\nComputation Energy: " + sizeEnergy + " mAh" +
                "\nNumber of requests: " + numRequests);
    }

    public static void memInfo() {
        Double allocated = new Double(Debug.getNativeHeapAllocatedSize()) / new Double((1048576));
        Double available = new Double(Debug.getNativeHeapSize()) / 1048576.0;
        Double free = new Double(Debug.getNativeHeapFreeSize()) / 1048576.0;
        DecimalFormat df = new DecimalFormat();
        df.setMaximumFractionDigits(2);
        df.setMinimumFractionDigits(2);
        Debug.MemoryInfo memoryInfo = new Debug.MemoryInfo();
        Debug.getMemoryInfo(memoryInfo);
        int total_dalvik = memoryInfo.getTotalPrivateClean() + memoryInfo.getTotalPrivateDirty() + memoryInfo.getTotalPss()
                + memoryInfo.getTotalSharedClean() + memoryInfo.getTotalSharedDirty() + memoryInfo.getTotalSwappablePss();

        Log.d("INFO", "debug. =================================");
        Log.d("INFO", "debug.heap native: allocated " + df.format(allocated) + "MB of " + df.format(available) + "MB (" + df.format(free) + "MB free)");
        Log.d("INFO", "debug.memory: allocated: " + df.format(new Double(Runtime.getRuntime().totalMemory() / 1048576)) + "MB of " + df.format(new Double(Runtime.getRuntime().maxMemory() / 1048576)) + "MB (" + df.format(new Double(Runtime.getRuntime().freeMemory() / 1048576)) + "MB free)");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Log.d("INFO", "Memory info (KB): " + Arrays.toString(memoryInfo.getMemoryStats().entrySet().toArray()));
        }
        Log.d("INFO", "Memory info (dalvik) (KB): " + total_dalvik);
    }
}
