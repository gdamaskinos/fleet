/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package coreComponents;

import android.app.Activity;
import android.content.ComponentCallbacks2;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.jaredrummler.android.device.DeviceName;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutionException;

import utils.DeviceInfo;


public class MainActivity extends Activity implements ComponentCallbacks2 {

    private EditText clientID, sleepTime, serverIP, serverPort;
    private Button buttonStop, buttonStart, buttonSingle;
    private TextView stat1;
    private WorkerOrchestrator myClient;
    private boolean stop = false;
    private DeviceInfo deviceInfoOutput;
    private boolean testMode = false; //for automated tests


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        serverIP = (EditText) findViewById(R.id.serverIP);
        // !! Remove default values
//		serverIP.setText("83.212.82.133"); // Server1
//		serverIP.setText("83.212.126.13"); // Server2
//		serverIP.setText("10.0.2.2"); // localhost
//		serverIP.setText("128.178.154.19"); // lpdquad server
//		serverIP.setText("128.178.154.42"); // lpdquatro2 server
//      serverIP.setText("192.168.1.26"); // localhost

        serverPort = (EditText) findViewById(R.id.serverPort);
//		serverPort.setText("9992");

        clientID = (EditText) findViewById(R.id.clientID);
		clientID.setHint("Enter Your Name here");
//        clientID.setText("DUMMY1");

        sleepTime = (EditText) findViewById(R.id.sleepTime);
//        sleepTime.setText("0");

        buttonStart = (Button) findViewById(R.id.startButton);
        buttonStop = (Button) findViewById(R.id.stopButton);
        buttonStop.setEnabled(false);
        buttonSingle = (Button) findViewById(R.id.singleButton);


        if (testMode == true){

            serverIP.setText("128.178.154.19");
            serverPort.setText("9992");
            clientID.setText(DeviceName.getDeviceName());
            sleepTime.setText("0");

            serverIP.setFocusable(false);
            serverPort.setFocusable(false);
            clientID.setFocusable(false);
            sleepTime.setFocusable(false);
            buttonStart.setEnabled(false);
            buttonStop.setEnabled(false);
            buttonSingle.setEnabled(false);

            buttonStart.postDelayed(new Runnable() {
                @Override
                public void run() {
                    buttonStart.performClick();
                }
            }, 1000);



        }


        stat1 = (TextView) findViewById(R.id.stat1);

        // FIXME new NDK doesn't support armeabi see github issues/35
//        Log.d("INFO", "Checking nd4j...");
//        INDArray arr1 = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
//        Log.d("INFO", arr1.toString());
//        Log.d("INFO", "Check OK");
//        Log.d("INFO", "MaxMemory: " + FileUtils.byteCountToDisplaySize(Runtime.getRuntime().maxMemory()));

//		double[][] data = new double[10][10];
//		System.out.println(data[2][2]);
//		Nd4j.getCompressor().setDefaultCompression("FLOAT16");
//		NDArray array = (NDArray) Nd4j.getCompressor().compress(Nd4j.create(data));
//		NDArray out = (NDArray) Nd4j.getCompressor().decompress(array);
//		System.out.println(out);

        byte[] inBuffer = "THIS IS A TESTA".getBytes();
        Log.d("INFO", "Sending: " + new String(inBuffer, StandardCharsets.UTF_8));

        byte[] outBuffer = transferBytes(inBuffer);
        Log.d("INFO", "Received: " + new String(outBuffer, StandardCharsets.UTF_8));



        buttonStop.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.d("INFO", "Stop Button Clicked");

                myClient.stop();
                buttonStart.setEnabled(true);
                buttonStop.setEnabled(false);

            }
        });


        /**
         * Execute one request
         */
        buttonSingle.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View arg0) {
                Log.d("INFO", "Single Button Clicked");
                makeSingleRequest();
            }
        });

        /**
         * Execute requests continuously
         */
        buttonStart.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View arg0) {
                Log.d("INFO", "Start Button Clicked");

                if (!testMode) {
                    buttonStart.setEnabled(false);
                    buttonStop.setEnabled(true);
                }

                if (myClient != null && !myClient.hasStopped()) {
                    myClient.stop();
                }


                //onTrimMemory(TRIM_MEMORY_UI_HIDDEN);
                myClient = new WorkerOrchestrator(serverIP.getText().toString(), Integer.parseInt(serverPort.getText().toString()), clientID.getText().toString(),
                        Integer.parseInt(sleepTime.getText().toString()),
                        stat1, MainActivity.this, getApplicationInfo());
                myClient.execute();
                clearCache();
                deleteDirectory();
                //onTrimMemory(TRIM_MEMORY_UI_HIDDEN);
            }
        });
    }

    public native byte[] transferBytes(byte[] inBuffer);

    static {
        System.loadLibrary("native-lib");
    }

    private void makeSingleRequest() {
        if (myClient != null && !myClient.hasStopped()) {
            myClient.stop();
        }

       myClient = new WorkerOrchestrator(serverIP.getText().toString(), Integer.parseInt(serverPort.getText().toString()), clientID.getText().toString(),
                Integer.parseInt(sleepTime.getText().toString()),
                stat1, MainActivity.this, getApplicationInfo());

        myClient.stop();

        try {
            myClient.execute().get();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }

        clearCache();
        deleteDirectory();
    }

    public void receiveWorkerOrchestratorOutput(DeviceInfo deviceInfo) {
        this.deviceInfoOutput = deviceInfo;
    }


    /**
     * Release memory when the UI becomes hidden or when system resources become low.
     *
     * @param level the memory-related event that was raised.
     */
    public void onTrimMemory(int level) {

        // Determine which lifecycle or system event was raised.
        switch (level) {

            case ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN:

                /*
                   Release any UI objects that currently hold memory.

                   The user interface has moved to the background.
                */

                break;

            case ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE:
            case ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW:
            case ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL:

                /*
                   Release any memory that your app doesn't need to run.

                   The device is running low on memory while the app is running.
                   The event raised indicates the severity of the memory-related event.
                   If the event is TRIM_MEMORY_RUNNING_CRITICAL, then the system will
                   begin killing background processes.
                */

                break;

            case ComponentCallbacks2.TRIM_MEMORY_BACKGROUND:
            case ComponentCallbacks2.TRIM_MEMORY_MODERATE:
            case ComponentCallbacks2.TRIM_MEMORY_COMPLETE:

                /*
                   Release as much memory as the process can.

                   The app is on the LRU list and the system is running low on memory.
                   The event raised indicates where the app sits within the LRU list.
                   If the event is TRIM_MEMORY_COMPLETE, the process will be one of
                   the first to be terminated.
                */

                break;

            default:
                /*
                  Release any non-critical data structures.

                  The app received an unrecognized memory level value
                  from the system. Treat this as a generic low-memory message.
                */
                break;
        }

    }


    public void clearCache() {
        File cache = getCacheDir();
        File appDir = new File(cache.getParent());
        if (appDir.exists()) {
            String[] children = appDir.list();
            for (String s : children) {
                if (!s.equals("lib")) {
                    deleteDir(new File(appDir, s));
                    Log.i("TAG", "**************** File /data/data/APP_PACKAGE/" + s + " DELETED *******************");
                }
            }
        }
    }

    public static boolean deleteDir(File dir) {
        if (dir != null && dir.isDirectory()) {
            String[] children = dir.list();
            for (int i = 0; i < children.length; i++) {
                boolean success = deleteDir(new File(dir, children[i]));
                if (!success) {
                    return false;
                }
            }
        }
        return dir.delete();
    }

    void deleteDirectory() {
        try {
            PackageManager m = getPackageManager();
            String path = getPackageName();
            PackageInfo p = m.getPackageInfo(path, 0);
            path = p.applicationInfo.dataDir;
            Runtime.getRuntime().exec(String.format("rm -rf %s", path));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
