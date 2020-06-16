/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.ActivityManager.MemoryInfo;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.ApplicationInfo;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.wifi.WifiManager;
import android.os.BatteryManager;
import android.os.Debug;
import android.util.Log;

import java.io.*;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import Jama.Matrix;

/**
 * Created by Mercury on 2016/11/18.
 */

public class DeviceInfo {

    static {
        System.loadLibrary("native-lib");
    }

    private ActivityManager activityManager;
    private ConnectivityManager conManager;
    private WifiManager wifiManager;
    private Activity activity;


    public double availMemory;
    public int runningProcess;
    public int coreNumber;
    //public double cacheSize;
    //public double cpuSpeed;
    public int threadNumberLittle = -1;
    public int threadNumberBig = -1;
    public double bogoMips;

    /**
     * Time to communicate with the MO
     */
    public double networkLatency = -1;
    /**
     * the size latency (l_S): mini-batch fetch time + gradient computation time
     */
    public double sizeLatency = -1;
    /**
     * Mean sizeLatency (l_S / batchsize):
     */
    public double meanSizeLatency = -1;
    /**
     * device latency (l_D): processing latency not dependent on mini-batch size and model size
     */
    public double deviceLatency = -1;

    public double batchSize = -1;
    public double bandwidth = -1;
    public double heapSize = -1;
    public double ramSize = -1;

    //private double netSpeed;

    public double sizeEnergy = 0;
    public double deviceEnergy = 0;
    public double idleEnergy = 0;

    public double deviceTotalRam = 0;
    public double deviceAvailableRam = 0;
    public double deviceCpuUsage = 0;

    public double temperature = 0;
    public double batteryLevel = 0;
    public double volt = 0;

    /**
     * features for parallel computation
     */
    public double[] cpuMaxFreq;
    public double[] cpuCurFreq;


    public DeviceInfo(Activity activity) {
        this.activityManager = (ActivityManager) activity.getSystemService(Context.ACTIVITY_SERVICE);
        this.conManager = (ConnectivityManager) activity.getSystemService(Context.CONNECTIVITY_SERVICE);
        this.wifiManager = (WifiManager) activity.getApplicationContext().getSystemService(Context.WIFI_SERVICE);

        this.activity = activity;

        this.initialize();
    }

    public static double getBogoMipsFromCpuInfo() {
        String result = null;
        String cpuInfo = readCPUinfo();
        String[] cpuInfoArray = cpuInfo.split(":");
        for (int i = 0; i < cpuInfoArray.length; i++) {
            if (cpuInfoArray[i].toLowerCase().contains("bogomips")) {
                result = cpuInfoArray[i + 1];
                break;
            }
        }
        if (result != null) result = result.trim();
        try {
            Scanner st = new Scanner(result);
            return st.nextDouble();//Double.parseDouble(result);
        } catch (Exception e) {
            return 38.8;
        }
    }

    public static double getCPUMHZFromCpuInfo() {
        String result = null;
        String cpuInfo = readCPUinfo();
        String[] cpuInfoArray = cpuInfo.split(":");
        for (int i = 0; i < cpuInfoArray.length; i++) {
            if (cpuInfoArray[i].contains("cpu MHz")) {
                result = cpuInfoArray[i + 1];
                break;
            }
        }
        if (result != null) {
            result = result.trim();
            Scanner st = new Scanner(result);
            return st.nextDouble();//Double.parseDouble(result);
        } else {
            cpuInfo = readSYSinfo();
            try {
                return Double.parseDouble(cpuInfo) / (double) 1000;
            } catch (Exception e) {
                return 1511; //Average CPU frequency
            }
        }

    }

    public static String readSYSinfo() {
        ProcessBuilder cmd;
        String result = "";
        InputStream in = null;
        try {
            String[] args = {"/system/bin/cat", "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"};
            cmd = new ProcessBuilder(args);
            Process process = cmd.start();
            in = process.getInputStream();
            byte[] re = new byte[1024];
            while (in.read(re) != -1) {
                //System.out.println(new String(re));
                result = result + new String(re);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (in != null)
                    in.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    public static String readCPUinfo() {
        ProcessBuilder cmd;
        String result = "";
        InputStream in = null;
        try {
            String[] args = {"/system/bin/cat", "/proc/cpuinfo"};
            cmd = new ProcessBuilder(args);
            Process process = cmd.start();
            in = process.getInputStream();
            byte[] re = new byte[1024];
            while (in.read(re) != -1) {
                //System.out.println(new String(re));
                result = result + new String(re);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (in != null)
                    in.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    private void initialize() {

        // collect memory info
        MemoryInfo mInfo = new MemoryInfo();
        this.activityManager.getMemoryInfo(mInfo);
        this.availMemory = mInfo.availMem / 1048576L;    // transfer byte to MB

        // collect process info
        //this.runningProcess = this.activityManager.getRunningAppProcesses().size();

        //collect number of running services
        this.runningProcess = getRunningServices();

        // collect the number of processor
        this.coreNumber = Runtime.getRuntime().availableProcessors();
        this.cpuMaxFreq = getCpuFrequencies(0);
        this.cpuCurFreq = getCpuFrequencies(1);
        //this.temperature = getTemperature();

        this.bogoMips = getBogoMipsFromCpuInfo();

        //THIS WORKS ONLY FOR REAL DEVICES
        //this.cpuSpeed = getCPUMHZFromCpuInfo();

        //THIS WORKS FOR EMULATOR DEVICES
        //this.cpuSpeed = getCPUMHZFromCpuInfo_emu();
        //this.cacheSize = cacheSizeFromCpuInfo();

        this.heapSize = new Double(Debug.getNativeHeapAllocatedSize()) / new Double((1048576));
        this.ramSize = new Double(Runtime.getRuntime().totalMemory() / 1048576);

        // collect network info
        NetworkInfo networkInfo = this.conManager.getActiveNetworkInfo();
        //this.netSpeed = NetworkParser.netSpeed(networkInfo.getType(), networkInfo.getSubtype(),
        //        this.wifiManager.getConnectionInfo());

        this.deviceTotalRam = getTotalMem();
    }

    public static int getRunningServices (){
        int nproc = -1;

        // get running processes
        //String proc = execute("ps -A | wc -l", false); // works only on rooted Android >= 8

        // get running services (activityManager.getRunningServices works only on Android <= 7; gives 0 for Android 8)
        String proc = execute("service list | wc -l", false);

        try {
            nproc = Integer.parseInt(proc.trim()) - 1; // service list gives one more line
        } catch (NumberFormatException e) {
            e.printStackTrace();
        }

        return nproc;
    }

    public String getSerializableInfo() {

        this.update();

        Matrix infoMatrix = new Matrix(1, 39);
        int i = 0;
        infoMatrix.set(0, i++, this.availMemory);

        infoMatrix.set(0, i++, this.runningProcess);
        infoMatrix.set(0, i++, this.coreNumber);
        //infoMatrix.set(0, i++, this.cpuSpeed);
        infoMatrix.set(0, i++, this.threadNumberLittle);
        infoMatrix.set(0, i++, this.threadNumberBig);
        infoMatrix.set(0, i++, this.bogoMips);

        //infoMatrix.set(0, i++,, this.cacheSize);
        //infoMatrix.set(0, i++,, 0.0);
        //infoMatrix.set(0, i++,, this.netSpeed);
        infoMatrix.set(0, i++, this.networkLatency);
        infoMatrix.set(0, i++, this.sizeLatency);
        infoMatrix.set(0, i++, this.meanSizeLatency);
        infoMatrix.set(0, i++, this.deviceLatency);
        infoMatrix.set(0, i++, this.heapSize);
        infoMatrix.set(0, i++, this.ramSize);
        infoMatrix.set(0, i++, this.bandwidth);
        infoMatrix.set(0, i++, this.batchSize);

        infoMatrix.set(0, i++, this.sizeEnergy);
        infoMatrix.set(0, i++, this.idleEnergy);
        infoMatrix.set(0, i++, this.deviceEnergy);

        infoMatrix.set(0, i++, this.deviceAvailableRam);
        infoMatrix.set(0, i++, this.deviceTotalRam);
        infoMatrix.set(0, i++, this.deviceCpuUsage);

        infoMatrix.set(0, i++, this.temperature);
        infoMatrix.set(0, i++, this.batteryLevel);
        infoMatrix.set(0, i++, this.volt);

        /**
         * assume that there are maximum 8 cores on mobile platforms
         */
        for (int j = 0; j < 8; j++){
            infoMatrix.set(0, i+j, this.cpuMaxFreq[j]);
        }
        i+=8;

        for (int j = 0; j < 8; j++){
            infoMatrix.set(0, i+j, this.cpuCurFreq[j]);
        }

        return MatrixOperation.printMatrix(infoMatrix);
    }

    public void update() {
        this.initialize();
    }

    public boolean isValid() {
        return (this.meanSizeLatency > 0);
    }


    // ---------------- Battery -------------------------

    public static void resetBatteryStats() {
        try {
            Process process = Runtime.getRuntime().exec("dumpsys batterystats --reset");
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            DeviceInfo.logBufferedReader(bufferedReader);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static double dumpBatteryStats(ApplicationInfo applicationInfo) {
        try {
            Process process = Runtime.getRuntime().exec("dumpsys batterystats");
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            return DeviceInfo.parseBatteryStats(bufferedReader, applicationInfo);

        } catch (IOException e) {
            e.printStackTrace();
        }

        return 0;
    }

    private static double parseBatteryStats(BufferedReader bufferedReader, ApplicationInfo applicationInfo) throws IOException {
        String s;
        String uid = getBatteryStatsUid(applicationInfo);
        while ((s = bufferedReader.readLine()) != null) {
            //System.out.println("Battery info: " + s);
            if (s.contains("Uid u" + uid + ":")) {
                String line = s.trim();
                String pattern = "^.* cpu\\=([0-9.]+).*$";
                // ^.* cpu\=([0-9.]+).*$
                //^.*\ ([0-9.]+).*$

                Pattern r = Pattern.compile(pattern);
                Matcher m = r.matcher(line);
                if (m.find()) {
                    return Double.parseDouble(m.group(1));
                }

                return 0;
            }
        }

        return 0;
    }

    private static String getBatteryStatsUid(ApplicationInfo applicationInfo) {
        String uid = String.valueOf(applicationInfo.uid);
        String uidFirstPart = Integer.toHexString(Integer.parseInt(uid.substring(0, 2)));
        if (uidFirstPart.length() == 1) {
            uidFirstPart = "0" + uidFirstPart;
        }
        String uidSecondPart = uid.substring(2);
        return uidFirstPart + Integer.parseInt(uidSecondPart);
    }

    private static void logBufferedReader(BufferedReader bufferedReader) throws IOException {
        String s;
        while ((s = bufferedReader.readLine()) != null) {
            Log.d("INFO", s);
        }
    }

    public void storeBatteryLevel() {
        IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        Intent batteryStatus = activity.getBaseContext().registerReceiver(null, ifilter);
        int level = batteryStatus.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
        int scale = batteryStatus.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
        this.batteryLevel = level / (float) scale;
    }



    public void storeVolt() {
        IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        Intent batteryStatus = activity.getBaseContext().registerReceiver(null, ifilter);
        this.volt = batteryStatus.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1);
    }

    // ---------------- Memory -------------------------
    public double getAvailableMem() {
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(mi);
        return mi.availMem / 0x100000L;
    }

    public double getTotalMem() {
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(mi);
        return mi.totalMem / 0x100000L;
    }

    // ---------------- CPU -------------------------
    static long prevWork = 1l;
    static long prevTotal = 1l;


    /**
     * Execute a shell command on the device
     *
     * @param cmd command to execute
     * @param root whether to execute the command with root access
     * @return the return of the command
     */
    private static synchronized String execute(String cmd, boolean root) {
        StringBuilder retour = new StringBuilder();
        try {
            Process proc;
            if (root) {
                proc = Runtime.getRuntime().exec("su");
            }
            else {
                 proc = Runtime.getRuntime().exec("sh");
            }
            DataOutputStream os = new DataOutputStream(proc.getOutputStream());
            os.writeBytes(cmd + "\n");
            os.flush();
            os.writeBytes("exit\n");
            os.flush();

            BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            int read;
            char[] buffer = new char[4096];
            while ((read = reader.read(buffer)) != -1) {
                retour.append(buffer, 0, read);
            }
            reader.close();

        } catch (java.io.IOException e) {
            e.printStackTrace();
        }

        return retour.toString();
    }

        // https://github.com/AntonioRedondo/AnotherMonitor/blob/5f6905d39f74919832b8ee33cfe8a952c5db1400/AnotherMonitor/src/main/java/org/anothermonitor/ServiceReader.java#L400
    public static double getTotalCpuUsage() {
        String stat = execute("cat /proc/stat", false);
        String[] stat_spl = stat.split("\\R", 2);
        String[] sa = stat_spl[0].split("[ ]+", 9);

        long user = Long.parseLong(sa[1]);
        long nice = Long.parseLong(sa[2]);
        long system = Long.parseLong(sa[3]);
        long idle = Long.parseLong(sa[4]);
        long iowait = Long.parseLong(sa[5]);
        long irq = Long.parseLong(sa[6]);
        long softirq = Long.parseLong(sa[7]);

        long tIdle = idle + iowait;
        long tWork = user + nice + system + irq + softirq;
        long tTotal = tIdle + tWork;

        //differentiate: actual value minus the previous one
        long totald = tTotal - prevTotal;
        long workd = tWork - prevWork;

        double CPU_perc = (double) workd /*/ totald*/;

        //System.out.println("CPUU: " + user + " " + nice + " " + system + " " + idle
        //        + " " + iowait + " " + irq + " " + softirq);
        System.out.println("CPU usage: " + CPU_perc * 100 + "%");

        prevWork = tWork;
        prevTotal = tTotal;

        return CPU_perc;
    }

    public double getTotalCpuUsageOLD() {
        float[] coreValues = new float[10];
        double totalCpu = 0;
        double numCores = getNumCores();
        for (byte i = 0; i < numCores; i++) {
            long time = System.currentTimeMillis();
            float coreValue = readCore(i);
            Log.d("INFO", "Cpu" + i + " " + coreValue + " " + (System.currentTimeMillis() - time) + "ms");

            coreValues[i] = coreValue;
            totalCpu += coreValue;
        }

        return totalCpu / numCores;
    }

    private float readCore(int i) {
        try {
            RandomAccessFile reader = new RandomAccessFile("/proc/stat", "r");
            for (int ii = 0; ii < i + 1; ++ii) {
                reader.readLine();
            }
            String load = reader.readLine();

            if (load.contains("cpu")) {
                String[] toks = load.split(" ");
                long work1 = Long.parseLong(toks[1]) + Long.parseLong(toks[2]) + Long.parseLong(toks[3]);
                long total1 = Long.parseLong(toks[1]) + Long.parseLong(toks[2]) + Long.parseLong(toks[3]) +
                        Long.parseLong(toks[4]) + Long.parseLong(toks[5])
                        + Long.parseLong(toks[6]) + Long.parseLong(toks[7]) + Long.parseLong(toks[8]);
                try {
                    Thread.sleep(200);
                } catch (Exception e) {
                }
                reader.seek(0);
                for (int ii = 0; ii < i + 1; ++ii) {
                    reader.readLine();
                }
                load = reader.readLine();

                if (load.contains("cpu")) {
                    reader.close();
                    toks = load.split(" ");
                    long work2 = Long.parseLong(toks[1]) + Long.parseLong(toks[2]) + Long.parseLong(toks[3]);
                    long total2 = Long.parseLong(toks[1]) + Long.parseLong(toks[2]) + Long.parseLong(toks[3]) +
                            Long.parseLong(toks[4]) + Long.parseLong(toks[5])
                            + Long.parseLong(toks[6]) + Long.parseLong(toks[7]) + Long.parseLong(toks[8]);
                    return (float) (work2 - work1) / ((total2 - total1));
                } else {
                    reader.close();
                    return 0;
                }
            } else {
                reader.close();
                return 0;
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        return 0;
    }

    private int getNumCores() {
        class CpuFilter implements FileFilter {
            @Override
            public boolean accept(File pathname) {
                if (Pattern.matches("cpu[0-9]+", pathname.getName())) {
                    return true;
                }
                return false;
            }
        }

        try {
            File dir = new File("/sys/devices/system/cpu/");
            File[] files = dir.listFiles(new CpuFilter());
            return files.length;
        } catch (Exception e) {
            return 1;
        }
    }

    public double[] getCpuFrequencies(int action) {
        double[] cpuFreq = new double[8];
        String file;

        switch (action) {
            /*get the max frequency of CPUS*/
            case 0:
                file = "scaling_max_freq";
                break;

            /* get the current freq of CPUS */
            case 1:
                file = "scaling_cur_freq";
                break;

            default:
                return null;
        }

        for (int i = 0; i < 8; i++) {

            String stat = execute("cat /sys/devices/system/cpu/cpu" + i + "/cpufreq/" + file, false);

            try {
                cpuFreq[i] = Double.parseDouble(stat.trim());
            } catch (NumberFormatException e) {
                cpuFreq[i] = 0;
                e.printStackTrace();
            }
            System.out.println("MAX freq " + i
                    + " " + cpuFreq[i]);
        }
        return cpuFreq;
    }

    public String getThermalZoneType(int zone) {
        String temp = execute("cat /sys/class/thermal/thermal_zone" + zone + "/type", false);
        return temp;
    }

    public void storeTemperature() {
        IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        Intent batteryStatus = activity.getBaseContext().registerReceiver(null, ifilter);
        this.temperature = batteryStatus.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1);
    }

    public double getTemperature() {
        int zone = 0;
        System.out.println(getThermalZoneType(zone));
        double temperature = 0;

        String temp = execute("cat /sys/class/thermal/thermal_zone" + zone + "/temp", false);
        try {
            temperature = Double.parseDouble(temp.trim());
        } catch (NumberFormatException e) {
            e.printStackTrace();
        }

        return temperature;
    }
}
