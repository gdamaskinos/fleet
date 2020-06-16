/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import android.net.ConnectivityManager;
import android.net.wifi.WifiInfo;
import android.telephony.TelephonyManager;

/**
 * Created by Mercury on 2016/11/18.
 */

public class NetworkParser {


    public static double netSpeed(int type, int subType, WifiInfo wifiInfo){

        if(type==ConnectivityManager.TYPE_WIFI){
            return wifiInfo.getLinkSpeed();

        }else if(type==ConnectivityManager.TYPE_MOBILE){
            switch(subType){
                case TelephonyManager.NETWORK_TYPE_1xRTT:
                    return 0.1; // ~ 50-100 kbps
                case TelephonyManager.NETWORK_TYPE_CDMA:
                    return 0.06; // ~ 14-64 kbps
                case TelephonyManager.NETWORK_TYPE_EDGE:
                    return 0.1; // ~ 50-100 kbps
                case TelephonyManager.NETWORK_TYPE_EVDO_0:
                    return 1; // ~ 400-1000 kbps
                case TelephonyManager.NETWORK_TYPE_EVDO_A:
                    return 1.4; // ~ 600-1400 kbps
                case TelephonyManager.NETWORK_TYPE_GPRS:
                    return 0.1; // ~ 100 kbps
                case TelephonyManager.NETWORK_TYPE_HSDPA:
                    return 14; // ~ 2-14 Mbps
                case TelephonyManager.NETWORK_TYPE_HSPA:
                    return 1.7; // ~ 700-1700 kbps
                case TelephonyManager.NETWORK_TYPE_HSUPA:
                    return 21; // ~ 1-23 Mbps
                case TelephonyManager.NETWORK_TYPE_UMTS:
                    return 6; // ~ 400-7000 kbps

            /*
             * Above API level 7, make sure to set android:targetSdkVersion
             * to appropriate level to use these
             */
                case TelephonyManager.NETWORK_TYPE_EHRPD: // API level 11
                    return 2; // ~ 1-2 Mbps
                case TelephonyManager.NETWORK_TYPE_EVDO_B: // API level 9
                    return 5; // ~ 5 Mbps
                case TelephonyManager.NETWORK_TYPE_HSPAP: // API level 13
                    return 18; // ~ 10-20 Mbps
                case TelephonyManager.NETWORK_TYPE_IDEN: // API level 8
                    return 0.025; // ~25 kbps
                case TelephonyManager.NETWORK_TYPE_LTE: // API level 11
                    return 20; // ~ 10+ Mbps

                case TelephonyManager.NETWORK_TYPE_UNKNOWN:
                default:
                    return 10;
            }
        }else{
            return 10;
        }
    }

}
