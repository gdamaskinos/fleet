#!/bin/bash

handler()
{
  echo "Killed"
        # kill all background processes
  echo "INFO: Terminating on $emuSerial..."
  echo $package
  adb -s emulator-$emuSerial uninstall com.androidsrc.client
  #~/opt/android-sdk-linux/platform-tools/adb shell am force-stop com.androidsrc.client" < /dev/tty
  exit
}

function pause(){
   read -p "$*"
}

trap handler SIGINT

if [ "$1" = "" ] || [ "$2" = "" ] || [ "$3" = "" ] || [ "$4" = "" ] || [ "$5" = "" ]; then
  echo -e "usage: \nfrontend:$ $0 <emu_serial (e.g., 5554)> <runner.py> <apkFile.apk> <server_ip> <server_port> <rerun_timeout_secs>" 
  exit 1
fi

emuSerial=$1
runner=$(readlink -f $2)
apk=$(readlink -f $3)
serverIP=$4
serverPort=$5
timeout=$6

while true; do

# install apk and add permissions
echo "INFO: Reinstalling $apk on: $emuSerial"
echo "INFO: If Failure [INSTALL_FAILED_UPDATE_INCOMPATIBLE] -> ctrl-c and rerun the script"
adb -s emulator-$emuSerial install -r $apk
adb -s emulator-$emuSerial shell pm grant com.androidsrc.client android.permission.DUMP

wait

echo "INFO: Running $runner on: $emuSerial"
echo "INFO: Make sure all .setText() commands are commented out on the Client!"
echo "INFO: To monitor output run: node$: adb logcat"
monkeyrunner $runner $emuSerial $serverIP $serverPort

wait

echo "App is running; Kill this script with -INT to stop the app. Sleeping for $timeout seconds before reinstalling it ..."
sleep $timeout

done
#pause 'Apps are running. Press any key to exit the script or ctrl-c to kill them and exit...'
