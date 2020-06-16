#!/bin/bash

handler()
{
  echo "Killed"
        # kill all background processes
  while read p; do
    echo "INFO: Terminating on $p..."
    echo $package
    ssh $p "export ANDROID_SDK_HOME=$ANDROID_SDK_HOME; \
    ~/opt/android-sdk-linux/platform-tools/adb uninstall com.androidsrc.client" < /dev/tty
    #~/opt/android-sdk-linux/platform-tools/adb shell am force-stop com.androidsrc.client" < /dev/tty
  done < ~/tf_nodes$OAR_JOBID
  
  rm ~/tf_nodes$OAR_JOBID

  exit
}

function pause(){
   read -p "$*"
}

trap handler SIGINT

if [ "$1" = "" ] || [ "$2" = "" ] || [ "$3" = "" ] || [ "$4" = "" ] ; then
        echo -e "usage: \nfrontend:$ $0 <#nodes> <runner.py> <apkFile.apk> <server_ip>" 
  exit 1
fi

server_ip=$4
exec=$(readlink -f $2)
#package=$(awk '/definitions/{getline; print $3}' $exec)

uniq $OAR_NODEFILE | head -$1 > ~/tf_nodes$OAR_JOBID

# install apk and add permissions
apk=$(readlink -f $3)
while read p; do
  echo "INFO: Reinstalling apk $exec on: "$p
  echo "INFO: If Failure [INSTALL_FAILED_UPDATE_INCOMPATIBLE] -> ctrl-c and rerun the script"
  ssh $p "export ANDROID_SDK_HOME=$ANDROID_SDK_HOME; \
    ~/opt/android-sdk-linux/platform-tools/adb -s emulator-5554 install -r $apk; \
    ~/opt/android-sdk-linux/platform-tools/adb shell pm grant com.androidsrc.client android.permission.DUMP" < /dev/tty & 
done < ~/tf_nodes$OAR_JOBID


wait

id=1
while read p; do
  echo "INFO: Running $exec on: "$p
  echo "INFO: Make sure all .setText() commands are commented out on the Client!"
  echo "INFO: To monitor output run: node$: adb logcat"
  ssh $p "export ANDROID_SDK_HOME=$ANDROID_SDK_HOME; export WORKER_ID=$id; \
	export JAVA_HOME=$JAVA_HOME; export PATH=$JAVA_HOME/bin:$PATH; \
	~/opt/android-sdk-linux/tools/bin/monkeyrunner $exec 1 $server_ip" < /dev/tty &
  id=$((id+1))
done < ~/tf_nodes$OAR_JOBID

wait
  
pause 'Apps are running. Press any key to exit the script or ctrl-c to kill them and exit...'
