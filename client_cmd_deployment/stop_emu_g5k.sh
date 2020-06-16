#!/bin/bash


uniq $OAR_NODEFILE > ~/tf_nodes$OAR_JOBID

while read p; do
	echo "Stopping emulator on: "$p
  ssh $p "export ANDROID_SDK_HOME=$ANDROID_SDK_HOME; \
	~/opt/android-sdk-linux/platform-tools/adb -s emulator-5554 emu kill" < /dev/tty
  	sleep 2
	ssh $p "pkill -9 -f adb" < /dev/tty
	ssh $p "pkill -9 -f monkeyrunner" < /dev/tty
#	fuser -k -n tcp 5554" < /dev/tty
done < ~/tf_nodes$OAR_JOBID

rm ~/tf_nodes$OAR_JOBID

