#!/bin/bash

if [ "$1" = "" ] || [ "$2" = "" ]; then
	echo -e "usage with kadeploy reservation: \n \
	$ oarsub -C [job_id] \n \
	frontend:$ $0 <#nodes> <RAM_SIZE (in MB)>" 
  exit 1
fi

uniq $OAR_NODEFILE | head -$1 > ~/tf_nodes$OAR_JOBID

# start emulator in each node
i=1
while read p; do
	echo "Starting emulator on: "$p
  ssh $p "export ANDROID_SDK_HOME=$ANDROID_SDK_HOME; \
	~/opt/android-sdk-linux/emulator/emulator -memory $2 -avd emu$i -no-window" < /dev/tty &
#	~/opt/android-sdk-linux/tools/emulator -memory $2 -avd emu$i -no-window" < /dev/tty &
#	~/opt/android-sdk-linux/tools/emulator64-x86 -memory 2048 -avd emu$i -noskin -no-window -partition-size 512 -no-snapshot-load" < /dev/tty &
  i=$((i+1))
done < ~/tf_nodes$OAR_JOBID

rm ~/tf_nodes$OAR_JOBID

