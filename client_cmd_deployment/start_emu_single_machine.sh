#!/bin/bash

if [ "$1" = "" ] || [ "$2" = "" ]; then
	echo -e "$0 <#emulators> <RAM_SIZE (in MB)>" 
  exit 1
fi


# start emulator in each node
i=1
for i in $(seq 1 $1); do
	echo "Starting emu$i"
	emulator -memory $2 -avd emu$i -no-window &
done 
