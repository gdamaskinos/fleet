#!/bin/bash

if [ "$1" = "" ]; then
	echo -e "$0 <#emulators>" 
  exit 1
fi



for i in $(seq 1 $1); do
    	p=$((5554+($i-1)*2))
	echo "Stopping emulator-$p"
	adb -s emulator-$p emu kill
done

sleep 2
pkill -9 -f adb 
pkill -9 -f monkeyrunner
