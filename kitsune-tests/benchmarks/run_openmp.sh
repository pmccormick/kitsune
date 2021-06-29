#!/bin/bash

echo "****************************"
echo "Running OpenMP Tests"
echo "****************************"

c=0

for i in test2/exe/*
do
	if [[ $c == 0 ]]
	then
		echo ""
		echo "=============================="
		echo "Test: `basename $i`"
	fi

	if [[ $c == 2 ]]
	then c=0
	else c=$((c+1))
	fi

	if [[ $i =~ \serial$ ]]
	then
		echo ""
		echo "SERIAL"
		
		$i
	else
		echo ""
		echo "DEFAULT"
		$i

		for thread in 1 2 4 8 16 32 64
		do
			export OMP_NUM_THREADS=$thread
			
			echo ""
			echo "THREAD: $thread"
			$i
		done

		unset OMP_NUM_THREADS
		echo ""
	fi
done

#########################################################
c=0

echo ""
echo "LLIR Line Count"
echo ""

for i in test2/ll/*
do
	if [[ $c == 0 ]]
	then
		echo ""
	fi

	wc -l $i

	if [[ $c == 2 ]]
	then c=0
	else c=$((c+1))
	fi
done

#########################################################
c=0

echo ""
echo "EXE Size"
echo ""

for i in test2/exe/*
do
	if [[ $c == 0 ]]
	then
		echo ""
	fi

	size=`stat -c %s $i`
	echo "`basename $i`: $size"

	if [[ $c == 2 ]]
	then c=0
	else c=$((c+1))
	fi
done

echo ""
echo "Done"

