#!/bin/bash

echo "****************************"
echo "Running Tapir OpenCilk Tests"
echo "****************************"

c=0

for i in test4/exe/*
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
			export CILK_NWORKERS=$thread
			
			echo ""
			echo "THREAD: $thread"
			$i
		done

		unset CILK_NWORKERS
		echo ""
	fi
done

#########################################################
c=0

echo ""
echo "LLIR Line Count"
echo ""

for i in test4/ll/*
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

for i in test4/exe/*
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

