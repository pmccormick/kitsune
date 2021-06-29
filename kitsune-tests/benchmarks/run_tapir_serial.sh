#!/bin/bash

echo "****************************"
echo "Running Serial Tapir Tests"
echo "****************************"

c=0

for i in test3/exe/*
do
	if [[ $c == 0 ]]
	then
		echo ""
		echo "Test: `basename $i`"
	fi

	if [[ $c == 2 ]]
	then c=0
	else c=$((c+1))
	fi

	$i
done

#########################################################
c=0

echo ""
echo "LLIR Line Count"
echo ""

for i in test3/ll/*
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

for i in test3/exe/*
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

