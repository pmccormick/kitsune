#!/bin/bash

if [[ -d ./test1 ]] ; then
    rm -rf ./test1
fi

if [[ -d ./test2 ]] ; then
    rm -rf ./test2
fi

if [[ -d ./test3 ]] ; then
    rm -rf ./test3
fi

if [[ -d ./test4 ]] ; then
    rm -rf ./test4
fi

mkdir test1
mkdir test2
mkdir test3
mkdir test4

mkdir test{1,2,3,4}/exe
mkdir test{1,2,3,4}/ll

O_LEVEL="-O2"

CFLAGS1="$O_LEVEL -I./ -I$LANL_INSTALL/kokkos2/include"
KOKKOS_FLAGS1="-lkokkoscore -L$LANL_INSTALL/kokkos2/lib64 -ldl"

CFLAGS2="-fopenmp $O_LEVEL -I./ -I$LANL_INSTALL/include"
KOKKOS_FLAGS2="-L$LANL_INSTALL/lib64 -lkokkoscore -ldl"

CFLAGS3="-I./ -I$LANL_INSTALL/include -fkokkos -fkokkos-no-init -ftapir=serial -fopenmp $O_LEVEL"
CFLAGS4="-I./ -I$LANL_INSTALL/include -fkokkos -fkokkos-no-init -ftapir=opencilk -fopenmp $O_LEVEL"

function compile() {
    for j in 1 2 3 4
    do
        EXE_FOLDER="test$j/exe"
        LL_FOLDER="test$j/ll"

        if [[ $j == 1 ]]
        then
            set -x
            $OCC $CFLAGS1 $1 -o "$EXE_FOLDER/`basename $1 .cpp`" $KOKKOS_FLAGS1
            $OCC $CFLAGS1 $1 -o "$LL_FOLDER/`basename $1 .cpp`.ll" -S -emit-llvm
            set +x
        
        elif [[ $j == 2 ]]
        then
            set -x
            $OCC $CFLAGS2 $1 -o "$EXE_FOLDER/`basename $1 .cpp`" $KOKKOS_FLAGS2
            $OCC $CFLAGS2 $1 -o "$LL_FOLDER/`basename $1 .cpp`.ll" -S -emit-llvm
            set +x
        
        elif [[ $j == 3 ]]
        then
            set -x
            $OCC $CFLAGS3 $KOKKOS_FLAGS3 $1 -o "$EXE_FOLDER/`basename $1 .cpp`"
            $OCC $CFLAGS3 $1 -o "$LL_FOLDER/`basename $1 .cpp`.ll" -S -emit-llvm
            set +x
        
        elif [[ $j == 4 ]]
        then
            set -x
            $OCC $CFLAGS4 $KOKKOS_FLAGS4 $1 -o "$EXE_FOLDER/`basename $1 .cpp`"
            $OCC $CFLAGS4 $1 -o "$LL_FOLDER/`basename $1 .cpp`.ll" -S -emit-llvm
            set +x
        fi
    done
}

###########################################################################################
## Serial tests

echo "Building Serial..."
for i in serial/*.cpp
do
    compile $i    
done

###########################################################################################
## Forall tests

echo "Building Forall..."
for i in forall/*.cpp
do
    compile $i
done

###########################################################################################
## Kitsunes tests

echo "Building Parallel..."
for i in kokkos/*.cpp
do
    compile $i
done

echo "Done"
