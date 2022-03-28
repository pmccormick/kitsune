#!/bin/bash 
nvidia-smi 
profile_size=268435456
export LD_LIBRARY_PATH=/projects/kitsune/13.x/x86_64/lib 
for exe in vecadd.clang.x86_64 vecadd-forall.cuda.1.x86_64 vecadd-forall.cuda.2.x86_64 vecadd-forall.cuda.4.x86_64
do 
	echo "Profiling $exe..."
	ncu --set=full \
            --call-stack \
            --page details \
            --print-summary per-gpu \
            --details-all \
            --log-file $PWD/$exe-ncu.$profile_size.log \
            --export $exe.%i.$profile_size ./$exe $profile_size pre-launch  
done	

