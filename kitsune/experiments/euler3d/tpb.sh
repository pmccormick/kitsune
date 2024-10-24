

for tpb in $(seq 128 128 1024);
do
    echo "$tpb (x launch codegen)"
     HSA_XNACK=1 && kit++ -DTPB=$tpb -fvectorize -mprefer-vector-width=64 -fno-rounding-math -std=c++17 -fno-exceptions -ftapir=hip -O3 -mllvm -hipabi-opt-level=3 -mllvm -hipabi-arch=gfx90a -fuse-ld=lld -o euler3d-attr-forall.hip.x86_64 euler3d-attr-forall.cpp -Xlinker -rpath=/projects/kitsune/x86_64/19.x/lib --gcc-install-dir=/projects/opt/centos8/x86_64/gcc/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/
     numactl --cpunodebind=0 --membind=0 --physcpubind=4 ./euler3d-attr-forall.hip.x86_64 fvcorr.domn.193K 16000
 
    echo "$tpb (y launch codegen)"
     HSA_XNACK=1 && kit++ -DTPB=$tpb -fvectorize -mprefer-vector-width=64 -fno-rounding-math -std=c++17 -fno-exceptions -ftapir=hip -O3 -mllvm -hipabi-opt-level=3 -mllvm -hipabi-arch=gfx90a -fuse-ld=lld -o euler3d-attr-forall.hip.x86_64 euler3d-attr-forall.cpp -Xlinker -rpath=/projects/kitsune/x86_64/19.x/lib --gcc-install-dir=/projects/opt/centos8/x86_64/gcc/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/ -mllvm -hipabi-y-launch
     numactl --cpunodebind=0 --membind=0 ./euler3d-attr-forall.hip.x86_64 fvcorr.domn.193K 16000
    
done 

echo "kokkos comparison"
./euler3d-kokkos.hipcc.x86_64 fvcorr.domn.193K  16000 
	   

