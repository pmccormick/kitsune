#!/bin/bash 
#
host_arch=$(/usr/bin/uname -o -m | /usr/bin/awk '{print $1}')
tstamp=$(/usr/bin/date +"%d-%m-%y:%H:%M")
gpuname=$(nvidia-smi --query-gpu name --format=csv,noheader | /usr/bin/head -1 | /usr/bin/sed 's/ /_/g')

outfile=lpb__$1__${gpuname}__${tstamp}.dat
echo "output file: $outfile"

echo "host architecture: $host_arch" > $outfile
echo "gpu: $gpuname" >> $outfile
echo "time stamp: $tstamp" >> $outfile
echo "ThreadsPerBlock,Time" >> $outfile

for tpb in {8..512..16}
do 
  export KITCUDA_THREADS_PER_BLOCK=$tpb
  echo "  running '$*' with $tpb threads-per-block"
  outstr=$(./$*)
  if [ $? -eq 0 ]; then
    entry=$(echo "${outstr}" | /usr/bin/grep '\*\*\*' | /usr/bin/awk '{print $2}' | /usr/bin/sed 's/,//g')
    echo "$tpb,$entry" >> $outfile
  else
    echo "  launch parameters potentially exceeded gpu resources."
    echo "  completing benchmark runs."
    break
  fi 
done 
echo ""
echo "done."

python3 ../plot-launch-bench.py $outfile



