#!/bin/bash
reptype="CM"
mkdir -p ./jobscripts/
mkdir input
cp -r /home/jan/projects/EML/input/data ./input/data_${reptype}
for ntrain in 32 # 64 128 256 512 1024 2048 4096 8192 16384
do
        for ntest in 0 1 # {0..19..1}
        do
                cp main_sub.sh main_sub_${ntrain}_${ntest}.sh
                sed -i "/reptype=/c\reptype=${reptype}" main_sub_${ntrain}_${ntest}.sh
                sed -i "/#SBATCH --job-name=/c\#SBATCH --job-name=${reptype}_${ntrain}_${ntest}" main_sub_${ntrain}_${ntest}.sh
                sed -i "/ntrain=/c\ntrain=${ntrain}"  main_sub_${ntrain}_${ntest}.sh
                sed -i "/ntest=/c\ntest=${ntest}"  main_sub_${ntrain}_${ntest}.sh
                #sbatch < main_sub_${ntrain}_${ntest}.sh
		bash main_sub_${ntrain}_${ntest}.sh
                mv main_sub_${ntrain}_${ntest}.sh ./jobscripts/${reptype}
        done
done
