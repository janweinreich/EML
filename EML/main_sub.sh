#!/bin/bash                                                                                             
#SBATCH --job-name=                    #This is the name of your job
#SBATCH --ntasks=24             # number of mpi tasks
#SBATCH --cpus-per-task=1       #This is the number of cores reserved
#SBATCH --nodes=1                   # number of compute nodes
#SBATCH --time=48:00:00        #T       his is the time that your task will run
#SBATCH --partition=long
# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=./slurm/myrun.o%j     #These are the STDOUT and STDERR files
#SBATCH --error=./slurm/myrun.e%j
ntrain=
ntest=
reptype=

date
hostname
here=`pwd` 
DATE=`date +%d-%m-%y`                               #name the folder in the scratch  
random=`date +%s | sha256sum | base64 | head -c30` 
 
work="/scratch/${USER}/${DATE}/${reptype}/${random}/${ntrain}/${ntest}" 
#work="${here}/calc/${USER}/${DATE}/${reptype}/${random}/${ntrain}/${ntest}" 
mkdir -p $work 

cp -r mp-spdz-0.2.5 $work

data_dir=${work}/mp-spdz-0.2.5/Player-Data
source_dir=${work}/mp-spdz-0.2.5/Programs/Source/
model_dir=${here}/qm9/data_${reptype}



cp CM_${reptype}.mpc ${source_dir}/CM_${reptype}.mpc

sed -i "/ntrain =/c\ntrain = ${ntrain}"  ${source_dir}/CM_${reptype}.mpc
cat ${model_dir}/train/MODEL_${ntrain}  > ${data_dir}/Input-P0-0


echo "TEST POINT" ${ntest}
sed -i "/testid =/c\testid = ${ntest}"  ${source_dir}/CM_${reptype}.mpc
cat ${model_dir}/test/X_QUERY_${ntest}  > ${data_dir}/Input-P1-0
cd ${work}/mp-spdz-0.2.5
./compile.py CM_${reptype}.mpc
sleep 3
touch out
./Scripts/mascot.sh CM_${reptype} >> out
sleep 3
echo "PYTHON/TEST"
cat ${model_dir}/test/PRED_${ntrain}_${ntest} >> out


cp out ${here}/results/${reptype}/out_${ntrain}_${ntest}
