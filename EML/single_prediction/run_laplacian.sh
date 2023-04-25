ntrain=32
testid=1
ntest=1


here=`pwd`
work=${here}/work

# remove old work directory
if [ -d "work" ]; then
  rm -rf work
fi

# create directory for results called work if not exists
if [ ! -d "work" ]; then
  mkdir work
fi

cp -r ../../mp-spdz-0.2.5 ${work}

mkdir ${work}/mp-spdz-0.2.5/Player-Data


data_dir=${work}/mp-spdz-0.2.5/Player-Data
source_dir=${work}/mp-spdz-0.2.5/Programs/Source
model_dir=../../input/data/data_CM

#convert relative path model_dir to absolute path
model_dir=$(cd ${model_dir}; pwd)


cp laplacian.mpc ${source_dir}/laplacian.mpc

sed -i "/ntrain =/c\ntrain = ${ntrain}"  ${source_dir}/laplacian.mpc
cat ${model_dir}/train/MODEL_${ntrain}  > ${data_dir}/Input-P0-0


echo "TEST POINT" ${ntest}
sed -i "/testid =/c\testid = ${testid}"  ${source_dir}/laplacian.mpc
sed -i "/ntest =/c\ntest = ${ntest}"  ${source_dir}/laplacian.mpc
cat ${model_dir}/test/X_QUERY_${testid}  > ${data_dir}/Input-P1-0

cd ${work}/mp-spdz-0.2.5
cp ${work}/mp-spdz-0.2.5/bin/Linux-avx2/mascot-party.x ${work}/mp-spdz-0.2.5

./compile.py laplacian.mpc
sleep 3
touch out
./Scripts/mascot.sh laplacian >> out
sleep 3
echo "PYTHON/TEST"  >> out
echo ${model_dir}
cat ${model_dir}/test/PRED_${ntrain}_${ntest} >> out

cat out