cd $1
dir=$2
files=$(ls $dir)
length=${#files[@]}
i=1
for filename in $files
do
     python3.7 DeepM.py --single_class_path $dir$filename >> $3 
     echo "#\c"
i=`expr $i + 1`
done

echo "\n"
