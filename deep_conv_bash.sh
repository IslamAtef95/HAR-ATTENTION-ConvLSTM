#!/bin/bash
echo $1
INPUT=$1
filename=$2
counter=0
OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read prid dropout lr_decay attention_dropout out_dir
do
	echo "dropout : $dropout"
	echo "lr_decay : $lr_decay"
	echo "attention_dropout : $attention_dropout"
	echo "prid :  $prid"
        echo "out dir : $out_dir"
        python $filename -id $prid -d $dropout -lr_decay $lr_decay -ad $attention_dropout -out $out_dir &
        echo  "python $filename -id $prid -d $dropout -lr_decay $lr_decay -ad $attention_dropout -out $out_dir"

        counter=$((counter + 1))
        mod=$((counter%5))

        #echo $mod
        #echo $counter
        if [ $mod -ne 0 ]
        then
        continue
        fi
        sleep 1
        wait

done < $INPUT
IFS=$OLDIFS
echo "All Processes Done"
