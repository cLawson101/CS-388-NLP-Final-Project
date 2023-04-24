#!/bin/sh
echo "Starting Tests"

for lr in 1e-4 1e-3
do
	for nh in 1 4
	do
		for nl in 1 4
		do
			for dm in 200 400
            do
                for hs in 200 400
                do
                    for d in 0 0.25
                    do
                        echo "Starting lr: $lr nh: $nh nl: $nl dm: $dm hs: $hs d: $d"
                        python transformer.py --lr ${lr} --nhead ${nh} --num_layers ${nl} --d_model ${dm} --hidden_size ${hs} --dropout ${d} >> temp_csv.csv
                    done
                done
            done
		done
	done
done
echo "Finished"
