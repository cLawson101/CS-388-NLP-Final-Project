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
                    echo "Starting lr: $lr nh: $nh nl: $nl dm: $dm hs: $hs"
                    python transformer.py --data_path new_data/final_albert_blank_eval.jsonl --lr ${lr} --nhead ${nh} --num_layers ${nl} --d_model ${dm} --hidden_size ${hs} >> albert_blank.csv
                done
            done
		done
	done
done
echo "Finished"
