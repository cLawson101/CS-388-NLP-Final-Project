
# Test script that can be run in python on non-linux machines
import os
learning_rates = [0.001, 0.005, 0.0001]
nums_epoch = [10, 15, 20, 30, 40, 50]
nums_layers = [1, 4, 8]
batch_sizes = [50, 100, 200, 300]
nums_heads = [1, 2, 4, 8]
d_models = [64, 128, 256, 512]
d_internals = [64, 128, 256, 512]

for lr in learning_rates:
    for nh in nums_heads:
        for nl in nums_layers:
            for dm in d_models:
                for hs in d_internals:
                    print("Starting lr: {} nh: {} nl: {} dm: {} hs: {}".format(lr, nh, nl, dm, hs))
                    command = "python transformer.py --data_path new_data/final_roberta_noans_eval.jsonl --lr {} --nhead {} --num_layers {} --d_model {} --hidden_size {} >> roberta_noans.csv".format(lr, nh, nl, dm, hs)
                    os.system(command)
