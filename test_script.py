
# Test script that can be run in python on non-linux machines
import os
datasets = ["roberta_noans", "roberta_blank", "albert_noans", "albert_blank"]
learning_rates = [0.001, 0.005, 0.0001]
nums_epoch = [10, 15, 20, 30, 40, 50]
nums_layers = [1, 4, 8]
batch_sizes = [50, 100, 200, 300]
nums_heads = [1, 2, 4, 8]
d_models = [64, 128, 256, 512]
d_internals = [64, 128, 256, 512]
for dataset in datasets:
    for lr in learning_rates:
        for nh in nums_heads:
            for nl in nums_layers:
                for dm in d_models:
                    for hs in d_internals:
                        data_file = "{}_{}_{}_{}_{}_{}_output.tsv".format(dataset, lr, nh, nl, dm, hs)
                        print("Starting lr: {} nh: {} nl: {} dm: {} hs: {}".format(lr, nh, nl, dm, hs))
                        command = "python transformer.py --data_path new_data/final_{}_eval.jsonl --lr {} --nhead {} --num_layers {} --d_model {} --hidden_size {} --output_data_path output_data/{} >> test_output/{}.csv".format(dataset, lr, nh, nl, dm, hs, data_file, dataset)
                        os.system(command)
