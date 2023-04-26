
# Test script that can be run in python on non-linux machines
import os

# albert noans best
noans_lr          = 1e-4
noans_nheads      = 1
noans_num_layers  = 1
noans_d_model     = 200
noans_hidden_size = 400

# albert blank best
blank_lr          = 1e-4
blank_nheads      = 1
blank_num_layers  = 4
blank_d_model     = 200
blank_hidden_size = 200

data_file_noans = "albert_noans_best_output.tsv"
print("starting albert_noans best")
c_noans = "python transformer.py"
c_noans += " --max_epochs 20"
c_noans += " --data_path new_data/final_albert_noans_eval.jsonl"
c_noans += " --lr %.4f" % (noans_lr)
c_noans += " --nhead %d" % (noans_nheads)
c_noans += " --num_layers %d" % (noans_num_layers)
c_noans += " --d_model %d" % (noans_d_model)
c_noans += " --hidden_size %d" % (noans_hidden_size)
c_noans += " --output_data_path best_output/%s" % (data_file_noans)
c_noans += " >> best_output/%s.csv" % ("albert_noans_best_loss")
# os.system(c_noans)
print(c_noans)

data_file_blank = "albert_blank_best_output.tsv"
print("starting albert_blank best")
c_blank = "python transformer.py"
c_blank += " --max_epochs 20"
c_blank += " --data_path new_data/final_albert_blank_eval.jsonl"
c_blank += " --lr %.4f" % (blank_lr)
c_blank += " --nhead %d" % (blank_nheads)
c_blank += " --num_layers %d" % (blank_num_layers)
c_blank += " --d_model %d" % (blank_d_model)
c_blank += " --hidden_size %d" % (blank_hidden_size)
c_blank += " --output_data_path best_output/%s" % (data_file_blank)
c_blank += " >> best_output/%s.csv" % ("albert_blank_best_loss")
# os.system(c_blank)
print(c_blank)
