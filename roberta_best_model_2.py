
# Test script that can be run in python on non-linux machines
import os

# Roberta noans best
noans_lr          = 1e-4
noans_nheads      = 1
noans_num_layers  = 1
noans_d_model     = 200
noans_hidden_size = 200

# Roberta blank best
blank_lr          = 1e-3
blank_nheads      = 1
blank_num_layers  = 1
blank_d_model     = 128
blank_hidden_size = 128

data_file_noans = "roberta_noans_best_output_model_2.tsv"
print("starting roberta_noans best")
c_noans = "python transformer_2.py"
c_noans += " --max_epochs 20"
c_noans += " --data_path new_data/final_roberta_noans_eval_model_2.jsonl"
c_noans += " --lr %.4f" % (noans_lr)
c_noans += " --nhead %d" % (noans_nheads)
c_noans += " --num_layers %d" % (noans_num_layers)
c_noans += " --d_model %d" % (noans_d_model)
c_noans += " --hidden_size %d" % (noans_hidden_size)
c_noans += " --output_data_path best_output/%s" % (data_file_noans)
c_noans += " >> best_output/%s.csv" % ("roberta_noans_best_loss_model_2")
os.system(c_noans)

data_file_blank = "roberta_blank_best_output_model_2.tsv"
print("starting roberta_blank best")
c_blank = "python transformer_2.py"
c_blank += " --max_epochs 20"
c_blank += " --data_path new_data/final_roberta_blank_eval_model_2.jsonl"
c_blank += " --lr %.4f" % (blank_lr)
c_blank += " --nhead %d" % (blank_nheads)
c_blank += " --num_layers %d" % (blank_num_layers)
c_blank += " --d_model %d" % (blank_d_model)
c_blank += " --hidden_size %d" % (blank_hidden_size)
c_blank += " --output_data_path best_output/%s" % (data_file_blank)
c_blank += " >> best_output/%s.csv" % ("roberta_blank_best_loss_model_2")
os.system(c_blank)
