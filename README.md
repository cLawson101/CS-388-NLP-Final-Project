# CS-388 NLP READ VERIFY STUDY
By Christopher Lawson and Rishi Salem

## Reader Implementation
In this current repo we do not include the code to train our two reader models (RoBERTa and ALBERT) but this code is the exact same as our Assignment 3 code just that we are training on the squad_v2 dataset and the RoBERTa and ALBERT models.

An example of the script to run is `final_proj_run_v3.py` with instructions on how to run in `example_run.txt`.

## Data
All data that was output from the reader implementation was saved in the `new_data\` folder. but specifically the following files:
- `final_albert_blank_eval.jsonl`
- `final_albert_noans_eval.jsonl`
- `final_roberta_blank_eval.jsonl`
- `final_roberta_noans_eval.jsonl`

Extra files include the `unique_words.txt` files that just contain the vocabulary of the entire space and the `ALL_DATA_NEW_SPLIT` which was used to help with visualizations.

## Training Shell
This is included in the following files:
- transformer.py
- transformer_2.py
Where `transformer.py` was the shell to run Model 1 and `transformer_2.py` was the shell to run Model 2.

## Models
This is included in the following files:
- `models/model_1.py`
- `models/model_2.py`

Where each file represents their respective model.

## Utils
An important file is `utils/util.py` which contains the code necessary to set up the DataLoader correctly set up.

## Hypertuning
The scripts involved in hypertuning Model 1 were in:
- `albert_blank_model1.sh`
- `albert_noans_model1.sh`
- `roberta_blank_model1.sh`
- `roberta_noans_model1.sh`

and the best models were subsequently run with scripts:
- `albert_best_model.py`
- `roberta_blank_model.py`

The scripts involved in hypertuning Model 2 were in:
- `roberta_script_2.py`
- `albert_script_2.py`

and the best models were subsequently run with scripts:
- `roberta_best_model_2.py`
- `albert_best_model_2.py`

## Results
The best model results are in the `best_output/` folder. Where the files that contain `_loss_` represent the per epoch training loss and files that end with `.tsv` are the overall predictions.

The exception is with the model 2 results where these are saved in the `.csv` files that end with `_model_2`.