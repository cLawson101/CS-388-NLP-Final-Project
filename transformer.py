import argparse
import json
from sklearn.model_selection import train_test_split
from utils.util import *
import numpy as np
import torch.nn as nn
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type = str, default = "testing")
    parser.add_argument("--data_path", required=True, type = str)
    parser.add_argument("--max_epochs", type=int, default = 10)
    parser.add_argument("--batch_size", type=int, default = 128)
    
    # Optimizer Stuff
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--lr', type=float, default=4e-4) # 8e-4
    parser.add_argument('--decay', type=float, default=0)
    
    # Turn on CUDA or not
    parser.add_argument("--cuda", type=bool, default = False)
    config = parser.parse_args().__dict__
    return config

def load_data(path):
    data = []
    label = []

    with open(path, 'r') as f:
        for line in f:
            case = json.loads(line)
            data.append(case)
            label.append(case['is_impossible'])
    return data, label

if __name__ == "__main__":
    config = parse_args()
    
    # LOAD DATA
    data, label = load_data(config["data_path"])

    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.33, random_state = 420)

    train_gen = BatchGen(data, label, config["batch_size"], True)

    # MODEL SPECIFIC STUFF
    model = None
    
    # OPTIMIZER 
    # TODO UNCOMMENT ONCE MODEL IS NOT NONE
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=config['lr'],
    #                              betas=(config['b1'], config['b2']),
    #                              eps=config['e'],
    #                              weight_decay=config['decay'])
    
    criterion = nn.CrossEntropyLoss()

    EPOCH = config["max_epochs"]


    if config["cuda"]:
        model.cuda()

    model.train()
    for e in range(EPOCH):
        for i, batch in enumerate(train_gen):
            optimizer.zero_grad()

            pa = batch["pred_ans"][0]
            q = batch["q"][0]
            pas = batch["pred_ans_sent"][0]
            y = batch["label"]

            prob = model(pa, q, pas)
            train_loss = torch.sum(criterion(prob, y))
            train_loss.backward()

            optimizer.step()
        
        train_gen.reset()
            
            

    