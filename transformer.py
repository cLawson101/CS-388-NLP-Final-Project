import argparse
import json
from sklearn.model_selection import train_test_split
from utils.util import *
from models.model_1 import *
import numpy as np
import torch.nn as nn
import pandas as pd

from nltk import word_tokenize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type = str, default = "testing")
    # parser.add_argument("--data_path", required=True, type = str)
    parser.add_argument("--data_path", default = "new_data/final_albert_blank_eval.jsonl", type = str)
    parser.add_argument("--vocab_path", default = "new_data/unique_words_v3.txt", type = str)
    parser.add_argument("--max_epochs", type=int, default = 1)
    parser.add_argument("--batch_size", type=int, default = 128)
    
    # Model Stuff
    parser.add_argument("--nhead", type=int, default = 1)
    parser.add_argument("--num_layers", type=int, default = 1)
    parser.add_argument("--d_model", type=int, default = 200)

    # Optimizer Stuff
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--lr', type=float, default=4e-4) # 8e-4
    parser.add_argument('--decay', type=float, default=0)
    
    # Turn on CUDA or not
    parser.add_argument("--cuda", type=bool, default = True)
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

def load_vocab(path):
    master_vocab = []
    with open(path, 'r', encoding="utf8", errors="ignore") as f:
        for line in f:
            line = str(line).encode("utf8")

            if line[-1] == 10:
                line = line[:-1]
            
            # TODO FIGURE OUT HOW TO READ IN THE LINES
            # print("Reading line:", line, "type:", type(line))
            master_vocab.append(line)
    return master_vocab

def embed(sent, vocab, max_len = 150):
    sent_emb = [0] * max_len
    for idx, word in enumerate(sent):
        word = word.encode("utf8")

        if word in vocab:
            word_emb = vocab.index(word)+2
            sent_emb[idx] = word_emb
        else:
            sent_emb[idx] = 1

    return sent_emb

# TODO deal with this
def pretty_print(vocab):
    for idx, word in enumerate(vocab):
        print("%5d, %20s" % (idx, word))

def evaluate(net, data_loader):
    for i, batch in enumerate(data_loader):
        pa = batch["pred_ans"]
        q = batch["q"]
        pas = batch["pred_ans_sent"]
        y = batch["label"]

        # Tokenize pa, q, pas
        # TODO THIS IS AN ASSUMPTION TO TAKE THE LOWER OF EVERYTHING
        tok_pa = [word_tokenize(s.lower()) for s in pa]
        tok_q = [word_tokenize(s.lower()) for s in q]
        tok_pas = [word_tokenize(s.lower()) for s in pas]

        emb_pa = [embed(s, master_vocab) for s in tok_pa]
        emb_q = [embed(s, master_vocab) for s in tok_q]
        emb_pas = [embed(s, master_vocab) for s in tok_pas]

        tensor_pa = torch.LongTensor(np.array(emb_pa))
        tensor_q = torch.LongTensor(np.array(emb_q))
        tensor_pas = torch.LongTensor(np.array(emb_pas))

        prob = net(tensor_pa, tensor_q, tensor_pas)
        predictions = np.argmax(prob.detach().numpy(), axis = 1)

        acc = sum(predictions[i] == y[i] for i in range(len(predictions))).item()
        print("TEST ACCURACY: %4.2f" % (acc))

if __name__ == "__main__":
    config = parse_args()
    
    master_vocab = load_vocab(config["vocab_path"])

    # LOAD DATA
    data, label = load_data(config["data_path"])

    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.33, random_state = 420)

    train_gen = BatchGen(train_X, train_y, config["batch_size"], True)
    test_gen = BatchGen(test_X, test_y, config["batch_size"], True)
    
    # MODEL SPECIFIC STUFF
    model = Model1(
        vocab_size = len(master_vocab) + 2, 
        num_positions = 150, 
        d_model = config["d_model"], 
        num_classes = 2,
        num_layers = config["num_layers"],
        nhead = config["nhead"]
    )
    
    # OPTIMIZER 
    # TODO UNCOMMENT ONCE MODEL IS NOT NONE
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 betas=(config['b1'], config['b2']),
                                 eps=config['e'],
                                 weight_decay=config['decay'])
    
    criterion = nn.CrossEntropyLoss()

    EPOCH = config["max_epochs"]


    if config["cuda"]:
        model.cuda()

    # TODO COMMENT THIS BACK IN ONCE READY
    model.train()
    for e in range(EPOCH):
        loss_epoch = 0.0
        for i, batch in enumerate(train_gen):
            optimizer.zero_grad()

            pa = batch["pred_ans"]
            q = batch["q"]
            pas = batch["pred_ans_sent"]
            y = batch["label"]

            # Tokenize pa, q, pas
            # TODO THIS IS AN ASSUMPTION TO TAKE THE LOWER OF EVERYTHING
            tok_pa = [word_tokenize(s.lower()) for s in pa]
            tok_q = [word_tokenize(s.lower()) for s in q]
            tok_pas = [word_tokenize(s.lower()) for s in pas]

            emb_pa = [embed(s, master_vocab) for s in tok_pa]
            emb_q = [embed(s, master_vocab) for s in tok_q]
            emb_pas = [embed(s, master_vocab) for s in tok_pas]

            tensor_pa = torch.LongTensor(np.array(emb_pa))
            tensor_q = torch.LongTensor(np.array(emb_q))
            tensor_pas = torch.LongTensor(np.array(emb_pas))
            
            prob = model(tensor_pa, tensor_q, tensor_pas)

            loss = torch.sum(criterion(prob, y))
            loss_epoch += loss.item()
            loss.backward()

            optimizer.step()
        print("Epoch %2d, Loss %.4f" % (e, loss_epoch))
        train_gen.reset()

print("Final Eval")

evaluate(model, test_gen)
            
            

    
