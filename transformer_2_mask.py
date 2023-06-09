import argparse
import json
from sklearn.model_selection import train_test_split
from utils.util import *
from models.model_1 import *
from models.model_2 import *
import numpy as np
import torch.nn as nn
import pandas as pd

from nltk import word_tokenize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type = str, default = "testing")
    # parser.add_argument("--data_path", required=True, type = str)
    parser.add_argument("--data_path", default = "new_data/final_albert_blank_eval.jsonl", type = str)
    parser.add_argument("--output_data_path", default = "output_data_2/final_albert_blank_output.tsv", type = str)
    parser.add_argument("--vocab_path", default = "new_data/unique_words_v3.txt", type = str)
    parser.add_argument("--max_epochs", type=int, default = 10)
    parser.add_argument("--max_char", type=int, default = 150)
    parser.add_argument("--batch_size", type=int, default = 150)
    parser.add_argument("--max_sent_len", type=int, default = 150)

    
    # Model Stuff
    parser.add_argument("--nhead", type=int, default = 1)
    parser.add_argument("--num_layers", type=int, default = 3)
    parser.add_argument("--d_model", type=int, default = 200)
    parser.add_argument("--hidden_size", type=int, default = 200)
    parser.add_argument("--dropout", type=int, default = 0)

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

def embed(sent, vocab, max_len):
    sent_emb = [0] * max_len
    for idx, word in enumerate(sent):
        word = word.encode("utf8")
        
        if idx == max_len:
            return sent_emb

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
    exit()

def evaluate(net, data_loader):
    final_acc_total = []
    final_acc_unans = []
    final_acc_answerable = []
    for i, batch in enumerate(data_loader):
        pa = batch["pred_ans"]
        q = batch["q"]
        pas = batch["pred_ans_sent"]
        y = batch["label"]

        # Tokenize pa, q, pas
        # TODO THIS IS AN ASSUMPTION TO TAKE THE LOWER OF EVERYTHING
        tok_pa = [word_tokenize(s.lower()) for s in pa]
        tok_q = [word_tokenize(s.lower()) for s in q]

        emb_pa = [embed(s, master_vocab, max_sent_len) for s in tok_pa]
        emb_q = [embed(s, master_vocab, max_sent_len) for s in tok_q]

        char_pa = [char_tokenize(sent, char_max) for sent in tok_pa]
        char_q = [char_tokenize(sent, char_max) for sent in tok_q]

        tensor_pa = torch.LongTensor(np.array(emb_pa))
        tensor_q = torch.LongTensor(np.array(emb_q))

        tensor_pa_char = torch.LongTensor(np.array(char_pa))
        tensor_q_char = torch.LongTensor(np.array(char_q))

        prob = net(tensor_pa, tensor_q, tensor_pa_char, tensor_q_char)
        predictions = np.argmax(prob.detach().numpy(), axis = 1)

        acc = sum(predictions[i] == y[i] for i in range(len(predictions))).item()
        final_acc_total.append(acc/100)

        unans_acc = sum((predictions[i] == y[i]) for i in range(len(predictions)) if not y[i]).item()
        total_unans = sum(1 for i in range(len(predictions)) if not y[i])
        final_acc_unans.append(unans_acc/100)

        answerable_acc = sum((predictions[i] == y[i]) for i in range(len(predictions)) if y[i]).item()
        total_ans = sum(1 for i in range(len(predictions)) if y[i])
        final_acc_answerable.append(answerable_acc/100)

    return avg_dataset(final_acc_answerable), avg_dataset(final_acc_unans), avg_dataset(final_acc_total)

def eval_final(net, data_loader):
    output_file = open(config["output_data_path"], "w", encoding="utf8")
    headers = "context\tquestion\tpred_answer\tlabel\tpred_verify\n"
    output_file.write(headers)
    for i, batch in enumerate(data_loader):
        pa = batch["pred_ans"]
        q = batch["q"]
        pas = batch["pred_ans_sent"]
        y = batch["label"]

        # Tokenize pa, q, pas
        # TODO THIS IS AN ASSUMPTION TO TAKE THE LOWER OF EVERYTHING
        tok_pa = [word_tokenize(s.lower()) for s in pa]
        tok_q = [word_tokenize(s.lower()) for s in q]

        emb_pa = [embed(s, master_vocab, max_sent_len) for s in tok_pa]
        emb_q = [embed(s, master_vocab, max_sent_len) for s in tok_q]

        char_pa = [char_tokenize(sent, char_max) for sent in tok_pa]
        char_q = [char_tokenize(sent, char_max) for sent in tok_q]

        tensor_pa = torch.LongTensor(np.array(emb_pa))
        tensor_q = torch.LongTensor(np.array(emb_q))

        tensor_pa_char = torch.LongTensor(np.array(char_pa))
        tensor_q_char = torch.LongTensor(np.array(char_q))

        prob = net(tensor_pa, tensor_q, tensor_pa_char, tensor_q_char)
        predictions = np.argmax(prob.detach().numpy(), axis = 1)

        for pred in range(len(predictions)):
            to_print = "{}\t{}\t{}\t{}\t{}\n".format(str(pas[pred]), str(q[pred]), str(pa[pred]), str(y[pred].item()), str(predictions[pred]))
            output_file.write(to_print)

def avg_dataset(dataset):
    if len(dataset):
        return sum(dataset) / len(dataset)
    else:
        # We could crash the program if the dataset (i.e. current total of unanswerable questions) is empty,
        # But we'll return an error code instead
        return -1

def char_tokenize(sent, char_max):
    char_rep = [0] * char_max
    i = 0
    for word in sent:
        for c in word:
            if i == char_max:
                # print("NOT ENOUGH CHARACTER SPACE")
                return char_rep
                # exit()
            char_rep[i] = ord(c) - 35
            i+=1
    return char_rep

if __name__ == "__main__":
    config = parse_args()
    
    master_vocab = load_vocab(config["vocab_path"])
    # pretty_print(master_vocab)

    char_max = config["max_char"]

    # LOAD DATA
    data, label = load_data(config["data_path"])

    # dataset_size = 11250
    dataset_size = 450

    data = data[:dataset_size]
    label = label[:dataset_size]

    train_X, test_X, train_y, test_y = train_test_split(data[:11250], label[:11250], test_size=(1/3), random_state = 420)

    train_gen = BatchGen(train_X, train_y, config["batch_size"], True)
    test_gen = BatchGen(test_X, test_y, config["batch_size"], True)
    
    # MODEL SPECIFIC STUFF
    model = Model2(
        vocab_size = len(master_vocab) + 2, 
        char_size = 8687+5, 
        d_model = config["d_model"], 
        hidden_size = config["hidden_size"], 
        dropout = config["dropout"],
        num_layers= config["num_layers"]
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
    max_sent_len = config["max_sent_len"]


    if config["cuda"]:
        model.cuda()
    
    

    model.train()
    for e in range(EPOCH):
        loss_epoch = 0.0
        print("%.4f,%2d,%2d,%2d,%2d,%.2f," % (config["lr"],
                                            config["nhead"],
                                            config["num_layers"],
                                            config["d_model"],
                                            config["hidden_size"],
                                            config["dropout"]
                                            ), end = "")
        for i, batch in enumerate(train_gen):
            optimizer.zero_grad()
            pa = batch["pred_ans"]
            q = batch["q"]
            y = batch["label"]

            # Tokenize pa, q, pas
            # TODO THIS IS AN ASSUMPTION TO TAKE THE LOWER OF EVERYTHING
            tok_pa = [word_tokenize(s.lower()) for s in pa]
            tok_q = [word_tokenize(s.lower()) for s in q]

            char_pa = [char_tokenize(sent, char_max) for sent in tok_pa]
            char_q = [char_tokenize(sent, char_max) for sent in tok_q]

            emb_pa = [embed(s, master_vocab, max_sent_len) for s in tok_pa]
            emb_q = [embed(s, master_vocab, max_sent_len) for s in tok_q]

            tensor_pa = torch.LongTensor(np.array(emb_pa))
            tensor_q = torch.LongTensor(np.array(emb_q))

            tensor_pa_char = torch.LongTensor(np.array(char_pa))
            tensor_q_char = torch.LongTensor(np.array(char_q))

            prob = model(tensor_pa, tensor_q, tensor_pa_char, tensor_q_char)

            loss = torch.sum(criterion(prob, y))
            loss_epoch += loss
            loss.backward()
            optimizer.step()

        train_gen.reset()
        print("%2d,%.4f," % (e, loss_epoch), end="")
        ans_acc, unans_acc, total_acc = evaluate(model, test_gen)
        test_gen.reset()
        print("%.4f, %.4f, %.4f" % (ans_acc, unans_acc, total_acc))
    eval_final(model, test_gen)
