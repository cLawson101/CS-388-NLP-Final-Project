import torch
import random

class BatchGen(object):
    def __init__(self, raw_data, label, batch_size, is_training):
        self.total_num = len(label)

        self.data = raw_data
        self.label = label

        self.batch_size = batch_size

        self.is_training = is_training

        if is_training:
            indices = list(range(self.total_num))
            random.shuffle(indices)
            self.data = [self.data[idx] for idx in indices]

        self.batches = [self.data[i: i+batch_size] for i in range(0, self.total_num, batch_size)]
        self.offset = 0

    def reset(self):
        if self.is_training:
            indices = list(range(len(self.batches)))
            random.shuffle(indices)
            self.batches = []
        self.offset = 0

    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        batch_size = self.batch_size
        
        while self.offset < len(self):
            batch = self.batches[self.offset]
            batch_dict = {
                "pred_ans": [],
                "q": [],
                "pred_ans_sent": []
            }
            N = batch_size

            label_tensor = torch.LongTensor(N).fill_(0)

            for i, case in enumerate(batch):
                batch_dict["pred_ans"].append(case["predicted_answer"])
                batch_dict["q"].append(case["question"])
                batch_dict["pred_ans_sent"].append(case["predicted_sentence"])
                cur_label = case["is_impossible"]
                label_tensor[i] = torch.LongTensor([cur_label])

            self.offset += 1

            batch_dict["label"] = label_tensor
            yield batch_dict