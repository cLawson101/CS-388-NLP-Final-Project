import json
import csv
import pandas as pd

def add_answer_sentence(input_file, output_file):
    input = list(input_file)
    output = list()
    for example in input:
        example = json.loads(example)
        paragraph = example['context']
        answer_start = paragraph.index(example['predicted_answer'])
        sent_start = answer_start
        sent_end = answer_start
        while sent_start > 0 and paragraph[sent_start - 1] != ".":
            sent_start -= 1
        while sent_end < len(paragraph) and paragraph[sent_end] != ".":
            sent_end += 1
        print(paragraph[sent_start:sent_end])
        example['predicted_sentence'] = paragraph[sent_start:sent_end]
        output.append(example)
    json.dump(output, output_file)

#files = ["albert_blank_eval.jsonl", "albert_noans_eval.jsonl", "roberta_blank_eval.jsonl", "roberta_noans_eval.jsonl"]
#for file_name in files:
#    input_file = open(file_name)
#    output_file = open("edited_" + file_name, "+w")
#    add_answer_sentence(input_file, output_file)

def get_ans_sentence(example):
    #add this python dict to json array
    paragraph = example['context']
    answer_start = paragraph.index(example['predicted_answer'])
    sent_start = answer_start
    sent_end = answer_start
    while sent_start > 0 and paragraph[sent_start - 1] != ".":
        sent_start -= 1
    while sent_end < len(paragraph) and paragraph[sent_end] != ".":
        sent_end += 1
    example['predicted_sentence'] = paragraph[sent_start:sent_end]
    return example['predicted_sentence']

def get_impossiblity(example):
    #add this python dict to json array
    return 1 - example['has_answer']

def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []
    df = pd.read_csv(csvFilePath)
    df['predicted_sentence'] = df.apply(get_ans_sentence, axis=1)
    df['is_impossible'] = df.apply(get_impossiblity, axis=1)
    df.to_json(jsonFilePath, orient="records")
    return
    jsonArray = []
    df = pd.read_csv(csvFilePath)
    df['predicted_sentence'] = df.apply(get_ans_sentence, axis=0)
    df.to_csv(output_csv, index=False)
    #read csv file
    with open(csvFilePath, encoding='utf-8-sig') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for example in csvReader: 
            #add this python dict to json array
            paragraph = example['context']
            answer_start = paragraph.index(example['predicted_answer'])
            sent_start = answer_start
            sent_end = answer_start
            while sent_start > 0 and paragraph[sent_start - 1] != ".":
                sent_start -= 1
            while sent_end < len(paragraph) and paragraph[sent_end] != ".":
                sent_end += 1
            example['predicted_sentence'] = paragraph[sent_start:sent_end]
            jsonArray.append(example)
  
    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        for elem in jsonArray:
            jsonString = json.dumps(elem, indent=4)
            jsonf.write(jsonString + ",\n")

def csv_to_csv(csvFilePath, output):
    jsonArray = []
    df = pd.read_csv(csvFilePath)
    df['predicted_sentence'] = df.apply(get_ans_sentence, axis=1)
    df['is_impossible'] = df.apply(get_impossiblity, axis=1)
    df.to_csv(output, index=False)
    return

    jsonArray = []
      
    #read csv file
    with open(csvFilePath, encoding='utf-8-sig') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for example in csvReader: 
            #add this python dict to json array
            paragraph = example['context']
            answer_start = paragraph.index(example['predicted_answer'])
            sent_start = answer_start
            sent_end = answer_start
            while sent_start > 0 and paragraph[sent_start - 1] != ".":
                sent_start -= 1
            while sent_end < len(paragraph) and paragraph[sent_end] != ".":
                sent_end += 1
            example['predicted_sentence'] = paragraph[sent_start:sent_end]
            jsonArray.append(example)
  
    #convert python jsonArray to JSON String and write to file
    with open(output, "w", newline="") as f:
        print(jsonArray)
        writer = csv.writer(f)
        writer.writerows(jsonArray)

            

json_file = "new_data/updated_trimmed_data.json"
csv_file = "new_data/TRIMMED_DATA.csv"
output_csv = "new_data/updated_trimmed_data.csv"
csv_to_csv(csv_file, output_csv)
csv_to_json(csv_file, json_file)
