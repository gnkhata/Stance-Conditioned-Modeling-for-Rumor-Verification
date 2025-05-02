# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:06:48 2024

@author: gnkhata
"""

import torch
import os
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import csv
from collections import Counter
from ReadInputPheme import read_data
from Utilities import count_text_lengths
# Import the custom model and dataset
from Stanc_Train import RumorVerificationModel, RumorDataset, custom_collate_fn  # Adjust import as needed

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a function to load the trained model
def load_model(model_path, lang_model_name, num_classes):
    if lang_model_name == "bert":
        lang_model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = RumorVerificationModel(lang_model, num_classes)
    elif lang_model_name == "roberta":
        lang_model = RobertaModel.from_pretrained("roberta-base")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RumorVerificationModel(lang_model, num_classes)
    elif lang_model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        lang_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=num_classes)
        tokenizer.pad_token = tokenizer.eos_token
        lang_model.config.pad_token_id = lang_model.config.eos_token_id
        model = lang_model
    else:
        raise ValueError("Unsupported language model name.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

# Define a function to predict the stance of input posts
def predict_stance(model, tokenizer, input_posts, post_ids, max_length=128):
    dataset = RumorDataset(post_ids, input_posts, [0] * len(input_posts), tokenizer, max_length, None, "one", None)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    predictions = []
    pred_stance = list()
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()

            for post_id, predicted_label in zip(batch["id"], predicted_labels):
                predictions.append({"post_id": post_id, "predicted_stance": int(predicted_label)})
                pred_stance.append([int(post_id), int(predicted_label)])

    return predictions,pred_stance

# Main function
if __name__ == "__main__":
    # Configuration
    model_path = "./Best_Model/Stanc_Det/bert/1best_model.pth"  # Adjust path as needed
    #./Best_Model/Stanc_Det/bert/1best_model2.pth
    lang_model_name = "bert"  # "bert", "roberta", or "gpt2"
    num_classes = 4  # Change if different in your setup
    max_length = 128
    
    # Initialize an empty dictionary
    #data_dict = {}
    post_ids, post_texts = list(), list()
    main_dir = "./PHEME_veracity/all-rnr-annotated-threads/"
    dirs = [
        "charliehebdo-all-rnr-threads", "ebola-essien-all-rnr-threads", 
        "ferguson-all-rnr-threads", "germanwings-crash-all-rnr-threads", 
        "gurlitt-all-rnr-threads", "ottawashooting-all-rnr-threads",
        "prince-toronto-all-rnr-threads", "putinmissing-all-rnr-threads", 
        "sydneysiege-all-rnr-threads"
    ]
    for topic in dirs:
        dir_path = os.path.join(main_dir, topic)
        ids, posts = read_data(dir_path, model_type="stanc_det")
        post_ids.extend(ids)
        post_texts.extend(posts)
    '''
    count_text_lengths(post_texts)
    print(len(post_ids))
    print(len(post_texts))
    print(post_ids)
    '''
    # Load the model and tokenizer
    model, tokenizer = load_model(model_path, lang_model_name, num_classes)
    # Predict stances
    results, pred_stances = predict_stance(model, tokenizer, post_texts, post_ids, max_length)
    '''
    # Output predictions
    print("Predictions:")
    for result in results:
        print(result)
    print(pred_stances[-2])
    '''
    results_df = pd.DataFrame(pred_stances, columns=["ID", "Stance_No"])
    results_df["ID"] = results_df["ID"]
    save_path = "./PHEME_veracity/All_Posts_Predicted_Stance/all_posts_predicted_stances5.csv"
    #results_df.to_csv(save_path, index=False, quoting=pd.io.common.csv.QUOTE_NONE, escapechar='\\')
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")