# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:01:09 2024

@author: gnkhata
"""
import pandas as pd
#import os
#import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import deque
from statistics import mean
import torch

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_class_weights(labels, num_classes):
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    class_weights = [total_count / label_counts[i] if i in label_counts else 0 for i in range(num_classes)]
    return torch.tensor(class_weights, dtype=torch.float).to(device)

def find_levels(data, level=0, result=None):
    if result is None:
        result = list()
    if isinstance(data, dict):
        for key, value in data.items():
            result.append((key, level))
            find_levels(value, level + 1, result)
    elif isinstance(data, list):
        for item in data:
            find_levels(item, level, result)
    else:
        print(f"Level {level}: {type(data).__name__}")
    return result

def add_spec_tokens(src_post, replies):
    src_post = "[CLS] "+src_post+ " [SEP] "
    replies = [" [CLS] " + tokens + " [SEP] " for tokens in replies]
    return src_post, replies
# Example usage:
def merge_thread1(src_post, replies, reply_ids, struct_data, d_type):
    levels = find_levels(struct_data)
    sorted_levels = sorted(levels, key=lambda x: x[1])
    bfs_fuse = src_post
    for key, level in sorted_levels:
        if d_type == "twitter":
            try:# Handle the non-integer case, or skip processing this entry
                key = int(key)
            except ValueError:
                print(f"Non-integer key encountered: {key}")
                continue
            #key = int(key)
        if key in reply_ids:
            #bfs_fuse += " [SEP] " + replies[reply_ids.index(key)]
            bfs_fuse +=  replies[reply_ids.index(key)]
            
    return bfs_fuse

# Example usage:
def merge_thread(src_post, replies, reply_ids, struct_data, d_type):
    levels = find_levels(struct_data)
    sorted_levels = sorted(levels, key=lambda x: x[1])
    bfs_thread = [src_post]
    #bfs_thread.append(src_post)
    #bfs_thread = src_post#use this if you wanna return a thread as one sentence
    for key, level in sorted_levels:
        if d_type == "twitter":
            try:# Handle the non-integer case, or skip processing this entry
                key = int(key)
            except ValueError:
                print(f"Non-integer key encountered: {key}")
                continue
        if key in reply_ids:
            #bfs_fuse += " [SEP] " + replies[reply_ids.index(key)]
            bfs_thread.append(replies[reply_ids.index(key)])
            #bfs_thread += replies[reply_ids.index(key)] #use this if you wanna return a thread as one sentence
            
    return bfs_thread

def count_replies_lengths(threads):
    # Calculate lengths of each inner list
    lengths = [len(thread) for thread in threads]
    
    # Calculate mean, max, and min lengths
    if lengths:
        mean_length = mean(lengths)
        max_length = max(lengths)
        min_length = min(lengths)
    else:
        mean_length = max_length = min_length = 0  # if no inner list has exactly three columns
    
    return mean_length, max_length, min_length

def gen_dfs_stances1(nested_dict, replies, data_id):
    # Convert the list of replies to a dictionary for quick lookup
    reply_lookup = {r[0]: r[2] for r in replies}
    sorted_reply_ids = []  # Store sorted reply IDs
    sorted_stances = []  # Store sorted stances only

    def dfs(node):
        """Recursive helper function to traverse the structure using DFS."""
        if isinstance(node, dict):  # If node is a dictionary
            for reply_id, children in node.items():
                # Only process if reply_id exists in the lookup table
                try:
                    if data_id == "twt":
                        if int(reply_id) in reply_lookup:
                            sorted_reply_ids.append(int(reply_id))  # Store ID
                            sorted_stances.append(reply_lookup[int(reply_id)])  # Store stance
                    elif data_id == "redt":
                        if reply_id in reply_lookup:
                            sorted_reply_ids.append(reply_id)  # Store ID
                            sorted_stances.append(reply_lookup[reply_id])  # Store stance
                except ValueError:
                    print(f"Non-integer key encountered: {reply_id}")
                    continue

                # Recursively explore children
                dfs(children)
        elif isinstance(node, list):  # If node is a list
            for item in node:
                dfs(item)  # Process each element in the list

    # Start DFS from the root node
    root_key = next(iter(nested_dict))  # Get the root key
    dfs(nested_dict[root_key])

    # Return the sorted stances
    return sorted_stances


# DFS function that performs all operations and returns three lists
def gen_dfs_stances(nested_dict, replies, data_id):
    # Convert the list of replies to a dictionary for quick lookup
    reply_lookup = {r[0]: r[2] for r in replies}
    sorted_reply_ids = []  # Store sorted reply IDs
    sorted_stances = []  # Store sorted stances only

    def dfs(node):
        """Recursive helper function to traverse the dictionary using DFS."""
        for reply_id, children in node.items():
            # Only process if reply_id exists in the lookup table
            
            if data_id == "twt":
                if int(reply_id) in reply_lookup:
                    sorted_reply_ids.append(int(reply_id))  # Store ID
                    sorted_stances.append(reply_lookup[int(reply_id)])  # Store stance
            
            elif data_id == "redt":
                if reply_id in reply_lookup:
                    sorted_reply_ids.append(reply_id)  # Store ID
                    sorted_stances.append(reply_lookup[reply_id])  # Store stance
            
            # Recursively explore children if they exist
            if isinstance(children, dict) and children:
                dfs(children)

    # Start DFS from the root node
    root_key = next(iter(nested_dict))  # Get the root key
    dfs(nested_dict[root_key])

    # Prepare the final lists
    #sorted_reply_and_stance = [(reply_id, reply_lookup[reply_id]) for reply_id in sorted_reply_ids]
    #sorted_stances_limited = sorted_stances[:20]  # Limit to 20 entries

    #return sorted_reply_and_stance, sorted_stances, sorted_stances_limited
    return sorted_stances
    

# Recursive BFS function to traverse the entire nested dictionary
def gen_bfs_stances(nested_dict, replies, data_id):
    # Convert the list of replies to a dictionary for quick lookup
    reply_lookup = {r[0]: r[2] for r in replies}
    queue = deque([nested_dict])  # Initialize the queue with the entire dictionary
    sorted_ids = []  # List to store the sorted reply IDs

    while queue:
        current_dict = queue.popleft()  # Dequeue the current dictionary

        for reply_id, children in current_dict.items():
            try:
                if data_id == 'twt':
                    if int(reply_id) in reply_lookup:
                        sorted_ids.append(int(reply_id))
                elif data_id == "redt":
                    if reply_id in reply_lookup:
                        sorted_ids.append(reply_id)
            except ValueError:
                print(f"Invalid reply_id encountered: {reply_id}")
                continue
        
            if isinstance(children, dict) and children:
                queue.append(children)
    # Generate the three required lists
    #sorted_reply_and_stance = [(reply_id, reply_lookup[reply_id]) for reply_id in sorted_ids]
    sorted_stances = [reply_lookup[reply_id] for reply_id in sorted_ids]
    #sorted_stances_limited = sorted_stances[:20]  # Limit to 20 entries

    #return sorted_reply_and_stance, sorted_stances, sorted_stances_limited
    return sorted_stances

def gen_sortedStances_on_TimePosted(replies, dates):
    # Create a dictionary to map reply_id to date_created for quick lookup
    date_dict = {reply_id: date for reply_id, date in dates}

    # Sort the first list based on the date from the second list
    sorted_replies = sorted(replies, key=lambda x: date_dict.get(x[0]))

    # Generate the required lists
    #sorted_reply_ids_and_stances = [(reply[0], reply[2]) for reply in sorted_replies]
    sorted_stances_only = [reply[2] for reply in sorted_replies]
    #sorted_stances_limited = sorted_stances_only[:20]
    #return sorted_reply_ids_and_stances, sorted_stances_only, sorted_stances_limited
    return sorted_stances_only
def sort_posts_by_created_redt(post_time_data):
    # Sort the data based on the creation id (second element in the inner list)
    sorted_data = sorted(post_time_data, key=lambda x: x[1])
    
    # Extract the sorted post ids and the sorted original list
    sorted_ids = [x[0] for x in sorted_data]
    sorted_original = sorted_data
    return sorted_ids, sorted_original

def sort_posts_by_time_created_twt(posts):
    # Sort posts based on the parsed date (column 2 in each sublist)
    sorted_posts = sorted(posts, key=lambda x: datetime.strptime(x[1], '%a %b %d %H:%M:%S +0000 %Y'))

    # Extract post ids from the sorted list
    sorted_post_ids = [post[0] for post in sorted_posts]
    
    return sorted_post_ids, sorted_posts

def remove_key_from_dicts(dict_list, key_to_remove):
    for dictionary in dict_list:
        if key_to_remove in dictionary:
            del dictionary[key_to_remove]

def count_structure_lengths(struct_lens):
    # Compute the length of each sentence

    # Calculate min, max, and average lengths
    min_length = min(struct_lens)
    max_length = max(struct_lens)
    average_length = sum(struct_lens) / len(struct_lens)

    print(f"Minimum thread length: {min_length}")
    print(f"Maximum thread length: {max_length}")
    print(f"Average thread length: {average_length:.2f}")

    # Plot the distribution of sentence lengths
    plt.figure(figsize=(10, 6))
    plt.hist(struct_lens, bins=range(min_length, max_length + 2), edgecolor='black')
    plt.title("Thread Length Distribution")
    plt.xlabel("Thread Length (Number of Words)")
    plt.ylabel("Frequency")
    plt.xticks(range(min_length, max_length + 1))
    plt.grid(axis='y', linestyle='--')
    plt.show()

def count_text_lengths(txt):
    # Compute the length of each sentence
    sentence_lengths = [len(sentence.split()) for sentence in txt]

    # Calculate min, max, and average lengths
    min_length = min(sentence_lengths)
    max_length = max(sentence_lengths)
    average_length = sum(sentence_lengths) / len(sentence_lengths)

    print(f"Minimum sentence length: {min_length}")
    print(f"Maximum sentence length: {max_length}")
    print(f"Average sentence length: {average_length:.2f}")

    # Plot the distribution of sentence lengths
    plt.figure(figsize=(10, 6))
    plt.hist(sentence_lengths, bins=range(min_length, max_length + 2), edgecolor='black')
    plt.title("Sentence Length Distribution")
    plt.xlabel("Sentence Length (Number of Words)")
    plt.ylabel("Frequency")
    plt.xticks(range(min_length, max_length + 1))
    plt.grid(axis='y', linestyle='--')
    plt.show()

def concatenate_samples(twt_bfs_threads_train, redt_bfs_threads_train, twt_bfs_threads_val, redt_bfs_threads_val):
    train_data  = pd.concat([twt_bfs_threads_train, redt_bfs_threads_train], ignore_index=True)
    val_data    = pd.concat([twt_bfs_threads_val, redt_bfs_threads_val], ignore_index=True)
    train_data  = train_data.dropna()#drop null entries
    train_data  = train_data.sample(frac = 1)#shuffle tuples
    val_data  = val_data.dropna()#drop null entries
    val_data  = val_data.sample(frac = 1)#shuffle tuples
    train_txt, train_label, train_stanc_distr, train_struct, train_stances, train_posts, train_posts_stances, train_all_stances  = train_data["Text"], train_data["Veracity_int"], train_data["Stanc_Distr"], train_data['Structure'], train_data["Rep_Stances"], train_data["Posts"], train_data["Posts_Stances"], train_data["All_Stances"]        
    val_txt, val_label, val_stanc_distr, val_struct, val_stances, val_posts, val_posts_stances, val_all_stances   = val_data["Text"], val_data["Veracity_int"], val_data["Stanc_Distr"], val_data['Structure'], val_data["Rep_Stances"], val_data["Posts"], val_data["Posts_Stances"], val_data["All_Stances"] 
    #val_data['Text_length'] = val_data['Text'].apply(lambda x: len(x.split()))
    
    return train_txt,  val_txt, train_label, val_label, train_stanc_distr, val_stanc_distr,train_struct, val_struct, train_stances, val_stances, train_posts, val_posts, train_posts_stances, val_posts_stances, train_all_stances, val_all_stances


def count_stance_freq(data):
    column_index = 2
    threads = data['Replies']
    flattened_threads = [inner for outer in threads for inner in outer]
    try:
        replies = [reply[column_index] for reply in flattened_threads]
    except IndexError:
        print(f"Error: Column index {column_index} is out of range for one or more rows.")
        replies = []
    
    # Count the frequency of each stance in the replies
    stanc_freq = Counter(replies)
    #print(stanc_freq)
    return stanc_freq 

def plot_data_distribution(true_stanc_freq, false_stanc_freq, unver_stanc_freq):
    # Extract the common keys (ensure all counters have the same set of keys)
    keys = sorted(true_stanc_freq.keys())  # Assuming all counters have the same set of keys
    
    # Prepare values for each counter, ensuring all have the same length as the keys
    values1 = [true_stanc_freq.get(key, 0) for key in keys]
    values2 = [false_stanc_freq.get(key, 0) for key in keys]
    values3 = [unver_stanc_freq.get(key, 0) for key in keys]
    
    # Dataset names for x-axis
    datasets = ['True', 'False', 'Unverified']
    
    # Set up the bar width and positions
    bar_width = 0.2
    x = np.arange(len(datasets))  # x positions for the datasets
    
    # Values grouped by dataset for bar positions
    values = [values1, values2, values3]
    
    # Plotting the bars for each key (category)
    plt.figure(figsize=(10, 6))
    for i, key in enumerate(keys):
        # Offset each category's bars by i * bar_width to group them by dataset
        plt.bar(x + i * bar_width - bar_width, [values[j][i] for j in range(len(datasets))], width=bar_width, label=key)
    
    # Adding labels and title
    plt.xlabel('Veracity')
    plt.ylabel('Stance Frequency')
    #Change title here based on what is being plot
    plt.title('Stance Distribution Across Veracity Labels with Query Stance')
    plt.xticks(x, datasets)  # Set x-ticks to the dataset names
    
    # Add a legend to show which category each bar color represents
    plt.legend(title='Stances')
    
    # Display the plot
    plt.tight_layout()
    plt.show() 

def stanc_distr_in_reply(replies):
    # Specified list of keys (from the third column)
    stance_labels = ['support', 'deny', 'comment', 'query']
    
    # Initialize dictionary with the specified keys and set initial frequency to 0
    stanc_distr = {key: 0 for key in stance_labels}
    
    for reply in replies:
    # Get the element in the third column
        key = reply[2]
        
        # Update the frequency in the dictionary
        if key in stanc_distr:
            stanc_distr[key] += 1

    return stanc_distr
    
