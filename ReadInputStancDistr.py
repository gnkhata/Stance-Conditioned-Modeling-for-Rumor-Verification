# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:21:37 2024

@author: gnkhata
"""
import pandas as pd
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
from Utilities import sort_posts_by_created_redt, sort_posts_by_time_created_twt, remove_key_from_dicts, count_structure_lengths, count_text_lengths, concatenate_samples 
from Utilities import find_levels, add_spec_tokens, merge_thread, count_stance_freq, plot_data_distribution, stanc_distr_in_reply 
from Utilities import gen_sortedStances_on_TimePosted, gen_bfs_stances, gen_dfs_stances
from Structure_selector import get_top_n_levels, generate_dfs_levels, get_levels_and_first_n_bfs, get_top_n_levels_without_comment_stance, get_levels_and_first_n_bfs_without_comment_stance, generate_dfs_levels_without_comment_stance
from Text_preprocessing import process_texts2
def read_twitr_data(directory, labels, data_mode):
    if data_mode == "src_twt_only":
        df = pd.DataFrame(columns=['Text', 'Veracity'])
        for dirpath, _, filenames in os.walk(directory):
            if os.path.basename(dirpath) == 'source-tweet':
                for file in filenames:
                    if file.endswith(".json") and file not in {"structure.json", "dev-key.json", "train-key.json"}:
                        file_path = os.path.join(dirpath, file)
                        with open(file_path, 'r') as json_file:
                            try:
                                data = json.load(json_file)
                                if str(data['id']) in labels['subtaskbenglish']:
                                    df.loc[len(df.index)] = [data['text'], labels['subtaskbenglish'][str(data['id'])]]
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON in file {file_path}: {e}")
        df['Veracity_int'] = df['Veracity'].map({'true': 0, 'false': 1, 'unverified': 2})
        return df

    elif data_mode == "src_twt_all_replies":
        df = pd.DataFrame(columns=['Src_post_Id', 'Text', 'Veracity', 'Stance', 'Replies', 'Stanc_Distr', 'Structure', 'Thread_len', 
                                   'Time_Created', 'Replies_Time_Created', 'NoCommentReply_Len', 'Rep_Stances', 'Posts', 'Posts_Stances', 'All_Stances'])
        
        for root_dirpath, dirnames, _ in os.walk(directory):
            for subdir in dirnames:
                path = os.path.join(root_dirpath, subdir)
                src_post, src_post_id, src_post_time_created = None, None, None
                r_label, src_post_s_label = None, None
                replies, reply_ids, repl_with_stance = [], [], []
                comment_reply_ids, posts, posts_stances = [], [], []
                post_time_created, comments, all_stances = [], [], []
                
                for dirpath, _, files in os.walk(path):
                    if "structure.json" in files:
                        with open(os.path.join(dirpath, "structure.json"), 'r') as json_file:
                            struct_data = json.load(json_file)
                    
                    if os.path.basename(dirpath) == 'source-tweet':
                        for file in files:
                            if file.endswith(".json") and file not in {"dev-key.json", "train-key.json"}:
                                file_path = os.path.join(dirpath, file)
                                with open(file_path, 'r') as json_file:
                                    try:
                                        data = json.load(json_file)
                                        src_post, src_post_id = data['text'], str(data['id'])
                                        src_post_time_created = data['created_at']
                                        post_time_created.append([src_post_id, src_post_time_created])
                                        src_post_s_label = labels['subtaskaenglish'].get(src_post_id)
                                        r_label = labels['subtaskbenglish'].get(src_post_id)
                                        all_stances.append([src_post_id, src_post, src_post_s_label])
                                        if r_label is not None:
                                            posts.append(src_post)
                                            posts_stances.append(src_post_s_label)
                                    except json.JSONDecodeError as e:
                                        print(f"Error decoding JSON in file {file_path}: {e}")
                    
                    elif os.path.basename(dirpath) == 'replies':
                        for file in files:
                            if file.endswith(".json"):
                                file_path = os.path.join(dirpath, file)
                                with open(file_path, 'r') as json_file:
                                    try:
                                        data = json.load(json_file)
                                        rep_id = data['id']
                                        rep_post = data['text']
                                        rep_s_label = labels['subtaskaenglish'].get(str(rep_id))
                                        if rep_s_label == 'comment':
                                            comment_reply_ids.append(rep_id)
                                            comments.append(rep_s_label)
                                        replies.append(rep_post)
                                        reply_ids.append(rep_id)
                                        repl_with_stance.append([rep_id, rep_post, rep_s_label])
                                        post_time_created.append([rep_id, data['created_at']])
                                        posts.append(rep_post)
                                        posts_stances.append(rep_s_label)
                                        all_stances.append([rep_id, rep_post, rep_s_label])
                                    except json.JSONDecodeError as e:
                                        print(f"Error decoding JSON in file {file_path}: {e}")
                
                if r_label is not None:
                    stanc_distr_in_replies = stanc_distr_in_reply(repl_with_stance)
                    #print(struct_data)
                    #print()
                    structure = find_levels(struct_data)
                    reply_ids_noCommentStance = list(set(reply_ids) - set(comment_reply_ids))
                    sorted_post_ids, sorted_posts = sort_posts_by_time_created_twt(post_time_created)
                    
                    """ Generating BiLSTM features"""
                    #sorted_stances_timePosted = gen_sortedStances_on_TimePosted(repl_with_stance, post_time_created)
                    sorted_stances_bfs = gen_bfs_stances(struct_data, repl_with_stance, data_id="twt")
                    #sorted_stances_dfs = gen_dfs_stances(struct_data, repl_with_stance, data_id="twt")
                    stances = sorted_stances_bfs
                    
                    #top_20_levels_time_posted = get_top_n_levels(structure, sorted_post_ids, data_type="reval")#based on post creation time
                    #top_20_levels_time_posted_noCommentStance = get_top_n_levels_without_comment_stance(structure, sorted_post_ids, comment_reply_ids)
                    #first_20_levels_bfs = get_levels_and_first_n_bfs(struct_data, data_type="reval")
                    #first_20_levels_bfs_noCommentStance = get_levels_and_first_n_bfs_without_comment_stance(struct_data, comment_reply_ids)
                    first_20_levels_dfs = generate_dfs_levels(struct_data, data_type="reval")
                    #first_20_levels_dfs_noCommentStance = generate_dfs_levels_without_comment_stance(struct_data, comment_reply_ids)
                    #structure = top_20_levels_time_posted
                    #structure = first_20_levels_bfs
                    structure = first_20_levels_dfs
                    #structure = first_20_levels_dfs_noCommentStance
                    
                    df.loc[len(df.index)] = [
                        src_post_id, src_post, r_label, src_post_s_label, repl_with_stance, stanc_distr_in_replies, structure, 
                        len(structure), src_post_time_created, post_time_created, len(reply_ids_noCommentStance), 
                        stances, posts, posts_stances, all_stances
                    ]                    
        df['Veracity_int'] = df['Veracity'].map({'true': 0, 'false': 1, 'unverified': 2})
        return df
    
def read_reddit_data(directory, labels, data_mode):
    if data_mode == "src_twt_only":
        df = pd.DataFrame(columns=['Text', 'Veracity'])
        for dirpath, _, filenames in os.walk(directory):
            if os.path.basename(dirpath) == 'source-tweet':
                for file in filenames:
                    if file.endswith(".json") and file not in {"structure.json", "dev-key.json", "train-key.json"}:
                        file_path = os.path.join(dirpath, file)
                        with open(file_path, 'r') as json_file:
                            try:
                                data = json.load(json_file)
                                if str(data['data']['children'][0]['data']['id']) in labels['subtaskaenglish'].keys():
                                    df.loc[len(df.index)] = [data['data']['children'][0]['data']['title'], labels['subtaskbenglish'][str(data['data']['children'][0]['data']['id'])]]
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON in file {file_path}: {e}")
        df['Veracity_int'] = df['Veracity'].map({'true': 0, 'false': 1, 'unverified': 2})
        return df

    elif data_mode == "src_twt_all_replies":
        df = pd.DataFrame(columns=['Src_post_Id', 'Text', 'Veracity', 'Stance', 'Replies', 'Stanc_Distr', 'Structure', 'Thread_len', 
                                   'Time_Created', 'Replies_Time_Created', 'NoCommentReply_Len', 'Rep_Stances', 'Posts', 'Posts_Stances', 'All_Stances'])
        
        for root_dirpath, dirnames, _ in os.walk(directory):
            for subdir in dirnames:
                path = os.path.join(root_dirpath, subdir)
                src_post, src_post_id, src_post_time_created = None, None, None
                r_label, src_post_s_label = None, None
                replies, reply_ids, repl_with_stance = [], [], []
                comment_reply_ids, posts, posts_stances = [], [], []
                post_time_created, comments, all_stances = [], [], []
                for dirpath, _, files in os.walk(path):
                    if "structure.json" in files:
                        with open(os.path.join(dirpath, "structure.json"), 'r') as json_file:
                            struct_data = json.load(json_file)
                    
                    if os.path.basename(dirpath) == 'source-tweet':
                        for file in files:
                            if file.endswith(".json") and file not in {"dev-key.json", "train-key.json"}:
                                file_path = os.path.join(dirpath, file)
                                with open(file_path, 'r') as json_file:
                                    try:
                                        data = json.load(json_file)
                                        src_post = data['data']['children'][0]['data']['title']
                                        src_post_id = str(data['data']['children'][0]['data']['id'])
                                        src_post_time_created = data['data']['children'][0]['data']['created']
                                        #post_time_created.append(src_post_time_created)
                                        post_time_created.append([src_post_id, src_post_time_created])
                                        if str(data['data']['children'][0]['data']['id']) in labels['subtaskaenglish'].keys():
                                            src_post_s_label = labels['subtaskaenglish'][str(data['data']['children'][0]['data']['id'])]
                                        if str(data['data']['children'][0]['data']['id']) in labels['subtaskbenglish'].keys():
                                            r_label = labels['subtaskbenglish'][str(data['data']['children'][0]['data']['id'])]
                                        else:
                                            continue
                                        posts.append(src_post)
                                        posts_stances.append(src_post_s_label)
                                        all_stances.append([src_post_id, src_post, src_post_s_label])
                                    except json.JSONDecodeError as e:
                                        print(f"Error decoding JSON in file {file_path}: {e}")
                                        
                    elif os.path.basename(dirpath) == 'replies':
                        for file in files:
                            if file.endswith(".json"):
                                file_path = os.path.join(dirpath, file)
                                with open(file_path, 'r') as json_file:
                                    try:
                                        data = json.load(json_file)
                                        rep_s_label = None
                                        try:
                                            rep_id = data['data']['id']
                                            rep_post = data['data']['body']
                                            if str(rep_id) in labels['subtaskaenglish'].keys():
                                                rep_s_label = labels['subtaskaenglish'][str(rep_id)]
                                            if rep_s_label == 'comment':
                                                comment_reply_ids.append(rep_id)
                                                comments.append(rep_s_label)
                                            replies.append(rep_post)
                                            reply_ids.append(rep_id)
                                            repl_with_stance.append([rep_id, rep_post, rep_s_label])
                                            post_time_created.append([rep_id, data['data']['created']])
                                            posts.append(rep_post)
                                            posts_stances.append(rep_s_label)
                                            all_stances.append([rep_id, rep_post, rep_s_label])
                                        except KeyError:
                                            continue
                                    except json.JSONDecodeError as e:
                                        print(f"Error decoding JSON in file {file_path}: {e}")
                
                if r_label is not None:
                    stanc_distr_in_replies = stanc_distr_in_reply(repl_with_stance)
                    structure = find_levels(struct_data)
                    reply_ids_noCommentStance = list(set(reply_ids) - set(comment_reply_ids))
                    sorted_post_ids, sorted_posts = sort_posts_by_created_redt(post_time_created)
                    
                    #sorted_stances_timePosted = gen_sortedStances_on_TimePosted(repl_with_stance, post_time_created)
                    sorted_stances_bfs = gen_bfs_stances(struct_data, repl_with_stance, data_id='redt')
                    #sorted_stances_dfs = gen_dfs_stances(struct_data, repl_with_stance, data_id="redt")
                    stances = sorted_stances_bfs
                    #top_20_levels_time_posted = get_top_n_levels(structure, sorted_post_ids, data_type="reval")
                    #top_20_levels_time_posted_noCommentStance = get_top_n_levels_without_comment_stance(structure, sorted_post_ids, comment_reply_ids)
                    #first_20_levels_bfs = get_levels_and_first_n_bfs(struct_data, data_type="reval")
                    #first_20_levels_bfs_noCommentStance = get_levels_and_first_n_bfs_without_comment_stance(struct_data, comment_reply_ids)
                    first_20_levels_dfs = generate_dfs_levels(struct_data, data_type="reval")
                    #first_20_levels_dfs_noCommentStance = generate_dfs_levels_without_comment_stance(struct_data, comment_reply_ids)
                    #structure = top_20_levels_time_posted
                    structure = first_20_levels_dfs
                    
                    #structure = first_20_levels_dfs_noCommentStance
                    
                    df.loc[len(df.index)] = [
                        src_post_id, src_post, r_label, src_post_s_label, repl_with_stance, stanc_distr_in_replies, structure, 
                        len(structure), src_post_time_created, post_time_created, len(reply_ids_noCommentStance), 
                        stances, posts, posts_stances, all_stances
                    ]
                    
        df['Veracity_int'] = df['Veracity'].map({'true': 0, 'false': 1, 'unverified': 2})
        return df


# Read tweet train data
def read_data(flag, data_mode, model):
    if flag == "train":
        if data_mode == "src_twt_only":
            
            #Read tweet train data
            directory_path = './Rumoureval2019/rumoureval-2019-training-data/twitter-english'
            label_path = './Rumoureval2019/rumoureval-2019-training-data/train-key.json'
            with open(label_path, 'r') as file:
                labels = json.load(file)
            twt_data_train = read_twitr_data(directory_path, labels, data_mode)
            
            # Read reddit train data
            directory_path = './Rumoureval2019/rumoureval-2019-training-data/reddit-training-data'
            redt_data_train = read_reddit_data(directory_path, labels, data_mode)
            
            # Read reddit dev data
            directory_path = './Rumoureval2019/rumoureval-2019-training-data/reddit-dev-data'
            label_path = './Rumoureval2019/rumoureval-2019-training-data/dev-key.json'
            with open(label_path, 'r') as file:
                labels = json.load(file)
            redt_data_dev = read_reddit_data(directory_path, labels, data_mode)
            # Read tweet val data
            directory_path = './Rumoureval2019/rumoureval-2019-training-data/twitter-english'
            twt_data_dev = read_twitr_data(directory_path, labels, data_mode)
            
            train_txt,  val_txt, train_label, val_label = concatenate_samples(twt_data_train, redt_data_train, 
                                                                              twt_data_dev, redt_data_dev)
            return train_txt,  val_txt, train_label, val_label
            
        elif data_mode == "src_twt_all_replies":
            twt_bfs_threads_train = pd.DataFrame()
            twt_bfs_threads_val = pd.DataFrame()
            twt_topics = ['charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash', 'illary', 'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege']
            
            label_path = './Rumoureval2019/rumoureval-2019-training-data/train-key.json'
            with open(label_path, 'r') as file:
                labels_train = json.load(file)
                
            label_path = './Rumoureval2019/rumoureval-2019-training-data/dev-key.json'
            with open(label_path, 'r') as file:
                labels_val = json.load(file)
            
            #Read Twitter train data
            for topic in twt_topics:
                directory_path = './Rumoureval2019/rumoureval-2019-training-data/twitter-english/'+topic
                topic_threads = read_twitr_data(directory_path, labels_train, data_mode)
                #result_df = pd.concat([result_df, df], ignore_index=True)
                twt_bfs_threads_train = pd.concat([twt_bfs_threads_train, topic_threads], ignore_index=True) 
                #print("Topic: ", topic)
                #break        
            
            #Read Twitter validation
            for topic in twt_topics:
                directory_path = './Rumoureval2019/rumoureval-2019-training-data/twitter-english/'+topic
                topic_threads = read_twitr_data(directory_path, labels_val, data_mode)
                #result_df = pd.concat([result_df, df], ignore_index=True)
                twt_bfs_threads_val = pd.concat([twt_bfs_threads_val, topic_threads], ignore_index=True)
            
            #Read Reddit train data
            directory_path = './Rumoureval2019/rumoureval-2019-training-data/reddit-training-data'
            redt_bfs_threads_train = read_reddit_data(directory_path, labels_train, data_mode)
            
            #Reddit val data
            directory_path = './Rumoureval2019/rumoureval-2019-training-data/reddit-dev-data'
            redt_bfs_threads_val  = read_reddit_data(directory_path, labels_val, data_mode)
            
            train_txt,  val_txt, train_labels, val_labels, train_stanc_distr, val_stanc_distr, train_struct, val_struct, train_stances, val_stances, train_posts, val_posts,  train_posts_stances, val_posts_stances, train_all_stances, val_all_stances = concatenate_samples(twt_bfs_threads_train, redt_bfs_threads_train, 
                                                                              twt_bfs_threads_val, redt_bfs_threads_val)
            #return train_txt,  val_txt, train_label, val_label, twt_bfs_threads_train
            #return train_txt,  val_txt, train_label, val_label  
            #return train_txt,  val_txt, train_label, val_label, train_stanc_distr, val_stanc_distr

            #return twt_bfs_threads_train, redt_bfs_threads_train, twt_bfs_threads_val, redt_bfs_threads_val 
            #return train_txt,  val_txt, train_label, val_label, train_stanc_distr, val_stanc_distr
            if model == "BiLSTM":
                return train_stances, val_stances, train_labels, val_labels
            elif model == "combined":
                #return train_txt,  val_txt, train_label, val_label, train_stanc_distr, val_stanc_distr, train_struct, val_struct, train_stances, val_stances 
                return train_txt, val_txt, train_stances, val_stances, train_labels, val_labels
            elif model == "source_agg_replies":
                return train_posts, val_posts, train_posts_stances, val_posts_stances, train_labels, val_labels
            elif model == "stanc_det":
                return train_all_stances, val_all_stances 
            elif model == "agg_stanc_struct":
                return train_posts, val_posts, train_posts_stances, val_posts_stances, train_stanc_distr, val_stanc_distr, train_struct, val_struct, train_labels, val_labels
            
    else:
        #Read twitter test data
        twt_dir_path = './Rumoureval2019/rumoureval-2019-test-data/twitter-en-test-data'
        redt_dir_path = './Rumoureval2019/rumoureval-2019-test-data/reddit-test-data'
        label_path = './Rumoureval2019/final-eval-key.json'
        with open(label_path, 'r') as file:
            labels_test = json.load(file)
        if data_mode == "src_twt_only":
            twt_data_test = read_twitr_data(twt_dir_path, labels_test, data_mode)
            # Read reddit dev data
            redt_data_test = read_reddit_data(redt_dir_path, labels_test, data_mode)
            
        elif data_mode == "src_twt_all_replies":
            #read twitter test data
            for dirpath, dirnames, filenames in os.walk(twt_dir_path):
                #print("Dir names: \n", dirnames)
                break
            twt_data_test = pd.DataFrame()
            count = 0
            for topic in dirnames:
                count += 1
                directory_path = './Rumoureval2019/rumoureval-2019-test-data/twitter-en-test-data/'+topic
                topic_threads = read_twitr_data(directory_path, labels_test, data_mode)
                #result_df = pd.concat([result_df, df], ignore_index=True)
                twt_data_test = pd.concat([twt_data_test, topic_threads], ignore_index=True)
            
            #read reddit test data
            redt_data_test  = read_reddit_data(redt_dir_path, labels_test, data_mode)
                        
        test_data    = pd.concat([redt_data_test, twt_data_test], ignore_index=True)
        test_data  = test_data.dropna()#drop null entries
        #test_data['Text_length'] = test_data['Text'].apply(lambda x: len(x.split()))
        
        test_data  = test_data.sample(frac = 1)#shuffle tuples
        #test_txt, test_label, test_stanc_distr   = test_data["Text"], test_data["Veracity_int"], test_data["Stanc_Distr"]
        test_txt, test_label, test_stanc_distr, test_struct, test_stances, test_posts, test_posts_stances, test_all_stances = test_data["Text"], test_data["Veracity_int"], test_data["Stanc_Distr"], test_data['Structure'], test_data["Rep_Stances"], test_data["Posts"], test_data["Posts_Stances"], test_data["All_Stances"]
        #test_txt   = test_data["Text"]
        #test_label = test_data["Veracity_int"]
        #return test_data, test_txt, test_label
        #return test_txt, test_label
        #return twt_data_test, redt_data_test
        #print(len(test_txt))
        #print(len(test_label))
        #print(len(test_stanc_distr))
        #   print(len(test_struct))
        #return test_txt, test_label, test_stanc_distr
        if model == "BiLSTM":
            return test_stances, test_label
        elif model == "combined":
            #return test_txt, test_label, test_stanc_distr, test_struct, test_stances
            return test_txt, test_label, test_stances
        elif model == "source_agg_replies":
            return test_posts, test_posts_stances, test_label
        elif model == "stanc_det":
            return test_all_stances
        elif model == "agg_stanc_struct":
            return  test_posts, test_posts_stances, test_stanc_distr, test_struct, test_label
def read_stanc_data():
    train_all_stances, val_all_stances = read_data(flag="train", data_mode="src_twt_all_replies", model="stanc_det")
    test_all_stances = read_data(flag="test", data_mode="src_twt_all_replies", model="stanc_det")
    data_threads    = pd.concat([train_all_stances, test_all_stances, test_all_stances], ignore_index=True)
    mapping = {
    'deny': 0,
    'query': 1,
    'comment': 2,
    'support': 3
    }
    
    data_threads = list(data_threads)
    data = []
    for thread in data_threads:
        data.extend(thread)
    
    
    data = list(map(list, set(map(tuple, data))))
    
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    # Extract the last column
    
    for row in train_data:
        row[-1] = mapping[row[-1]]
    for row in val_data:
        row[-1] = mapping[row[-1]]
    train_ids, train_posts, train_labels = zip(*train_data)
    val_ids, val_posts, val_labels = zip(*val_data)
    train_posts, val_posts = process_texts2(train_posts), process_texts2(val_posts)
    '''
    print("Train text lengths: ")
    count_text_lengths(train_posts)
    print("Val text lengths")
    count_text_lengths(val_posts)
    '''
    return train_ids, train_posts, train_labels, val_ids, val_posts, val_labels 
    
"""       
if __name__ == "__main__":
    #read_stanc_data()
    read_data(flag="train", data_mode="src_twt_all_replies", model="source_agg_replies")
    #train_posts, val_posts, train_posts_stances, val_posts_stances, train_labels, val_labels = read_data(flag="train", data_mode="src_twt_all_replies", model="source_agg_replies")
    #print(len(val_posts))
    #read_data(flag="test", data_mode="src_twt_all_replies")
    '''
    twt_threads_train, redt_threads_train, twt_threads_val, redt_threads_val = read_data(flag="train", data_mode="src_twt_all_replies")
    data = pd.concat([twt_threads_train,redt_threads_train, twt_threads_val,redt_threads_val], ignore_index=True)
    twt_data = pd.concat([twt_threads_train, twt_threads_val], ignore_index=True)
    redt_data = pd.concat([redt_threads_train, redt_threads_val], ignore_index=True)
    print('Twt data thread lengths: \n')
    count_structure_lengths(twt_data['NoCommentReply_Len'])
    print('\nRedt data thread lengths: \n')
    count_structure_lengths(redt_data['NoCommentReply_Len'])
    print('\nAll data thread lengths: \n')
    count_structure_lengths(data['NoCommentReply_Len'])
    #print(data.iloc[0]['Replies'][0][2])
    '''
    #train_txt,  val_txt, train_label, val_label, train_stanc_distr, val_stanc_distr = read_data(flag="train", data_mode="src_twt_all_replies")
    #test_txt,   test_label, test_stanc_distr = read_data(flag="test", data_mode="src_twt_all_replies")
    '''
    train_txt,  val_txt, train_label, val_label, train_stanc_distr, val_stanc_distr, train_struct, val_struct = read_data(flag="train", data_mode="src_twt_all_replies")
    print(len(train_struct))
    print()
    print(len(val_struct))
    '''
    
    '''
    print(len(test_txt))
    print()
    print(len(test_label))
    print()
    print(len(test_stanc_distr))
    print()
    '''
    '''
    # Sample list of sentences
    #print(train_txt)
    count_text_lengths(val_txt)
    print(train_stanc_distr)
    # Example usage
    key_to_remove = 'comment'
    remove_key_from_dicts(train_stanc_distr, key_to_remove)
    remove_key_from_dicts(val_stanc_distr, key_to_remove)
    print(train_stanc_distr)
    print()
    print(val_stanc_distr)
    '''
    '''
    # Sample list of sentences
    #print(train_txt)
    count_text_lengths(test_txt)
    print(test_stanc_distr)
    print()
    # Example usage
    key_to_remove = 'comment'
    remove_key_from_dicts(test_stanc_distr, key_to_remove)
    print()
    print(test_stanc_distr)
    '''
 """   