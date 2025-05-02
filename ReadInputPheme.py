# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:56:11 2024

@author: gnkhata
"""
import pandas as pd
import os
import json
import pandas as pd
from PHEME_veracity.convert_veracity_annotations import convert_annotations
from Utilities import gen_sortedStances_on_TimePosted, gen_bfs_stances, gen_dfs_stances1
from Utilities import find_levels, add_spec_tokens, merge_thread1, merge_thread
from Utilities import sort_posts_by_time_created_twt, count_text_lengths, count_replies_lengths
from Utilities import stanc_distr_in_reply
from Structure_selector import get_top_n_levels, get_levels_and_first_n_bfs, generate_dfs_levels
from Text_preprocessing import process_texts2

stance_file_path = "./PHEME_veracity/All_Posts_Predicted_Stance/all_posts_predicted_stances5.csv"
stanc_df = pd.read_csv(stance_file_path)
#print(stanc_df.head())
pred_stances = {0:'deny', 1:'query', 2:'comment', 3:'support'}
stanc_df['Stance'] = stanc_df['Stance_No'].apply(lambda x: pred_stances.get(x, -1))
#print(stanc_df.head())
#print(len(stanc_df['Stance']))

ids_stancs = dict(zip(stanc_df.iloc[:, 0], stanc_df.iloc[:, 2]))

#print(ids_stancs)
#len(print(ids_stancs))

def read_pheme_data(main_dir, model_type):
    mid_sub_dirs = ["rumours", "non-rumours"]
    
    df = pd.DataFrame(columns=['Src_post_Id', 'Text', 'Veracity', 'Time_Created', 'Replies', 'Structure', 'Merged_Thread', 'Stanc_Distr', 'Rep_Ids', 'Rep_Stances', 'Posts', 'Posts_Stances'])
    #df = pd.DataFrame(columns=['Src_post_Id', 'Text', 'Veracity', 'Time_Created', 'Replies', 'Structure', 'Merged_Thread', 'Rep_Ids'])
    count = 0
    for mid_sub_dir in mid_sub_dirs:
        directory = os.path.join(main_dir, mid_sub_dir)
        '''
        if count == 500:
            break
        '''
        for dirpath, dirnames, filenames in os.walk(directory):
            '''
            if count == 500:
                break
            '''
            # Check if 'annotation.json' and 'structure.json' exist at this level
            if "annotation.json" in filenames:
                # Read the annotation to get the veracity label
                with open(os.path.join(dirpath, "annotation.json"), 'r', encoding='utf-8') as json_file:
                    label = json.load(json_file)
                    r_label = convert_annotations(label)
                
                # Initialize variables to store the tweet data and structure
                src_post, src_post_id, src_post_time_created, replies, structure = None, None, None, [], None
                
                # Read the structure if 'structure.json' is present
                if "structure.json" in filenames:
                    with open(os.path.join(dirpath, "structure.json"), 'r', encoding='utf-8') as json_file:
                        try:
                            struct_data = json.load(json_file)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in structure.json at {dirpath}: {e}")

                # Look for source tweet in 'source-tweets' directory
                source_tweet_path = os.path.join(dirpath, 'source-tweets')
                if os.path.isdir(source_tweet_path):
                    for file in os.listdir(source_tweet_path):
                        if file.endswith(".json") and not file.startswith("._"):
                            file_path = os.path.join(source_tweet_path, file)
                            with open(file_path, 'r', encoding='utf-8') as json_file:
                                try:
                                    data = json.load(json_file)
                                    src_post_id = data.get('id')
                                    src_post = data.get('text')
                                    src_post_time_created = data.get('created_at')
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON in file {file_path}: {e}")

                # Look for replies in 'reactions' directory
                reactions_path = os.path.join(dirpath, 'reactions')
                if os.path.isdir(reactions_path):
                    for file in os.listdir(reactions_path):
                        if file.endswith(".json") and not file.startswith("._"):
                            file_path = os.path.join(reactions_path, file)
                            with open(file_path, 'r', encoding='utf-8') as json_file:
                                try:
                                    data = json.load(json_file)
                                    replies.append([data.get('id'), data.get('text'), data.get('created_at')])
                                    #repl_with_stance.append([data['id'], data['text'],rep_s_label])
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON in file {file_path}: {e}")
                '''
                print(struct_data)
                print()
                '''
                # Append data if source tweet exists
                if src_post_id is not None:
                    posts = [src_post]
                    posts_stances = [ids_stancs.get(int(src_post_id))]
                    count  += 1
                    post_time_created = [[src_post_id, src_post_time_created]]
                    replies_time_created= [[row[0], row[2]] for row in replies]
                    post_time_created.extend(replies_time_created)
                    
                    #post_time_created = [[row[0], row[2]] for row in replies]
                    struct_levels = find_levels(struct_data)
                    sorted_post_ids, sorted_posts = sort_posts_by_time_created_twt(post_time_created)
                    top_20_levels_time_posted = get_top_n_levels(struct_levels, sorted_post_ids, data_type="pheme")#based on post creation time
                    #first_20_levels_bfs = get_levels_and_first_n_bfs(struct_data, data_type="pheme")
                    #first_20_levels_dfs = generate_dfs_levels(struct_data, data_type="pheme")
                    structure = top_20_levels_time_posted
                    
                    reply_ids = [row[0] for row in replies]
                    reply_texts = [row[1] for row in replies]
                    posts.extend(reply_texts)
                    
                    repl_with_stance =  [[row[0], row[1], ids_stancs.get(int(row[0]))] for row in replies]
                    rep_stances = [ids_stancs.get(int(row[0])) for row in replies]
                    posts_stances.extend(rep_stances)
                    
                    stanc_distr_in_replies = stanc_distr_in_reply(repl_with_stance)
                    #id_stanc = [[row[0],  ids_stancs.get(row[0])] for row in replies]
                    
                    sorted_stances_timePosted = gen_sortedStances_on_TimePosted(repl_with_stance, post_time_created)
                    #sorted_stances_bfs = gen_bfs_stances(struct_data, repl_with_stance, data_id="twt")
                    #sorted_stances_dfs = gen_dfs_stances1(struct_data, repl_with_stance, data_id="twt")
                    
                    stances = sorted_stances_timePosted
                    
                    if model_type=="src_twt_all_replies":
                        src_post, reply_texts = add_spec_tokens(src_post, reply_texts)
                        merged_thread_bfs = merge_thread1(src_post, reply_texts, reply_ids, struct_data, d_type="twitter")
                    else:
                        merged_thread_bfs = merge_thread(src_post, reply_texts, reply_ids, struct_data, d_type="twitter")

                    #df.loc[len(df.index)] = [merged_thread_bfs, r_label]
                    
                    df.loc[len(df.index)] = [src_post_id, src_post, r_label, src_post_time_created, replies, structure, merged_thread_bfs, stanc_distr_in_replies, reply_ids, stances, posts, posts_stances]
                    #df.loc[len(df.index)] = [src_post_id, src_post, r_label, src_post_time_created, replies, structure, merged_thread_bfs, reply_ids]
                    #print()
    # Map veracity labels to integer values
    zero_numbering = {'true': 0, 'false': 1, 'unverified': 2}
    df['Veracity_int'] = df['Veracity'].apply(lambda x: zero_numbering.get(x, -1))
    
    return df
   

def read_data(main_dir, model_type):
    data = read_pheme_data(main_dir, model_type=model_type)
    data  = data.dropna()#drop null entries 
    data = data.sample(frac=1)
    src_post_ids, src_posts, structure, replies, labels, merged_threads, stanc_distrs, reply_ids = data['Src_post_Id'], data['Text'], data['Structure'], data['Replies'], data['Veracity_int'], data['Merged_Thread'], data['Stanc_Distr'], data['Rep_Ids']
    posts, posts_stances = data["Posts"], data["Posts_Stances"]
    rep_stances = data['Rep_Stances']
    #src_post_ids, src_posts, structure, replies, labels, merged_threads = data['Src_post_Id'], data['Text'], data['Structure'], data['Replies'], data['Veracity_int'], data['Merged_Thread']
    #print(stanc_distrs)
    #Replace @user menations with $mention$ and urls with $URL$ and normalize #hashtags 
    '''
    id_distr = zip(labels, rep_stances)
    for _, rep_stanc in id_distr:
        print(rep_stanc)
        print()
    '''
    #return data
    if model_type == "src_twt_only":
        src_posts = process_texts2(src_posts)
        return (src_posts, list(labels))
    elif model_type == "struct_levels":
        src_posts = process_texts2(src_posts)
        return (src_posts, list(structure), list(labels))
    elif model_type == "src_twt_all_replies":
        merged_threads = process_texts2(merged_threads)
        #print(merged_threads)
        return (merged_threads, list(labels))
    elif model_type == "hierarchical":
        merged_threads = [process_texts2(thread) for thread in merged_threads]
        #print(merged_threads)
        return (merged_threads, list(labels))
    elif model_type == 'stanc_det':
        replies = list(replies)
        #print(replies)
        #print(replies[0])
        
        post_ids, posts = list(src_post_ids), list(src_posts)
        for thread in replies:
            #print(len(thread[0]))
            for reply in thread: 
                post_ids.append(reply[0])
                posts.append(reply[1])
        posts = process_texts2(posts)
        return post_ids, posts
    elif model_type == "llm_stanc_distr":
        src_posts = process_texts2(src_posts)
        return src_posts, stanc_distrs, list(labels)
    elif model_type == "BiLSTM":
        rep_stancs_labl = zip(rep_stances, labels)
        rep_stances, labels = list(), list() 
        for stances, label in rep_stancs_labl:
            #remove posts without replies, and replies without stances
            if len(stances) == 0:
                continue
            else:
                rep_stances.append(stances)
                labels.append(label)
        return rep_stances, labels
    elif model_type == "BiLSTM_LLM":
        rep_stancs_labl = zip(src_posts, rep_stances, labels)
        rep_stances, labels, src_posts = list(), list(), list() 
        for src_post, stances, label in rep_stancs_labl:
            #remove posts without replies, and replies without stances
            if len(stances) == 0:
                continue
            else:
                src_posts.append(src_post)
                rep_stances.append(stances)
                labels.append(label)
        src_posts = process_texts2(src_posts)
        return src_posts, rep_stances, labels
    elif model_type == "source_agg_replies":
        posts_stancs_labl = zip(posts, posts_stances, rep_stances, labels)
        posts, posts_stances, labels = list(), list(), list()
        
        for thread, stances, rep_stancs, label in posts_stancs_labl:
            #remove posts without replies, and replies without stances
            if len(rep_stancs) == 0:
                continue
            else:
                posts.append(thread)
                posts_stances.append(stances)
                labels.append(label)
        posts = [process_texts2(thread) for thread in posts]
        return posts, list(posts_stances), list(labels)
    elif model_type == "agg_stanc_struct":
        t_data = zip(posts, posts_stances, rep_stances, stanc_distrs, structure, labels)
        posts, posts_stances, stanc_distrs, structure, labels = list(), list(), list(), list(), list() 
        for thread, stances, rep_stancs, stanc_distr, struct, label in t_data:
            if len(rep_stancs) == 0:
                continue
            else:
                posts.append(thread)
                posts_stances.append(stances)
                stanc_distrs.append(stanc_distr)
                structure.append(struct)
                labels.append(label)
        posts = [process_texts2(thread) for thread in posts]
        return posts, posts_stances, stanc_distrs, structure, labels 
        
    #return src_posts, list(structure)

if __name__ == "__main__":
    #data = pd.DataFrame()
    all_posts = list()
    all_stances = list()
    all_labels = list()
    all_stanc_distrs = list()
    all_structs = list()
    count = 0
    # Initialize an empty dictionary
    data_dict = {}
    count_stances = 0
    count_nan = 0
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
        #src_posts, stanc_distrs, labels = read_data(dir_path, model_type="llm_stanc_distr")
        posts, stances, stanc_distrs, structs, labels = read_data(dir_path, model_type="agg_stanc_struct")
        all_posts.extend(posts)
        all_stances.extend(stances)
        all_stanc_distrs.extend(stanc_distrs)
        all_structs.extend(structs)
        all_labels.extend(labels)
        count += 1
        '''
        if count == 9:
            break
        '''
        #all_stanc_distrs.extend(stanc_distrs)
        '''
        topic_data = read_data(dir_path, model_type="src_twt_only")
        data_dict[topic] = topic_data
        '''
        #data = pd.concat([data, topic_data], ignore_index=True)
    
    for stances in all_stances:
        if len(stances) == 0:
            count_nan += 1
        print(stances)
        print(len(stances))
        count_stances += len(stances)
        print()
    #print(all_posts)
    #print(all_stanc_distrs)
    print(len(all_posts))
    print(len(all_stances))
    print(len(all_stanc_distrs))
    print(len(all_structs))
    print(len(all_labels))
    print("Stances count: ",count_stances)
    print("Nan samples: ", count_nan)
    
    #print(len(all_stanc_distrs))
    #print(all_replies)
    #count_text_lengths(all_posts)
    #mean_length, max_length, min_length = count_replies_lengths(all_posts)
    #print(f"Mean length: {mean_length}, Max length: {max_length}, Min length: {min_length}")
    
    '''   
    # Print the dictionary to check
    for key, df in data_dict.items():
        print(f"Data for {key}:\n{df}\n")
        print()
    '''
    # Display the output
    '''
    print(data)
    print("Unique Veracity Labels:", data["Veracity"].unique())
    value_counts = data['Veracity'].value_counts()
    print("Veracity Value Counts:\n", value_counts)
    print(len(data))
    print("Unique Veracity Labels:", data["Veracity_int"].unique())
    '''
