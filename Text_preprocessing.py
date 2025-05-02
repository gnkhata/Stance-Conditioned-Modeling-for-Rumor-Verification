# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:18:00 2024

@author: gnkhata
"""

import pandas as pd
import re

def process_texts2(texts):
    processed_texts = []
    
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    mention_pattern = re.compile(r'@\w+')
    hashtag_pattern = re.compile(r'#\w+')
    
    for text in texts:
        # Replace URLs with $URL$
        text = re.sub(url_pattern, '$url$', text)
        
        # Replace mentions with $mention$
        text = re.sub(mention_pattern, '$mention$', text)
        
        # Extract and separate hashtags into individual words
        hashtags = hashtag_pattern.findall(text)
        separated_hashtags = []
        
        for hashtag in hashtags:
            # Remove the '#' and split words while keeping numbers intact
            words = re.findall(r'[A-Za-z]+|\d+', hashtag[1:])
            separated_hashtags.append(words)
        
        # Flatten the list of hashtags and join them into the processed text
        flattened_hashtags = ' '.join([' '.join(words) for words in separated_hashtags])
        text = re.sub(hashtag_pattern, flattened_hashtags, text)
        
        processed_texts.append(text)
    
    return processed_texts
'''
# Example usage
texts = [
    "Check this out @user! https://example.com #CamelCaseHashtag2024 #anotherOne123",
    "Another example @anotherUser http://example.org #Python3 #100DaysOfCode"
]

processed = process_texts2(texts)
for text in processed:
    print(text)
'''