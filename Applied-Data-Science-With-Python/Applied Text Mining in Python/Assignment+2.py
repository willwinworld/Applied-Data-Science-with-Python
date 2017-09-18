
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[3]:

import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)  # 分词
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[ ]:

def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[ ]:

def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[ ]:

from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[11]:

import nltk

def answer_one():
    total_word_list = nltk.word_tokenize(moby_raw)
    unique_word_set = set(total_word_list)
    total_length = len(total_word_list)
    unique_length = len(unique_word_set)
    float_res = unique_length / total_length
#     print(float_res)
    return float_res

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[21]:

def answer_two():
    total_word_list = nltk.word_tokenize(moby_raw)
#     print(total_word_list)
    word_list = ['whale', 'Whale']
    count = 0
    for word in total_word_list:
        if word == word_list[0] or word == word_list[1]:
            count += 1
    float_res = count / len(total_word_list)
    return float_res

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[28]:

def answer_three():
    word = nltk.word_tokenize(moby_raw)
    dist = nltk.FreqDist(word)
    vocab = dist.keys()
    twenty_frequently_token = list(vocab)[:20]
#     print(dist[twenty_frequently_token[0]])
    unsorted_result = [(x,dist[x]) for x in twenty_frequently_token]
    result = sorted(unsorted_result, key=lambda x: x[1], reverse=True)
    return result

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[23]:

def answer_four():
    word = nltk.word_tokenize(moby_raw)
    dist = nltk.FreqDist(word)
    unsorted_non_repeat_result = set([(w,dist[w]) for w in word if len(w) > 5 and dist[w] > 150])
    sorted_tuple_result = sorted(unsorted_non_repeat_result, key=lambda x: x[1], reverse=True)
    result = [x[0] for x in sorted_tuple_result]
    return result

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[28]:

def answer_five():
    unique_word = list(set(text1))
    unique_word_list = [(w, len(w)) for w in unique_word]
    result = max(unique_word_list, key=lambda item: item[1])
    return result

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[31]:

def answer_six():
    dist = nltk.FreqDist(text1)
    vocab = dist.keys()
    freqword = sorted([(dist[w], w) for w in vocab if w.isalpha() and dist[w] > 2000], key=lambda x: x[0], reverse=True)
    return freqword# Your answer here

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[7]:

def answer_seven():
    sentence = nltk.sent_tokenize(moby_raw)
    word = nltk.word_tokenize(moby_raw)
    sentence_length = len(sentence)
    word_length = len(word)
    average_token = word_length / sentence_length
    return average_token

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[16]:

def answer_eight():
    sentence = nltk.sent_tokenize(moby_raw)
    dist = nltk.FreqDist(sentence)
    vocab = dist.keys()
    unsorted_list = [(s,dist[s]) for s in vocab]
    sorted_list = sorted(unsorted_list, key=lambda x: x[1], reverse=True)[:5]
    return sorted_list

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[25]:

from nltk.corpus import words

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[33]:

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
   result = []
   import operator
   for entry in entries:
       spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
#        print(spell_list)
       distance_list = [(spell, nltk.jaccard_distance(set(nltk.ngrams(entry, n=3)), set(nltk.ngrams(spell, n=3)))) for spell in spell_list]
#        print(distance_list)
#        result.append(sorted(distance_list, key=operator.itemgetter(1))[0][0])
       result.append(sorted(distance_list, key=lambda x: x[1])[0][0]) 
    
   return result
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[ ]:

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    result = []
    import operator
    for entry in entries:
        spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
        distance_list = [(spell, nltk.jaccard_distance(set(nltk.ngrams(entry, n=4)), set(nltk.ngrams(spell, n=4)))) for spell in spell_list]

        result.append(sorted(distance_list, key=operator.itemgetter(1))[0][0])
    
    return result # Your answer here
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[34]:

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
   result = []
   import operator
   for entry in entries:
       spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
       distance_list = [(spell, nltk.edit_distance(entry, spell, transpositions=True)) for spell in spell_list]

       result.append(sorted(distance_list, key=operator.itemgetter(1))[0][0])
    
   return result# Your answer here 
       
answer_eleven()


# In[ ]:



