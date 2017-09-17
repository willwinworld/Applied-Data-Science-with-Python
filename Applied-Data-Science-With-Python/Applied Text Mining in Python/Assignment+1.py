
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[30]:

import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
#         print(line)
        doc.append(line)
print(doc)
df = pd.Series(doc)
df.head(10)


# In[28]:

# import re
# import pandas as pd

# transform_dict = {'Jan': 1, 'January': 1, 'Feb': 2, 'February': 2,
#                   'Mar': 3, 'March': 3, 'Apr': 4, 'April': 4, 'May': 5,
#                   'Jun': 6, 'June': 6, 'Jul': 7, 'July': 7, 'Aug': 8,
#                   'August': 8, 'Sep': 9, 'September': 9, 'Oct': 10,
#                   'October': 10, 'Nov': 11, 'November': 11, 'Dec': 12,
#                   'December': 12}

# def date_sorter():
#     doc = []
#     with open('dates.txt') as file:
#         for line in file:
# #         print(line)
#             doc.append(line)
# #     print(doc)
#     df = pd.Series(doc)
#     pd.set_option("display.max_rows", 999)
# #     print(df.iloc[10])
# #     print('------------------------------------')
# #     print(df.str.findall(r'\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{3}|\d{1,2}/\d{1,2}/\d{2}'))
#     number_regex = df.str.findall(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}[-]\d{1,2}[-]\d{2,4}|\d{1,2}/\d{2,4}')
# #     print(number_regex)
# #     print('------------------------------------')
#     for ind, val in number_regex.iteritems():
#         if len(val) > 0:
#             df.iloc[ind] = val
# #     print(df)
# #     print('---------------------------------------')
# #     print(df.iloc[125])
#     """
#     将数字类型的全部提取并替换了
#     """
#     for ind, val in df.iteritems():
#         if not isinstance(val, list):
#             df.iloc[ind] = re.findall(r'\w{3,}[-]\d+[-]\d{2,4}|\d{1,2}\s\w{3,}\s\d{2,4}|\w{3,}\s\d{2,4},\s\d{2,4}|\.\w{3,}\s\d{2,4},\s\d{2,4}|\w{3,}\.\s\d{2,4},\s\d{2,4}|\w{3,}\s\d{2,}\s\d{2,}|\w{3,}\s\d{2,}|\w{3,},\s\d{2,4}|\d{4}|p\w{3,}\s\d{2,4}', val)
#             if len(df.iloc[ind]) == 0:
#                 df.iloc[ind] = val
# #     print(df)
# #     print('----------------------------------------')
# #     print(df.iloc[10])
# #     print('----------------------------------------')
#     """
#     处理正则提取出列表长度为2的文本
#     """
#     df.iloc[10] = [df.iloc[10][0]]
#     df.iloc[72] = [df.iloc[72][0]]
# #     print('----------------------------------------')
# #     print(df)
# #     print('----------------------------------------')
#     """
#     将全部日期格式转换成三个斜杠格式
#     """
#     for ind, val in df.iteritems():
#         text = val[0]
#         if '/' in text:
#             split_text_list = text.split('/')
#             if len(split_text_list) == 2:
#                 split_text_list.insert(0, '1')
#                 df.iloc[ind] = '/'.join(split_text_list)
#             elif len(split_text_list) == 3:
#                 df.iloc[ind] = text
#         elif '-' in text:
#             split_text_list = text.split('-')
#             df.iloc[ind] = '/'.join(split_text_list)
#         elif len(text) == 4:
#             df.iloc[ind] = '1'+'/'+'1'+'/'+text
#         else:
#             raw_list = re.findall(r'\d{2}\s\w{3,}\s\d{4}', text)
#             if len(raw_list) > 0:
#                 empty_space_split_list = raw_list[0].split(' ')
#                 if empty_space_split_list[1] in transform_dict.keys():
#                     empty_space_split_list[1] = str(transform_dict[empty_space_split_list[1]])
#                     df.iloc[ind] = '/'.join(empty_space_split_list)
# #             raw_list_two = re.findall(r'\d{2},\s\w{3,}\s\d{4}', text)
# #             print(raw_list_two)
# #     print('---------------------------------------')
# #     print(df)
#     df.iloc[248] = '7/1/1995'
#     df.iloc[271] = '8/1/2008'
#     df.iloc[297] = '1/1/2009'
#     df.iloc[326] = '10/1/1995'
#     df.iloc[331] = '12/1/1993'
#     df.iloc[490] = '1/1/2007'
#     for ind, val in df.iteritems():
#         if isinstance(val, list):
#             text = ' '.join(val)
#             raw_list = re.findall(r'\d{2}\s\w{3,}\s\d{4}', text)
#             if len(raw_list) > 0:
#                 empty_space_split_list = raw_list[0].split(' ')
#                 empty_space_split_list[1] = str(transform_dict[empty_space_split_list[1]])
#                 df.iloc[ind] = '/'.join(empty_space_split_list)
#                 print(df.iloc[ind])
# #                 print(empty_space_split_list)
#     print(df)
#     print('-------------------------------------')
# #     df.iloc[194] = 
    

# date_sorter()


# In[40]:

def date_sorter():
    
    # Your code here
    # Full date
    global df
    dates_extracted = df.str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>\d?\d)[/|-](?P<year>\d{4}))')
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>([0-2]?[0-9])|([3][01]))[/|-](?P<year>\d{2}))'))
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    del dates_extracted[3]
    del dates_extracted[4]
    dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<day>\d?\d) ?(?P<month>[a-zA-Z]{3,})\.?,? (?P<year>\d{4}))'))
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>[a-zA-Z]{3,})\.?-? ?(?P<day>\d\d?)(th|nd|st)?,?-? ?(?P<year>\d{4}))'))
    del dates_extracted[3]
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    
    # Without day
    dates_without_day = df[index_left].str.extractall('(?P<origin>(?P<month>[A-Z][a-z]{2,}),?\.? (?P<year>\d{4}))')
    dates_without_day = dates_without_day.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>\d\d?)/(?P<year>\d{4}))'))
    dates_without_day['day'] = 1
    dates_extracted = dates_extracted.append(dates_without_day)
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    
    # Only year
    dates_only_year = df[index_left].str.extractall(r'(?P<origin>(?P<year>\d{4}))')
    dates_only_year['day'] = 1
    dates_only_year['month'] = 1
    dates_extracted = dates_extracted.append(dates_only_year)
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

    # Year
    dates_extracted['year'] = dates_extracted['year'].apply(lambda x: '19' + x if len(x) == 2 else x)
    dates_extracted['year'] = dates_extracted['year'].apply(lambda x: str(x))

    # Month
    dates_extracted['month'] = dates_extracted['month'].apply(lambda x: x[1:] if type(x) is str and x.startswith('0') else x)
    month_dict = dict({'September': 9, 'Mar': 3, 'November': 11, 'Jul': 7, 'January': 1, 'December': 12,
                       'Feb': 2, 'May': 5, 'Aug': 8, 'Jun': 6, 'Sep': 9, 'Oct': 10, 'June': 6, 'March': 3,
                       'February': 2, 'Dec': 12, 'Apr': 4, 'Jan': 1, 'Janaury': 1,'August': 8, 'October': 10,
                       'July': 7, 'Since': 1, 'Nov': 11, 'April': 4, 'Decemeber': 12, 'Age': 8})
    dates_extracted.replace({"month": month_dict}, inplace=True)
    dates_extracted['month'] = dates_extracted['month'].apply(lambda x: str(x))

    # Day
    dates_extracted['day'] = dates_extracted['day'].apply(lambda x: str(x))

    # Cleaned date
    dates_extracted['date'] = dates_extracted['month'] + '/' + dates_extracted['day'] + '/' + dates_extracted['year']
    dates_extracted['date'] = pd.to_datetime(dates_extracted['date'])

    dates_extracted.sort_values(by='date', inplace=True)
    df1 = pd.Series(list(dates_extracted.index.labels[0]))
    print(df1)
    
    return df1# Your answer here

date_sorter()


# In[ ]:



