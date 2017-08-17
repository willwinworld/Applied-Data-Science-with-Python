#! python3
# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


def get_list_of_university_towns():
    data = []
    with open("university_towns.txt", "r") as f:
        for line in f.readlines():
            m = re.search('(.+)\[edit\]', line)
            if m:
                state = m.group(1)
            else:
                town = line.split('(')[0].strip()  # 当没有匹配到state时，上一个在解释器中的state还并没有消失
                data.append([state, town])
    df = pd.DataFrame(data, columns=["State", "RegionName"])
    return df


def get_recession_start():
    gdp = pd.ExcelFile("gdplev.xls")
    gdp = gdp.parse(skiprows=7)
    gdp = gdp[['Unnamed: 4', 'Unnamed: 5']]
    gdp = gdp.loc[212:]
    gdp.columns = ['Quarter', 'GDP']
    gdp['GDP'] = pd.to_numeric(gdp['GDP'])
    quarters = []
    for i in range(len(gdp) -2):
        if (gdp.iloc[i][1] > gdp.iloc[i+1][1]) and (gdp.iloc[i+1][1] > gdp.iloc[i+2][1]):
            quarters.append(gdp.iloc[i][0])
    # print(quarters)
    return quarters[0]


def get_recession_end():
    gdp = pd.ExcelFile("gdplev.xls")
    gdp = gdp.parse(skiprows=7)
    gdp = gdp[['Unnamed: 4', 'Unnamed: 5']]
    gdp = gdp.loc[212:]
    gdp.columns = ['Quarter', 'GDP']
    gdp['GDP'] = pd.to_numeric(gdp['GDP'])
    start = get_recession_start()
    start_index = int(gdp.loc[gdp['Quarter'] == start].index[0])
    gdp = gdp.loc[start_index:]
    quarters = []
    for i in range(len(gdp) - 2):
        if (gdp.iloc[i][1] < gdp.iloc[i+1][1]) and (gdp.iloc[i+1][1] < gdp.iloc[i+2][1]):
            quarters.append(gdp.iloc[i][0])
    # print(quarters)
    return quarters[0]


def get_recession_bottom():
    gdp = pd.ExcelFile("gdplev.xls")
    gdp = gdp.parse(skiprows=7)
    gdp = gdp[['Unnamed: 4', 'Unnamed: 5']]
    gdp = gdp.loc[212:]
    gdp.columns = ['Quarter', 'GDP']
    gdp['GDP'] = pd.to_numeric(gdp['GDP'])
    start = get_recession_start()
    start_index = int(gdp.loc[gdp['Quarter'] == start].index[0])
    end = get_recession_end()
    end_index = int(gdp.loc[gdp['Quarter'] == end].index[0])
    part = gdp.loc[start_index:end_index]
    min_gdp = np.nanmin(part[['GDP']].values)
    min_index = int(gdp.loc[gdp['GDP'] == min_gdp].index[0])
    gdp_bottom = gdp.loc[min_index]
    return gdp_bottom['Quarter']


# def new_col_names():
#     # generating the new coloumns names
#     years = list(range(2000, 2017))
#     quars = ['q1', 'q2', 'q3', 'q4']
#     quar_years = []
#     for i in years:
#         for x in quars:
#             quar_years.append((str(i) + x))
#     # print(quar_years[:67])
#     return quar_years[:67]
#
#
# def convert_housing_data_to_quarters1():
#     '''Converts the housing data to quarters and returns it as mean
#     values in a dataframe. This dataframe should be a dataframe with
#     columns for 2000q1 through 2016q3, and should have a multi-index
#     in the shape of ["State","RegionName"].
#
#     Note: Quarters are defined in the assignment description, they are
#     not arbitrary three month periods.
#
#     The resulting dataframe should have 67 columns, and 10,730 rows.
#     '''
#     data = pd.read_csv('City_Zhvi_AllHomes.csv')
#     data.drop(['Metro', 'CountyName', 'RegionID', 'SizeRank'], axis=1, inplace=True)
#     data['State'] = data['State'].map(states)
#     data.set_index(['State', 'RegionName'], inplace=True)
#     col = list(data.columns)
#     col = col[0:45]
#     data.drop(col, axis=1, inplace=True)
#
#     # qs is the quarters of the year
#     qs = [list(data.columns)[x:x + 3] for x in range(0, len(list(data.columns)), 3)]
#
#     # new columns
#     column_names = new_col_names()
#     for col, q in zip(column_names, qs):
#         data[col] = data[q].mean(axis=1)
#
#     data = data[column_names]
#     # print(data)
#     return data


def convert_housing_data_to_quarters():
    hdata = pd.read_csv('City_Zhvi_AllHomes.csv')
    hdata = hdata.drop(hdata.columns[[0]+list(range(3,51))], axis=1)
    target = pd.DataFrame(hdata[['State', 'RegionName']])
    for year in range(2000, 2016):
        target[str(year)+'q1'] = hdata[[str(year)+'-01', str(year)+'-02', str(year)+'-03']].mean(axis=1)
        target[str(year)+'q2'] = hdata[[str(year)+'-04', str(year)+'-05', str(year)+'-06']].mean(axis=1)
        target[str(year)+'q3'] = hdata[[str(year)+'-07', str(year)+'-08', str(year)+'-09']].mean(axis=1)
        target[str(year)+'q4'] = hdata[[str(year)+'-10', str(year)+'-11', str(year)+'-12']].mean(axis=1)
    year = 2016
    target[str(year) + 'q1'] = hdata[[str(year) + '-01', str(year) + '-02', str(year) + '-03']].mean(axis=1)
    target[str(year) + 'q2'] = hdata[[str(year) + '-04', str(year) + '-05', str(year) + '-06']].mean(axis=1)
    target[str(year) + 'q3'] = hdata[[str(year) + '-07', str(year) + '-08', str(year) + '-09']].mean(axis=1)
    target = target.set_index(['State', 'RegionName'])
    # print(target)
    return target


def run_ttest():
    """
    First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values,
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence.

    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).
    2008q3-2009q2 经济消退开始-经济消退谷底
    """
    housing_data = convert_housing_data_to_quarters()
    data = housing_data.loc[:, '2008q3': '2009q2']
    # print(data.head())
    data = data.reset_index()
    # print('-----------------')
    # print(data.head())

    def price_change(row):
        return (row['2008q3'] - row['2009q2']) / row['2008q3']
    data['up&down'] = data.apply(price_change, axis=1)
    # print(data.head())

    university_towns = get_list_of_university_towns()['RegionName']
    university_towns = set(university_towns)

    def is_uni_town(row):
        if row['RegionName'] in university_towns:
            return 1
        else:
            return 0
    data['is_uni'] = data.apply(is_uni_town, axis=1)

    not_uni = data[data['is_uni']==0].loc[:, 'up&down'].dropna()
    is_uni = data[data['is_uni']==1].loc[:, 'up&down'].dropna()

    def better():
        if not_uni.mean() < is_uni.mean():
            return 'non-university town'
        else:
            return 'university town'
    p_val = ttest_ind(not_uni, is_uni)[1]
    print(p_val)
    res = (True, p_val, better())
    print(res)
    return res


if __name__ == "__main__":
    # res1 = get_list_of_university_towns()
    # res2 = get_recession_start()
    # print(res2)
    # get_recession_end()
    # res3 = get_recession_bottom()
    # print(res1)
    # print(res2)
    # print(res3)
    # print(res4)
    # convert_housing_data_to_quarters()
    # convert_housing_data_to_quarters1()
    run_ttest()
