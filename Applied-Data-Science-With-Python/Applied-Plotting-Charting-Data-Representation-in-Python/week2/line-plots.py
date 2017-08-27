#! python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

linear_data = np.array([1,2,3,4,5,6,7,8])
exponential_data = linear_data**2

plt.figure()
# plot the linear data and the exponential data
plt.plot(linear_data, '-o', exponential_data, '-o')
# plt.show()

# plot another series with a dashed red line
plt.plot([22, 44, 55], '--r')
# print(plt.legend())
# plt.show()
plt.xlabel('Some data')
plt.ylabel('Some other data')
plt.title('A title')
# add a legend with legend entries (because we didn't have labels when we plotted the data series)
plt.legend(['Baseline', 'Competition', 'Us'])
plt.gca().fill_between(range(len(linear_data)), linear_data, exponential_data, facecolors='blue', alpha=0.25)
# plt.show()
# print(plt.figure())
# plt.show()
observation_dates = np.arange('2007-01-01', '2017-01-09', dtype='datetime64[D]')
observation_dates = map(pd.to_datetime, observation_dates)
plt.plot(observation_dates, linear_data, '-o', observation_dates, exponential_data, '-o')
