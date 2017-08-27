#! python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
# import mplleaflet
# import pandas as pd
#
#
# def leaflet_plot_stations(binsize, hashid):
#
#     df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))
#
#     station_locations_by_hash = df[df['hash'] == hashid]
#
#     lons = station_locations_by_hash['LONGITUDE'].tolist()
#     lats = station_locations_by_hash['LATITUDE'].tolist()
#
#     plt.figure(figsize=(8,8))
#
#     plt.scatter(lons, lats, c='r', alpha=0.7, s=200)
#
#     return mplleaflet.display()
#
# leaflet_plot_stations(400,'6c8d642f28d9321421519c91b4ae6955a5796edb65a8b14e2257a994')

"""
Before working on this assignment please read these instructions fully.
In the submission area, you will notice that you can click the link to
Preview the Grading for each step of the assignment.
This is the criteria that will be used for peer grading.
Please familiarize yourself with the criteria before beginning the assignment.
An NOAA dataset has been stored in the file
data/C2A2_data/BinnedCsvs_d400/6c8d642f28d9321421519c91b4ae6955a5796edb65a8b14e2257a994.csv.
The data for this assignment comes from a subset of The National Centers for Environmental
Information (NCEI) Daily Global Historical Climatology Network (GHCN-Daily).
The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
Each row in the assignment datafile corresponds to a single observation.
The following variables are provided to you:

id : station identification code
date : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
element : indicator of element type
TMAX : Maximum temperature (tenths of degrees C) 十分之一
TMIN : Minimum temperature (tenths of degrees C) 十分之一
value : data value for element (tenths of degrees C)
For this assignment, you must:

1.Read the documentation and familiarize yourself with the dataset,
then write some python code which returns a line graph of the record high and record
low temperatures by day of the year over the period 2005-2014.
# 线条图形 2005-2014年
2.The area between the record high and record low temperatures for each day should be shaded.
Overlay a scatter of the 2015 data for any points (highs and lows)
for which the ten year record (2005-2014) record high or record low was broken in 2015.
Watch out for leap days (i.e. February 29th),
# 
it is reasonable to remove these points from the dataset for the purpose of this visualization.
Make the visual nice! Leverage principles from the first module in this course
when developing your solution.

Consider issues such as legends, labels, and chart junk.
The data you have been given is near White Rock, British Columbia, Canada,
and the stations the data comes from are shown on the map below.

TMAX   
TMIN
将数据分成两部分
"""





