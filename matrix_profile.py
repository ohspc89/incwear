# Isn't it better if we just read it from h5 file directly???
import sys
import matrixprofile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

sys.path.append('..')
# This is the script I wrote to read raw data from h5 file directly
#   and do some pre-processing (up to magnitude calculation)
from ReadSensorLog import *

# Move to the directory where the h5 files are stored
#   so that you can later use dir to list all filenames at once and loop around
filename = '/Users/joh/Documents/PYTHON_porting/MATLAB/h5files/20200217-082740_106v1.h5'

# Before reading an individual h5 file, we need to read the csv file from REDCap.
redcap_file = pd.read_csv('~/Documents/PYTHON_porting/MATLAB/WearableSensorsGuate_DATA_2022-09-19_0904.csv')

# fullid = infant_id + visit
fullid = filename.split('_')[-1].split('.h5')[0]
time_pts = find_timepts_from_redcap(redcap_file, fullid)

# Up to this point works (Oct.18, 22)
test = Subject(filename, time_pts)

########
# almost implemented

# Timepoint
tp = test.left['Time'][0]

def calc_date(tp):
# div
    div = 24*3600*1e6

    # Days
    days = tp // div
    rem1 = tp % div

    # Hours (Guatemala: UTC-6)
    hrs = rem1 // (3600*1e6) - 6 
    rem2 = rem1 % (3600*1e6)

    # Minutes
    mins = rem2 // (60*1e6)
    rem3 = rem2 % (60*1e6)

    # Seconds
    secs = rem3 // 1e6

    start_date = datetime(1970, 1, 1, 0, 0, 0, 0) + timedelta(days = days,\
            seconds = secs, minutes = mins, hours = hrs, microseconds = rem3 % 1e6)
    return(start_date)



# make_start_end_datetime will return the datetimes that reflect
#   donned and doffed times
def make_start_end_datetime(don_doff, full_id, filename):

    donned_t, doffed_t = find_start_end(don_doff, full_id)
    # Getting year, month, and day from the filename
    temp = filename.split('/')[2].split('-')[0]
    year    = int(temp[0:4])
    month   = int(temp[4:6])
    day     = int(temp[6:8])
    # split hour and minute
    donned_h, donned_m = donned_t.split(':')
    doffed_h, doffed_m = doffed_t.split(':')
    # donned_dt could be the earliest point that matches donned_h
    donned_dt = datetime(year, month, day, int(donned_h), int(donned_m), 0, 0)
    # Let's be lenient, and give 30 seconds of datapoints more...
    if int(doffed_h) < 13 & int(doffed_h) > int(donned_h):
        doffed_dt = datetime(year, month, day+1, int(doffed_h), int(doffed_m), 30, 0)
    else:
        doffed_dt = datetime(year, month, day, int(doffed_h), int(doffed_m), 30, 0)
    return([donned_dt, doffed_dt])

in_en_dts = make_start_end_datetime(don_doff, full_id, test.filename)

# Here you're getting the time difference between the start_date and donned_dt / doffed_dt
#   in microseconds
diffs_in_microsec = map(lambda x: x - test.calc_datetime(test.left['Time'][0]), in_en_dts)

# This list will include indices of in and en
# It seems like map objects disappear once it's used...
indices = list(map(lambda x: round((x.seconds*1e6 + x.microseconds)/50000), diffs_in_microsec))

a = test._get_mag(dtype = 'Accelerometer', row_idx = [x for x in range(indices[0], indices[1])])
a.head

# An exemple figure similar to what's produced in MATLAB
fig, ax = plt.subplots()
ax.set_xlim([0, test.time[len(test.time)-1]])
ax.plot(test.time[range(indices[0], indices[1])], a)
plt.show()


