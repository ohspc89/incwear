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
import ReadSensorLog

# A 'Subject' object will have some basic features...
test = ReadSensorLog.Subject('./h5files/20200217-082740_106v1.h5')
test.left

times = list(test.rtime)
times[0]

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

# How to get in and en values? You need to make use of id_doc.csv file.
don_doff = pd.read_csv('id_doc.csv')
full_id = (test.filename.split('_')[1]).split('.h5')[0]

# find_start_end will return the times when the sensor was donned or doffed
def find_start_end(don_doff, full_id):
    # don_doff should be the table read from id_doc.csv file
    # full_id will be in the format: '@@@v$' where @@@ is the infant id and $ is the number of visit
    # times will be the sensor donned and doffed times, based on the full_id
    times = don_doff[don_doff.full_id == full_id][['time_donned', 'time_doffed']]
    return(times.values[0])

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


