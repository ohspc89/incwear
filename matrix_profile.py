# Isn't it better if we just read it from h5 file directly???
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

#sys.path.append('..')
# This is the script I wrote to read raw data from h5 file directly
#   and do some pre-processing (up to magnitude calculation)
from ReadSensorLog import *

# Move to the directory where the h5 files are stored
#   so that you can later use dir to list all filenames at once and loop around
filename = '/home/jinseok/Downloads/20200217-082740_106v1.h5'

# Before reading an individual h5 file, we need to read the csv file from REDCap.
redcap_file = pd.read_csv('~/Downloads/WearableSensorsGuate_DATA_2022-10-28_1019.csv')

# fullid = infant_id + visit (ex. '106v1')
fullid = filename.split('_')[-1].split('.h5')[0]
# time_pts will be the recording start and end times (ex. '10:23', '22:42')
time_pts = find_timepts_from_redcap(redcap_file, fullid)
# time_pts_dt will be in the datetime format
#   (ex. datetime(2020, 2, 17, 10, 23, 00))
time_pts_dt = make_start_end_datetime(time_pts, filename)

# Up to this point works (Oct.18, 22)
test = Subject(filename, time_pts_dt)   # input: a h5 filename (required)
                                        #      : time_pts_dt (optional), det_option (optional, default = 'median')
                                        # You either specify the datetimes of recording start and end
                                        # or it will simply analyze the entire recording.
                                        # To test another scenario, try running it without any parameter.
                                        # You can also specify the detrending option. Default is subtracting the 'median'
'''
Once you generate a [Subject] object, it will contain some useful variables.
1. accmags: a DataFrame storing the accelerometer signal norms (L/R)
2. laccth : left accelerometer value threshold
3. raccth : right accelerometer value threshold
4. lnaccth: left accelerometer value negative threshold
5. rnaccth: right accelerometer value negative threshold
6. Tmov : counts that correspond to full movements
'''

tcount = test._get_Tcount()

a = 137800
b = 137900
fig, ax = plt.subplots(2)
ax[0].plot(test.accmags.lmag[a:b], marker='o', c = 'pink')
ax[0].axhline(y=test.laccth, color = 'b', linestyle='--')
ax[0].axhline(y=test.lnaccth, color = 'b', linestyle='--')
#ax[0].stem(np.arange(a, b), tcount.L[a:b])
ax[0].stem(np.arange(a, b), test.Tmov.L[a:b]*2, markerfmt = '+')
rect1 = Rectangle((137839, 0), width=3, height=1, ec='g', fc='none', lw=2)
rect2 = Rectangle((137845, 0), width=5, height=1, ec='g', fc='none', lw=2)
ax[0].add_patch(rect1)
ax[0].add_patch(rect2)
ax[1].plot(test.accmags.rmag[a:b], marker='o', c='pink')
ax[1].axhline(y=test.raccth, color = 'b', linestyle='--')
ax[1].axhline(y=test.rnaccth, color = 'b', linestyle='--')
#ax[1].stem(np.arange(a, b), tcount.R[a:b])
ax[1].stem(np.arange(a, b), test.Tmov.R[a:b]*2, markerfmt = '+')
rect1 = Rectangle((137845, 0), width=4, height=1, ec='g', fc='none', lw=2)
rect2 = Rectangle((137839, 0), width=5, height=1, ec='g', fc='none', lw=2)
ax[1].add_patch(rect1)
ax[1].add_patch(rect2)
plt.show()

test.kinematics['Rkinematics']['RavepMov']
test.kinematics['Lkinematics']['LavepMov']

r_dict = test.kinematics['Rkinematics'].copy()
l_dict = test.kinematics['Lkinematics'].copy()

# 137847
r_dict['RMovStart'][47]
r_dict['RMovStart'][48]
r_dict['RMovEnd'][48]
l_dict['LMovStart'][77]
l_dict['LMovEnd'][77]

r_dict['RMovStart'][46]
l_dict['LMovStart'][76]

sole_r = pd.DataFrame(data = {'RMovIdx':r_dict['RMovIdx']})
sole_l = pd.DataFrame(data = {'LMovIdx':l_dict['LMovIdx']})

sole_r['range'] = list(np.arange(int(x), int(y)) for x, y in zip(r_dict['RMovStart'], r_dict['RMovEnd']))
sole_l['range'] = list(np.arange(int(x), int(y)) for x, y in zip(l_dict['LMovStart'], l_dict['LMovEnd']))

set(sole_r.range)

import time
tic = time.perf_counter()
simul_rl = np.array(list(filter(lambda tup: tup in np.concatenate(sole_l.range), sole_r.RMovIdx)))
toc = time.perf_counter()
toc-tic
simul_lr = np.array(list(filter(lambda tup: tup in np.concatenate(sole_r.range), sole_l.LMovIdx)))
simul_lr

np.intersect1d(simul_rl, simul_lr)
sole_r.range

simul_rl[10]
len(simul_rl)
len(sole_r.range)
[tup for tup in sole_r.RMovIdx if tup in sole_l.range]
sole_r.RMovIdx[:-20]
sole_l.range

out = test._mark_simultaneous()
out
list(zip(test.kinematics.RMovStart, test.kinematics.RMovEnd))

temp3 = np.nonzero(test.kinematics.RMovIdx.values)[0]
temp3
sum(test.Tmov.R)

Tmovidx = np.nonzero(test.Tmov.L.values)[0]
len(Tmovidx)
len(test.kinematics.LMovIdx.values)
test.kinematics.LMovIdx.values
len(test.Tmov.L.values)


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

