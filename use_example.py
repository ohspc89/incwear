# Isn't it better if we just read it from h5 file directly???
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

#sys.path.append('..')
# This is the script I wrote to read raw data from h5 file directly
#   and do some pre-processing (up to magnitude calculation)
from incwear import *

# Move to the directory where the h5 files are stored
#   so that you can later use dir to list all filenames at once and loop around
filename = '/Users/joh/Documents/PYTHON_porting/MATLAB/h5files/20201111-113232_117v1.h5'

# Before reading an individual h5 file, we need to read the csv file from REDCap.
redcap_file = pd.read_csv('~/Downloads/mixed_bag.csv')

# time_pts_dt will be in the datetime format
#   (ex. datetime(2020, 2, 17, 10, 23, 00, tzinfo=datetime.timezone.utc))
time_pts_dt = make_start_end_datetime(redcap_file, filename, 'America/Guatemala')

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
sensors = test.sensors
rowidx = test._prep_row_idx(sensors['L'], time_pts_dt)
accmags = test._get_mag(sensors['L']['Accelerometer'], sensors['R']['Accelerometer'], rowidx)
a = 137800
b = 137900
fig, ax = plt.subplots(2)
ax[0].plot(accmags.lmag[a:b], marker='o', c = 'pink')
ax[0].axhline(y=test.thresholds.laccth, color = 'b', linestyle='--')
ax[0].axhline(y=test.thresholds.lnaccth, color = 'b', linestyle='--')
#ax[0].stem(np.arange(a, b), tcount.L[a:b])
ax[0].stem(np.arange(a, b), test.Tmov.L[a:b]*2, markerfmt = '+')
rect1 = Rectangle((137839, 0), width=3, height=1, ec='g', fc='none', lw=2)
rect2 = Rectangle((137845, 0), width=5, height=1, ec='g', fc='none', lw=2)
ax[0].add_patch(rect1)
ax[0].add_patch(rect2)
ax[1].plot(accmags.rmag[a:b], marker='o', c='pink')
ax[1].axhline(y=test.thresholds.raccth, color = 'b', linestyle='--')
ax[1].axhline(y=test.thresholds.rnaccth, color = 'b', linestyle='--')
#ax[1].stem(np.arange(a, b), tcount.R[a:b])
ax[1].stem(np.arange(a, b), test.Tmov.R[a:b]*2, markerfmt = '+')
rect1 = Rectangle((137845, 0), width=4, height=1, ec='g', fc='none', lw=2)
rect2 = Rectangle((137839, 0), width=5, height=1, ec='g', fc='none', lw=2)
ax[1].add_patch(rect1)
ax[1].add_patch(rect2)
plt.show()

test.kinematics['Rkinematics']['avepMov']
test.kinematics['Lkinematics']['avepMov']

r_dict = test.kinematics['Rkinematics'].copy()
l_dict = test.kinematics['Lkinematics'].copy()

# 137847
r_dict['MovStart'][47]
r_dict['MovStart'][48]
r_dict['MovEnd'][48]
l_dict['MovStart'][77]
l_dict['MovEnd'][77]

r_dict['MovStart'][46]
l_dict['MovStart'][76]

sole_r = pd.DataFrame(data = {'MovIdx':r_dict['MovIdx']})
sole_l = pd.DataFrame(data = {'MovIdx':l_dict['MovIdx']})

sole_r['range'] = list(np.arange(int(x), int(y)) for x, y in zip(r_dict['MovStart'], r_dict['MovEnd']))
sole_l['range'] = list(np.arange(int(x), int(y)) for x, y in zip(l_dict['MovStart'], l_dict['MovEnd']))

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
list(zip(test.kinematics['MovStart'], test.kinematics['MovEnd']))

temp3 = np.nonzero(test.kinematics['MovIdx'].values)[0]
temp3
sum(test.Tmov.R)

Tmovidx = np.nonzero(test.Tmov.L.values)[0]
len(Tmovidx)
len(test.kinematics.LMovIdx.values)
test.kinematics.LMovIdx.values
len(test.Tmov.L.values)
