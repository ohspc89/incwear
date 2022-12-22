import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from incwear.subject import subject, make_start_end_datetime

# Move to the directory where the h5 files are stored
#   so that you can later use dir to list all filenames at once and loop around
FNAME = '/Users/joh/Documents/PYTHON_porting/MATLAB/h5files/20201111-113232_117v1.h5'

# Before reading an individual h5 file, we need to read the csv file from REDCap.
redcap_file = pd.read_csv('~/Downloads/mixed_bag.csv')

# time_pts_dt will be in the datetime format
#   (ex. datetime(2020, 2, 17, 10, 23, 00, tzinfo=datetime.timezone.utc))
time_pts_dt = make_start_end_datetime(redcap_file, FNAME, 'America/Guatemala')

# The class:subject takes three parameters.
#   1) h5 filename
#   2) a list of datetime objects that mark
#       the start and the end of recording
#   3) a string used to indicate the "right" side
LABEL_R = 'derecho'
test = subject(FNAME, time_pts_dt, LABEL_R)

# a subject contains the following attributes:
# tderivs (class)
#   - accmags (pd.DataFrame): detrended acc norms
#                           colnames = ['lmag', 'rmag']
#   - thresholds (dict): four acc thresholds;
#                           keys = ['laccth', 'lnaccth', 'raccth', 'rnaccth']
#   - over_accth (dict): data point greater than a positive threshold (T/F)
#                           keys = ['L', 'R']
#   - under_naccth (dict): data point smaller than a negative threshold (T/F)
#                           keys = ['L', 'R']
#   - th_crossed (dict): over a positive threshold (+1),
#                        under a negative one (-1),
#                        neither (0)
#                           keys = ['L', 'R']

# Tmov (pd.DataFrame): second crossing of a threshold per movement marked
#                           keys = ['L', 'R']
# kinematics (dict): kinematic variables

# If you want to do it step by step...
import h5py
from datetime import datetime, timezone
f = h5py.File(FNAME, 'r')
sensors = f['Sensors']
sensorids = list(sensors.keys())
label = sensors[sensorids[1]]['Configuration']\
        ['Config Strings'][0][2].decode()
ridx = LABEL_R in label
sensordict = {'L': sensors[sensorids[not ridx]],
              'R': sensors[sensorids[ridx]]}
# Up to this point, same result as the MATLAB code
rowidx = test._prep_row_idx(sensordict['L'], time_pts_dt)
print(datetime.fromtimestamp(sensordict['L']['Time'][1]/1e6, timezone.utc))

accmags = test._get_mag(sensordict, 'acc', rowidx)
angvels = test._get_mag(sensordict, 'gyro', rowidx)
acounts = test._get_count(accmags, angvels)
tcounts = test._get_tcount(acounts)


rowidx[0]
accmags = test.tderivs.accmags

a = 399641
b = 399878
fig, ax = plt.subplots(2)
ax[0].plot(accmags.lmag[a:b], marker='o', c = 'pink')
ax[0].axhline(y=test.tderivs.thresholds['laccth'], color = 'b', linestyle='--')
ax[0].axhline(y=test.tderivs.thresholds['lnaccth'], color = 'b', linestyle='--')
#ax[0].stem(np.arange(a, b), tcount.L[a:b])
ax[0].stem(np.arange(a, b), test.Tmov.L[a:b]*2, markerfmt = '+')
#rect1 = Rectangle((137839, 0), width=3, height=1, ec='g', fc='none', lw=2)
#rect2 = Rectangle((137845, 0), width=5, height=1, ec='g', fc='none', lw=2)
#ax[0].add_patch(rect1)
#ax[0].add_patch(rect2)
ax[1].plot(accmags.rmag[a:b], marker='o', c='pink')
ax[1].axhline(y=test.tderivs.thresholds['raccth'], color = 'b', linestyle='--')
ax[1].axhline(y=test.tderivs.thresholds['rnaccth'], color = 'b', linestyle='--')
#ax[1].stem(np.arange(a, b), tcount.R[a:b])
ax[1].stem(np.arange(a, b), test.Tmov.R[a:b]*2, markerfmt = '+')
#rect1 = Rectangle((137845, 0), width=4, height=1, ec='g', fc='none', lw=2)
#rect2 = Rectangle((137839, 0), width=5, height=1, ec='g', fc='none', lw=2)
#ax[1].add_patch(rect1)
#ax[1].add_patch(rect2)
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
