import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from incwear.preprocessing import opalv2, make_start_end_datetime

# Move to the directory where the h5 files are stored
#   so that you can later use dir to list all filenames at once and loop around
FNAME = '/Users/joh/Documents/PYTHON_porting/MATLAB/h5files/20201111-113232_117v1.h5'
FNAME = '/Users/joh/Documents/PYTHON_porting/MATLAB/h5files/20210301-091819_132v2.h5'

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
test = opalv2(FNAME, time_pts_dt, LABEL_R)

# check if movement counts were correctly done
test.plot_segment(time_passed = 10000, duration = 100)

# you can check the right side as well
test.plot_segment(time_passed = 10000, duration = 100, side='R')

# you can directly check where the movements occurred
movidx = test.get_mov()

# average acceleration / peak acceleration per movement
accmeasures = test.acc_per_mov()
accmeasures

# If you want to check the simultaneous moves...
movidx_r = test.get_mov('R')

# Synchronous: if two movements (L/R) have the identical mov start times,
#              they are 'bilateral synchronous'
bilat_sync = np.intersect1d(movidx[:,0], movidx_r[:,0])

# If the two movements don't share the same mov start time,
#   but overlap, they are bilateral asynchronous
# The definition of the overlap is for the "middle point" of
#   a movement is located within the duration of another movement.
temp_r = list(map(lambda x: np.arange(x[0], x[1]), zip(movidx_r[:,0], movidx_r[:,2])))
all_ridx = np.concatenate(temp_r)
# find the "middle points" of left movements that are within
#   the indices of all right movements
l_overlap = np.intersect1d(movidx[:,1], all_ridx)

# overlaps would contain bilateral sync movements as well.
# eliminate them and you get pure bilateral asynchronous movements.
bilat_async_l = np.setdiff1d(l_overlap, bilat_sync)

temp_l = list(map(lambda y: np.arange(y[0], y[1]), zip(movidx_r[:,0], movidx[:,2])))
all_lidx = np.concatenate(temp_l)
r_overlap = np.intersect1d(movidx_r[:,1], all_lidx)

bilat_async_r = np.setdiff1d(r_overlap, bilat_sync)

test.plot_segment(time_passed = 250, duration = 100, side='R')

lmag = test.measures.accmags['lmag']
rmag = test.measures.accmags['rmag']
sidx = 5000; eidx = 5300
lmov_start = np.where(movidx[:,0] > sidx)[0][0]
lmov_end = np.where(movidx[:,2] < eidx)[0][-1]
rmov_start = np.where(movidx_r[:,0] > sidx)[0][0]
rmov_end = np.where(movidx_r[:,2] < eidx)[0][-1]

_, ax = plt.subplots(1)
xcoord = np.arange(sidx, eidx)
ax.plot(xcoord, lmag[sidx:eidx], c='pink', marker='o')
ax.plot(xcoord, rmag[sidx:eidx], c='lightblue', marker='+')
ax.axhline(y=0, c='k', linestyle='--')
for i in range(lmov_start, lmov_end+1):
    movxcoord = np.arange(movidx[i,0], movidx[i,2])
    ax.plot(movxcoord, lmag[movxcoord], c='r')

for j in range(rmov_start, rmov_end+1):
    movxcoord2 = np.arange(movidx_r[j,0], movidx_r[j,2])
    ax.plot(movxcoord2, rmag[movxcoord2], c='b')

plt.show()
