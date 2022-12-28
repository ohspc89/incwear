import pandas as pd

from incwear.Preprocessing import opalv2, make_start_end_datetime

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
