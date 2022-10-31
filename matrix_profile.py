# Isn't it better if we just read it from h5 file directly???
import pandas as pd
import matplotlib.pyplot as plt

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
6. Tcount : counts used to define movements
'''

tcount = test._get_Tcount() 
temp = tcount.R.values
Tmov = np.zeros(len(temp))
nonzeroTC = np.where(temp != 0)[0]
for i, j in enumerate(nonzeroTC[:-1]):
    if (np.diff([j, nonzeroTC[i+1]])[0] > 8) or (Tmov[j] == 1):
        continue
    else:
        if np.sign(temp[j]) != np.sign(temp[nonzeroTC[i+1]]):
            Tmov[nonzeroTC[i+1]] = 1
        else:
            continue

valmags = test._get_mag('Gyroscope', test.row_idx)
acounts = test._get_count()

tcount = test._get_Tcount()

fig, ax = plt.subplots(1)
ax.plot(test.accmags.lmag[134500:135500], marker='o', c = 'pink')
ax.axhline(y=test.laccth, color = 'b', linestyle='--')
ax.axhline(y=test.lnaccth, color = 'b', linestyle='--')
ax.stem(np.arange(134500, 135500), tcount.L[134500:135500])
ax.stem(np.arange(134500, 135500), test.Tmov.L[134500:135500]*2, markerfmt = '+')

plt.show()

plt.plot(test.accmags.lmag.values)
plt.show()

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

