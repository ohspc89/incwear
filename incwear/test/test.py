"""
test.py is prepared to showcase how you can analyze
spontaneous leg movements of infants recorded using Ax6
wearable sensors (axivity.com/product/ax6).
The script assumes either .cwa or .tsv file(s) as an input.

To learn more about the leg movement detection algorithm, refer to:
    Smith et al. (2015) Daily Quantity of Infant Leg movement:
        Wearable Sensor Algorithm and Relationship to Walking Onset

This is an intellectual property of the Infant Neuromotor Control Laboratory at
Children's Hospital Los Angeles.
If you have any questions, please contact Jinseok Oh (joh@chla.usc.edu)
"""
from pathlib import Path

from incwear.core import axivity
import incwear.computation.movement_metrics as mm
import incwear.computation.entropy as ent
from incwear.utils.plot_segment import plot_segment

# Assumes four .cwa files to be located
# 1. Two (2) recording files (left leg & right leg movements)
# 2. Two (2) calibration files (~1 min recording of 1g along axes)
directory = Path('/Users/joh/Downloads/OneDrive_1_7-21-2025')
left_recording = directory / 'leftlegmovement.cwa'
left_calibration = directory / 'leftcalibration.cwa'
right_recording = directory / 'rightlegmovement.cwa'
right_calibration = directory / 'rightcalibration.cwa'

# You can do it by leg
# Ax6 class of axivity module now takes two arguments in the following order:
#   1. Name of the sensor calibration file
#   2. Name of the sensor movement recording file
leftleg = axivity.Ax6(str(left_calibration),
                      str(left_recording),
                      )
# You can make the right leg object the same way.
rightleg = axivity.Ax6(str(right_calibration),
                       str(right_recording),
                       )

# You can also make a subject containing two leg objects.
# Use Ax6Subject class of axivity module.
# Arguments should be:
#   1. Name of the left leg sensor calibration file
#   2. Name of the left leg sensor movement recording file
#   3. Name of the right leg sensor calibration file
#   4. Name of the right leg sensor movement recording file
subj = axivity.Ax6Subject(
        str(left_calibration),
        str(left_recording),
        str(right_calibration),
        str(right_recording),
        )

"""
When you want to trim the data before processing, please provide 'trim_window'.
A trim_window is a tuple of two elements: trim_start, trim_end
A trim_start can be 'None' (if you don't trim at the beginning)
or the duration in seconds you want to cut (ex. 300)
A trim_end is similar in that it can be 'None' if you don't trim at the end,
or the duration in seconds. However, this time it should be the duration
from the start of the full recording.
For example, if you want to use the data with the first hour removed,
your command will be:
subj = axivity.Ax6Subject(
        str(left_calibration),
        ...
        str(right_recording),
        trim_window=(3600, None),    << 3600 seconds = 1 hour
        )
If you want to use the data with the last hour removed,
your command will be:
subj = axivity.Ax6Subject(
        str(left_calibration),
        ...
        str(right_recording),
        trim_window=(None, {full_recording_length_in_seconds - 3600}),
        )
So if your recording was 48 hours, the number inside {} will be:
    48 * 3600 - 3600 = 169200

If you provide trim_window to an Ax6Subject object,
the same trim window will be applied to both leg data.
If you want to apply separate windows, create leg-specific objects
and provide distinct trim_window's.
"""

# From a 'leg' instance, you get the movements in the following way:
lmovs = leftleg.detect_movements()

# From a 'subject', this way:
lmovs_subj = subj.left.detect_movements()
rmovs_subj = subj.right.detect_movements()

# `cycle_filt`
lmovs_filt = mm.cycle_filt(lmovs)

"""
Plotting can be done, but you may want to be more specific.
`plot_segment` function takes:
  - One Ax6 class instance (a.k.a. each leg)
  - Time passed from the (trimmed) start of the recording in seconds
  - Duration to plot
  - [Optional] movement matrix
"""
# So if you draw with an Ax6 class instance (ex. leftleg):
plot_segment(leftleg,
             time_passed=200,
             duration=2000,
             movmat = lmovs_filt,
             )

# If you draw with an Ax6Subject class instance (ex. subj):
plot_segment(subj.left,
             time_passed=0,
             duration=60,
             movmat=lmovs_subj,
             )

# If you draw right leg movements...
plot_segment(subj.right,
             time_passed=60,
             duration=20,
             side='R',
             movmat=rmovs_subj,
             )

# `rate_calc` function is identical in shape
record_len = list(subj.left.info.recordlen,
                  subj.right.info.recordlen)

mov_summary = mm.rate_calc(
        lmovs_subj,
        rmovs_subj,
        record_len,
        subj.left.info.fs)

# `acc_per_mov` works differently
# Provide acceleration magnitude vector, and the (filtered) movement matrix.
mm.acc_per_mov(leftleg.measures.accmags, lmovs_filt)
