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

# You can do it by leg.
# Ax6 takes the recording file first; the calibration file is optional:
#   axivity.Ax6(recording, calibration=None, **kwargs)
leftleg = axivity.Ax6(str(left_recording),
                      str(left_calibration),
                      )
# You can make the right leg object the same way.
rightleg = axivity.Ax6(str(right_recording),
                       str(right_calibration),
                       )

# You can also make a subject containing two leg objects.
# Use Ax6Subject class of axivity module.
# Arguments are KEYWORD-ONLY and recording-first (matching Ax6);
# calibration files are optional:
#   Ax6Subject(recording_left=..., recording_right=...,
#              calibration_left=..., calibration_right=...)
subj = axivity.Ax6Subject(
        recording_left=str(left_recording),
        calibration_left=str(left_calibration),
        recording_right=str(right_recording),
        calibration_right=str(right_calibration),
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
        recording_left=str(left_recording),
        ...
        calibration_right=str(right_calibration),
        trim_window=(3600, None),    # 3600 seconds = 1 hour
        )
If you want to use the data with the last hour removed,
your command will be:
subj = axivity.Ax6Subject(
        recording_left=str(left_recording),
        ...
        calibration_right=str(right_calibration),
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
record_len = [subj.left.info.recordlen,
              subj.right.info.recordlen]

mov_summary = mm.rate_calc(
        lmovs_subj,
        rmovs_subj,
        record_len,
        subj.left.info.fs)

# `acc_per_mov` works differently
# Provide acceleration magnitude vector, and the (filtered) movement matrix.
mm.acc_per_mov(leftleg.measures.accmags, lmovs_filt)


"""
============================================================================
Sensor-swap recordings
============================================================================
Sometimes the sensors are placed on the wrong sides for part of the
collection (e.g. Sensor A is on the LEFT leg for the first 24 hours, then the
sensors are swapped so Sensor A ends up on the RIGHT leg for the final 24
hours). After splicing, each leg's data was collected by TWO physical
sensors, so a single calibration file no longer applies to the whole file.

Use `Ax6SwapSubject`. Provide each leg as an ORDERED (chronological) list of
segments; each segment is
    (recording_file, calibration_file, trim_window)
and is calibrated with its OWN sensor's calibration. Internally the package:
  1. calibrates each segment with its own sensor calibration,
  2. computes ONE acceleration threshold per leg over the FULL duration
     (so the threshold is not derived from a short sub-recording),
  3. detects movements within each segment and merges them (no movement is
     ever counted across the swap boundary).

`trim_window` follows the same convention as Ax6. Keep adjacent windows from
overlapping: because the trim end is inclusive, use e.g. (None, 86400) then
(86400.04, None) rather than (86400, None), so the boundary sample is not
duplicated.
"""
# Example: Sensor A left-then-right, Sensor B right-then-left, swap at 24h.
swap_subj = axivity.Ax6SwapSubject(
    left=[(str(left_recording),  str(left_calibration),  (None, 86400)),
          (str(left_recording),  str(right_calibration), (86400.04, None))],
    right=[(str(right_recording), str(right_calibration), (None, 86400)),
           (str(right_recording), str(left_calibration),  (86400.04, None))],
)

# The merged legs are DROP-IN leg objects: every function above works on them
# unchanged (detect_movements, cycle_filt, rate_calc, acc_per_mov,
# plot_segment).
swap_lmovs = swap_subj.left.detect_movements()
swap_rmovs = swap_subj.right.detect_movements()
swap_rates = mm.rate_calc(
    swap_lmovs,
    swap_rmovs,
    [swap_subj.left.info.recordlen, swap_subj.right.info.recordlen],
    swap_subj.left.info.fs,
)

# Data-quality check: compare each segment's own threshold to the global one
# actually used. A large gap flags sensors that calibration did not reconcile.
print("left global accth:", swap_subj.left.measures.thresholds.accth)
print("left per-segment accth:",
      [t.accth for t in swap_subj.left.segment_thresholds])
print("swap points (samples):", swap_subj.left.splice_indices)
