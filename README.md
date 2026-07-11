# ⚙️ incwear: Wearable Sensor Data Preprocessing for Infant Motor Analysis

`incwear` is a Python package developed to preprocess and analyze **wearable sensor data** in infant motor behavior research. It is a translation and extension of the original MATLAB algorithm developed by the **Infant Neuromotor Control Laboratory (INC Lab)** at Children's Hospital Los Angeles.

📄 Reference:  
Smith et al. (2015), *Daily Quantity of Infant Leg Movement: Wearable Sensor Algorithm and Relationship to Walking Onset*  
[DOI: 10.3390/s150819006](https://doi.org/10.3390/s150819006)

---

## 📦 What This Package Does

- Loads raw IMU data from Axivity Ax6 sensors (`.cwa` or `.tsv`)
- Applies per-sensor calibration (offset/gain) and validated preprocessing
- Detects and counts bouts of infant leg movement
- Handles recordings where the sensors were swapped mid-collection
- Outputs summary statistics (movement rate, per-movement acceleration, rest)

---

## 🧪 Origin

The original algorithm was developed in **MATLAB** by the INC Lab. This Python version was translated and modularized for broader accessibility, improved maintainability, and integration with future tools.

**`incwear = INC (Lab) + WEARable`**

---

## 🧰 How to Use

```bash
git clone https://github.com/ohspc89/incwear.git
cd incwear
pip install -e .
```

### A single leg

`Ax6` takes the **recording file first**; the calibration file is optional:

```python
from incwear.core import axivity
import incwear.computation.movement_metrics as mm
from incwear.utils.plot_segment import plot_segment

recording = '~/Downloads/recording.cwa'
calibration = '~/Downloads/calibration.cwa'   # optional

leg = axivity.Ax6(recording, calibration)

# Identify movements
leg_movs = leg.detect_movements()

# Plot a 60-second segment to inspect the detections
plot_segment(leg, time_passed=0, duration=60, movmat=leg_movs)
```

### Both legs (a subject)

Arguments are **keyword-only** and **recording-first**, matching `Ax6`:

```python
subj = axivity.Ax6Subject(
    recording_left='~/Downloads/left.cwa',
    calibration_left='~/Downloads/left_cal.cwa',
    recording_right='~/Downloads/right.cwa',
    calibration_right='~/Downloads/right_cal.cwa',
)
lmovs = subj.left.detect_movements()
rmovs = subj.right.detect_movements()
```

### Recordings where the sensors were swapped

If the sensors were on the wrong sides for part of the collection, each leg's
data was collected by **two different physical sensors**, so a single
calibration file does not apply to the whole file. Use `Ax6SwapSubject` and
give each leg an ordered (chronological) list of segments, one per sensor
placement: `(recording, calibration, trim_window)`.

Each segment is calibrated with its **own** sensor's calibration; one
acceleration threshold is then computed per leg over the **full** duration, and
movements are detected within each segment and merged (no movement is counted
across the swap boundary).

```python
# Sensor A left-then-right, Sensor B right-then-left, swapped at 24 h (86400 s).
# Note: keep adjacent windows from overlapping — the trim end is inclusive,
# so use (None, 86400) then (86400.04, None).
subj = axivity.Ax6SwapSubject(
    left=[('~/Downloads/left.cwa',  '~/Downloads/calA.cwa', (None, 86400)),
          ('~/Downloads/left.cwa',  '~/Downloads/calB.cwa', (86400.04, None))],
    right=[('~/Downloads/right.cwa', '~/Downloads/calB.cwa', (None, 86400)),
           ('~/Downloads/right.cwa', '~/Downloads/calA.cwa', (86400.04, None))],
)

# The merged legs are drop-in leg objects — every function works unchanged.
lmovs = subj.left.detect_movements()
rates = mm.rate_calc(
    subj.left.detect_movements(),
    subj.right.detect_movements(),
    [subj.left.info.recordlen, subj.right.info.recordlen],
    subj.left.info.fs,
)

# Data-quality check: each segment's own threshold vs. the global one used.
print(subj.left.measures.thresholds.accth, [t.accth for t in subj.left.segment_thresholds])
```

> 📁 See `incwear/test/test.py` for a fuller, runnable walkthrough of every step (trimming, plotting, rates, per-movement acceleration, and the swap case).

---

## 🧠 Applications

- Longitudinal sensor-based developmental monitoring
- Clinical and experimental wearable studies

---

## 📄 License & Credits

**This software is not open-source.** The algorithm it implements is the
**intellectual property of the Infant Neuromotor Control Laboratory (INC Lab)
and its collaborators**. This Python package is a translation and extension of
the laboratory's original MATLAB implementation; it is provided for research
use by and with the permission of the INC Lab. **All rights reserved.**

- © 2026 Infant Neuromotor Control Laboratory, Children's Hospital Los Angeles.
- Original algorithm: Smith et al. (2015) and INC Lab collaborators.
- Python translation & extension: Jinseok Oh, Ph.D. (Postdoctoral Fellow, CHLA).

For licensing, reuse, redistribution, or collaboration inquiries, please
contact the INC Lab.

---

## Third-party components

This project uses the following third-party components. Each is distributed
under a permissive license compatible with this package's proprietary
distribution:

- **scikit-digital-health (SKDH)**: `skdh.io.ReadCwa` is used to read Axivity `.cwa` files in `incwear.utils.io_utils`.
  Licensed under the MIT License.

- **Open Movement**: `incwear.core.cwa_metadata` contains code adapted from the Open Movement CWA metadata reader by Dan Jackson / Newcastle University.
  Distributed under the original Open Movement (BSD-style) license terms.
