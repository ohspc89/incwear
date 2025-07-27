# ⚙️ incwear: Wearable Sensor Data Preprocessing for Infant Motor Analysis

`incwear` is a Python package developed to preprocess and analyze **wearable sensor data** in infant motor behavior research. It adapts and extends the original MATLAB algorithm developed by the **Infant Neuromotor Control Laboratory (INC Lab)** at Children's Hospital Los Angeles.

📄 Reference:  
Smith et al. (2015), *Daily Quantity of Infant Leg Movement: Wearable Sensor Algorithm and Relationship to Walking Onset*  
[DOI: 10.3390/s150819006](https://doi.org/10.3390/s150819006)

---

## 📦 What This Package Does

- Loads raw IMU data (e.g., accelerometry from ankle-worn devices)
- Applies validated preprocessing steps (e.g., filtering, thresholding)
- Detects and counts bouts of leg movement
- Outputs summary statistics for infant motor activity

---

## 🧪 Origin

The original algorithm was developed in **MATLAB**.  
This Python version was adapted and modularized for broader accessibility, improved maintainability, and integration with future tools.

**`incwear = INC (Lab) + WEARable`**

---

## 🧰 How to Use

```bash
git clone https://github.com/ohspc89/incwear.git
cd incwear
pip install -e .
```

```python
from incwear.core import axivity
import incwear.computation.movement_metrics as mm
from incwear.utils.plot_segment import plot_segment

# Example usage
calibration_filename = '~/Downloads/calibration.cwa'
recording_filename = '~/Downloads/recording.cwa'
leg = axivity.Ax6(calibration_filename, recording_filename)

## Identify movements using the algorithm
leg_movs = leg.detect_movements()

## Plot a segment (60 seconds) to visually identify movements
plot_segment(leg, time_passed=0, duration=60, movmat=leg_movs)
```

> 📁 Refer to the `examples/` folder for sample scripts and test data.

---

## 🧠 Applications

- Longitudinal sensor-based developmental monitoring
- Clinical and experimental wearable studies

---

## 📄 License & Credits

- © Infant Neuromotor Control Laboratory, 2022–2025
- Adapted by Jinseok Oh, Postdoctoral Fellow @ CHLA
- Open-source license: *(pending — recommend MIT or CC BY-NC-SA)*
