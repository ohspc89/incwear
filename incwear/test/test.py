import site
site.addsitedir('/Users/joh/Downloads/incwear/core')
site.addsitedir('/Users/joh/Downloads/incwear/computation')
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import axivity
import movement_metrics as mm

# Passes cwa test...
directory = Path('/Users/joh/Downloads/OneDrive_1_7-21-2025')
cwas = list(directory.glob('*.cwa'))
left_cwas = [x for x in cwas if '_L' in x.name]
right_cwas = [y for y in cwas if '_R' in y.name]

# You can do it by leg
leftleg = axivity.Ax6(*[str(x) for x in left_cwas])
rightleg = axivity.Ax6(*[str(x) for x in right_cwas])

# Or make a subject
subj1 = axivity.Ax6Subject(
        *[str(x) for x in left_cwas + right_cwas])


# TSV version?
directory_tsv = Path('/Users/joh/Downloads/sample_hbcd/sub-21135/ses-V02/motion')
motion_tsvs = list(directory_tsv.glob('*_motion.tsv'))
left_tsvs = [l for l in motion_tsvs if 'LeftLegMovement' in l.name]
right_tsvs = [r for r in motion_tsvs if 'RightLegMovement' in r.name]

tsv_leftleg = axivity.Ax6(*[str(x) for x in left_tsvs])

# Subject version
tsv_subj = axivity.Ax6Subject(
        *[str(x) for x in left_tsvs + right_tsvs])

lmovs = tsv_subj.left.detect_movements()
rmovs = tsv_subj.right.detect_movements()

# Cycle filt
lmovs_filt = mm.cycle_filt(lmovs)
