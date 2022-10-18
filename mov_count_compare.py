import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

# Compare the two outputs
orig = pd.read_excel('Processed_JO.xlsx')
test = pd.read_excel('Processed_JO_V2.xlsx')

# What should I compare?
# How much movement per wake-hour decreased by the method I introduce.
movph_l = 'Movements per hour awake left leg'
movph_r = 'Movements per hour awake right leg'
movph_t = 'Average movements per hour awake'

test_sub = test[['Infant', 'visit', movph_l, movph_r, movph_t]]

#full = pd.concat([orig_sub, test_sub])
full = pd.merge(orig, test_sub, on=['Infant', 'visit'])

fig, axes = plt.subplots(1, 3, figsize = (15,5))
for _, ax in enumerate(axes):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
sns.scatterplot(ax = axes[0], data = full, x=movph_l+'_x', y = movph_l+'_y')
axes[0].plot(full[movph_l+'_x'], full[movph_l+'_x'], 'r')
sns.scatterplot(ax = axes[1], data = full, x=movph_r+'_x', y = movph_r+'_y')
sns.scatterplot(ax = axes[2], data = full, x=movph_t+'_x', y = movph_t+'_y')
plt.tight_layout()
plt.show()

test.columns
# If you want to make a figure based on 
test['Infant'] = list(map(str, test['Infant']))
fig, ax = plt.subplots(1,1, figsize = (16,6))
sns.lineplot(ax = ax, data = test, x='age (days)', y='Average movements per hour awake',\
        style='adjust by 1 hour?', markers = True, hue='Infant')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.get_legend().remove()
ax.set_xlabel('Age (in days)', fontweight='bold', fontsize=12)
ax.set_ylabel('Average Movements per Hour Awake, both legs', fontweight='bold', fontsize=12)
ax.xlim = [0, 200]
ax.ylim = [0, 2000]
ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
ax.set_xticklabels([0, 25, 50, 75, 100, 125, 150, 175, 200], fontweight='bold')
ax.set_yticklabels(labels = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000], fontweight='bold')
plt.savefig('avg_mov_wake_hrs.tiff')
plt.show()


import os
print(os.path.abspath(os.curdir))
