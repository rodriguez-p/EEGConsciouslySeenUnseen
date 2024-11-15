from pymer4.models import Lmer
from pymer4.stats import lrt
import pandas as pd
import os
import janitor
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import ptitprince as pt

os.chdir("../behav_data")

pd.set_option('display.float_format', '{:0.3f}'.format)

FIGURES_DIR = '../figures'

#%%
data = pd.read_excel('raw_data.xlsx')

awareness_mapping = {1: 'Seen', 0: 'Unseen'}

# Use map function to apply the mapping and create a new column 'Awareness'
data['Awareness'] = data['Seen'].map(awareness_mapping)

data.dropna(subset=data.columns[163], inplace=True) 

#%% ACC
data_acc = data
data_acc = data_acc[data_acc['TargetPresence'] == 'Present']
data_acc = data_acc[data_acc['BlockType'] == 'Experimental']

data_acc = data_acc.drop(data_acc[data_acc['Subject'] == 24].index)
data_acc = data_acc.drop(data_acc[data_acc['Subject'] == 26].index)
data_acc = data_acc.drop(data_acc[data_acc['Subject'] == 38].index)

model_acc = Lmer(formula="RespTilt.ACC ~ Awareness + (1 | Subject)", data=data)
model_acc.fit(factors={"Awareness": ['Seen', 'Unseen']},
    ordered=True,
    summarize=True,
    family = 'binomial'
)
    
anova_results_acc = model_acc.anova()
print(anova_results_acc)

marginal_estimates_acc, comparisons_acc = model_acc.post_hoc(marginal_vars = 'Awareness', p_adjust="bonf")

print(comparisons_acc)

#%% Fig. 2A: Tilt ACC for Seen/Unseen in Experimental blocks

mean_acc = data_acc.groupby(['Awareness', 'Subject'])['RespTilt.ACC'].mean().reset_index()

f, ax = plt.subplots(1, 1, figsize=(16, 8), sharey = 'all', tight_layout = True)   

pt.RainCloud(x = 'Awareness', y = 'RespTilt.ACC', data = mean_acc,
            palette = 'Set1', order = ['Seen', 'Unseen'], box_showfliers = False, ax = ax, bw = 0.1)

ax.set_xlabel('', fontsize = 24, fontweight = 'regular', labelpad = 20.0)
xlabel_1 = ax.xaxis.label
xlabel_1.set_x(xlabel_1.get_position()[0] + 0.07)
ax.set_ylabel('Accuracy', fontsize = 24, fontweight = 'regular', labelpad = 20.0)
ax.tick_params(axis='both', which='major', labelsize=24)

pairs=[("Seen", "Unseen")]
annotator = Annotator(ax = ax, pairs = pairs, data=mean_acc, x='Awareness', y='RespTilt.ACC')
annotator.set_custom_annotations(["*"]) 
annotator.configure(text_format='star', loc='outside', line_height = 0)
annotator.annotate()

sns.despine(trim=True)

plt.show()
plt.savefig(f'{FIGURES_DIR}/Fig. 2A.svg', format = 'svg', dpi = 1200)
plt.close()

#%% RTs
df = data[data['RespTilt.RT'] > 150]

mean = df['RespTilt.RT'].mean()
std = df['RespTilt.RT'].std()
lower = mean - 2.5 * std
upper = mean + 2.5 * std

# Filtrar nuevamente el DataFrame según el criterio de +/- 2.5 SD
df_rt = df[(df['RespTilt.RT'] >= lower) & (df['RespTilt.RT'] <= upper)]

model_rt = Lmer(formula="RespTilt.RT ~ Awareness + (1 | Subject)", data=df_rt)
model_rt.fit(factors={"Awareness": ['Seen', 'Unseen']},
    ordered=True,
    summarize=True,
    family = 'inverse_gaussian'
)
    
anova_results_rt = model_rt.anova()
print(anova_results_rt)

marginal_estimates_rt, comparisons_rt = model_rt.post_hoc(marginal_vars = 'Awareness', p_adjust="bonf")

print(comparisons_rt)

#%%
model_seen = Lmer(formula="Seen ~ TargetPresence*BlockType + (1 | Subject)", data=data)
model_seen.fit(factors={"TargetPresence": ['Present', 'Absent'],
                        "BlockType": ['Localizer', 'Experimental']},
    ordered=True,
    summarize=True,
    family = 'binomial'
)

anova_results_seen = model_seen.anova()
print(anova_results_seen)

marginal_estimates_seen, comparisons_seen = model_seen.post_hoc(marginal_vars = ['BlockType', 'TargetPresence'], p_adjust="bonf")

print(comparisons_seen)

#%% Fig. 2B: % Seen for each block (Localizer, Experimental) for Target present/absent

mean_seen = data.groupby(['BlockType', 'TargetPresence', 'Subject'])['Seen'].mean().reset_index()

mean_seen = mean_seen.drop(mean_seen[mean_seen['Subject'] == 24].index)
mean_seen = mean_seen.drop(mean_seen[mean_seen['Subject'] == 26].index)
mean_seen = mean_seen.drop(mean_seen[mean_seen['Subject'] == 41].index)

mean_seen_localizer = mean_seen[mean_seen['BlockType'] == 'Localizer']
mean_seen_experimental = mean_seen[mean_seen['BlockType'] == 'Experimental']

f, ax = plt.subplots(1, 2, figsize=(16, 8), sharey = 'all', tight_layout = True)    

pt.RainCloud(x = 'TargetPresence', y = 'Seen', data = mean_seen_localizer,
            palette="Set2", bw=.2, width_viol=.6, orient='v', ax=ax[0], box_showfliers = False)
ax[0].set_xlabel('Localizer blocks', fontsize = 24, fontweight = 'regular', labelpad = 24.0)
xlabel_0 = ax[0].xaxis.label
xlabel_0.set_x(xlabel_0.get_position()[0] + 0.06)
ax[0].set_ylabel('Proportion Seen', fontsize = 24, fontweight = 'regular', labelpad = 24.0)
ax[0].tick_params(axis='both', which='major', labelsize=24)

pt.RainCloud(x = 'TargetPresence', y = 'Seen', data = mean_seen_experimental,
            palette="Set2", bw=.2, width_viol=.6, orient='v', ax=ax[1], box_showfliers = False)
ax[1].set_xlabel('Experimental blocks', fontsize = 24, fontweight = 'regular', labelpad = 24.0)
xlabel_1 = ax[1].xaxis.label
xlabel_1.set_x(xlabel_1.get_position()[0] + 0.07)
ax[1].set_ylabel('')
ax[1].tick_params(axis='both', which='major', labelsize=24)

sns.despine(trim=True)
ax[1].yaxis.set_visible(False)
ax[1].spines['left'].set_visible(False)
plt.tight_layout()

plt.show()
plt.savefig(f'{FIGURES_DIR}/Fig. 2B.svg', format = 'svg', dpi = 1200)
plt.close()


#%% combine figures

import matplotlib.image as mpimg

os.chdir("C../figures")

img1 = mpimg.imread('Fig. 2B.png')
img2 = mpimg.imread('Fig. 2A.png')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0)

# Plot the images on subplots
axes[0].imshow(img1)
axes[0].axis('off')
# axes[0].set_title('Target presence', x = 0.46, fontsize = 16, color = 'navy', ha = 'center', va = 'center')
axes[0].text(0, 0.15, 'A', fontsize = 16, fontweight='bold')

axes[1].imshow(img2)
axes[1].axis('off')
# axes[1].set_title('Awareness', x = 0.47, fontsize = 16, color = 'red', ha = 'center', va = 'center')
axes[1].text(0, 0.15, 'B', fontsize = 16, fontweight='bold')

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGURES_DIR}/Fig. 2.svg', format = 'svg', dpi = 1200)
plt.close()




