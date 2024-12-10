import pandas as pd
import numpy as np
df = pd.read_csv("D:\\DTU\\Courses\\02455 Experiment in cognitive science\\data\\Formal data\\1_IBI.csv", skiprows=2, sep=";")

#DataFrame was converted to a Numpy array
raw_data=df.values

data=raw_data[:,1]
print(np.isnan(data).sum())

lower_bound=600
upper_bound=1200
outliers=(data<lower_bound)|(data>upper_bound)
skiprows=3
original_indices = np.where(outliers)[0] + skiprows + 1 
print(f"outlier index:{original_indices}")
print(f"outlier:{data[outliers]}") 

from scipy.interpolate import interp1d
normal_indices=np.where(~outliers)[0]
normal_values=data[normal_indices]
linear_function=interp1d(normal_indices,normal_values, kind='linear',fill_value="extrapolate")
data[outliers]=linear_function(np.where(outliers)[0])
print(f"data_after_interp1d:{data}")

#IBI data during different task
# def extract_task_data(start_time, end_time, data, ibi_start_time):
#     target_indices=[]
#     current_time=ibi_start_time
#     for i, ibi in enumerate(data):
#         current_time+=ibi/1000
#         if start_time<=current_time<=end_time:
#             target_indices.append(i)
#     return data[target_indices]

def extract_task_data_indices(start_time, end_time, data, ibi_start_time):
    target_indices=[]
    current_time=ibi_start_time
    for i, ibi in enumerate(data):
        current_time+=ibi/1000
        if start_time<=current_time<=end_time:
            target_indices.append(i)
    return target_indices

# range of indices
visual_task_indices=extract_task_data_indices(start_time=31*60, end_time=34*60,data=data,ibi_start_time=25*60+9)
text_task_indices=extract_task_data_indices(start_time=40*60, end_time=43*60,data=data,ibi_start_time=25*60+9)

# print indices
visual_task_start_idx, visual_task_end_idx = visual_task_indices[0], visual_task_indices[-1]
text_task_start_idx, text_task_end_idx = text_task_indices[0], text_task_indices[-1]
print(f"Visual task start index: {visual_task_start_idx}, end index: {visual_task_end_idx}")
print(f"Text task start index: {text_task_start_idx}, end index: {text_task_end_idx}")

# task data
visual_task_data=data[visual_task_indices]
text_task_data=data[text_task_indices]


#RMSSD
def calculate_rmssd(data):
    diffs=np.diff(data)
    rmssd=np.sqrt(np.mean(diffs**2))
    return rmssd

visual_rmssd=calculate_rmssd(visual_task_data)
text_rmssd=calculate_rmssd(text_task_data)

print(f"RMSSD of visual task: {visual_rmssd}")
print(f"RMSSD of text task: {text_rmssd}")

#plot
#Box plot
import matplotlib.pyplot as plt
rmssd_data=[[visual_rmssd],[text_rmssd]]
task_labels=['visual_task','text_based_task']

plt.boxplot(rmssd_data, tick_labels=task_labels)
plt.ylabel("RMSSD(ms)")
plt.title("Comparison of RMSSD under different task conditions")
plt.show()

#trend chart
plt.plot(range(len(data)),data,label="IBI data")
plt.axvspan(visual_task_start_idx, visual_task_end_idx, color='blue', alpha=0.2, label="visual task")
plt.axvspan(text_task_start_idx, text_task_end_idx, color='orange', alpha=0.2, label="text-based task")
plt.xlabel("indices of bound")
plt.ylabel("IBI (ms)")
plt.title("IBI Trend data during tasks")
plt.legend()
plt.show()

from scipy.stats import ttest_rel, wilcoxon, shapiro
_, p_visual = shapiro(visual_task_data)
_, p_text = shapiro(text_task_data)

print(p_visual)
print(p_text)

print(f"Length of visual_task_data: {len(visual_task_data)}")
print(f"Length of text_task_data: {len(text_task_data)}")

min_length = min(len(visual_task_data), len(text_task_data))
visual_task_data = visual_task_data[:min_length]
text_task_data = text_task_data[:min_length]

if p_visual > 0.05 and p_text > 0.05:
    t_stat, p_value = ttest_rel(visual_task_data, text_task_data)
else:
    t_stat, p_value = wilcoxon(visual_task_data, text_task_data)

print(f"Statistical test results: t={t_stat}, p={p_value}")
