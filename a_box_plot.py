import matplotlib.pyplot as plt
import numpy as np
visual_rmssd = [68.75, 47.29, 42.21,35.01,76.662, 66.579] 
text_rmssd = [74.60,40.52, 31.71,35.19,67.742, 52.765] 


data = [visual_rmssd, text_rmssd]
labels = ['Visual Task', 'Textual Task']

plt.boxplot(data, tick_labels=labels)
plt.ylabel('RMSSD (ms)')
plt.title('RMSSD Distribution across Tasks')



# Add a scatter plot for each task
for i in range(len(data)):
    x=np.full(len(data[i]),i+1)
    y = data[i] 
    plt.scatter(x, y, color='red', label='Data points' if i == 0 else "") 


plt.legend(['Data Points'])
plt.show()
