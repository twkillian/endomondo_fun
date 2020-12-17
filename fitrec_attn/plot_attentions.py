import pickle
import numpy as np
import matplotlib.pyplot as plt

f = open('../evaluated_data.pkl', 'rb')
data = pickle.load(f)
f.close()

# Get all attentions as a 2D numpy array
all_attentions = []

for key, val in data.items():
    all_attentions.append(val['attention'])
    
cleaned = [x for x in all_attentions if len(x) == 280]

arr = np.array(cleaned)

# Get averages
means = np.mean(arr, axis=0) 
stds = np.std(arr, axis=0)

mean = np.mean(means, axis=0)
std = np.mean(stds, axis=0)

mean = np.flip(mean)
std = np.flip(std)

# Plot
plt.rc('axes', labelsize=12)

plt.xlabel('Time steps prior to t')
plt.ylabel('Attention')
plt.grid()

plt.plot(range(1,len(mean)+1), mean, lw=2)
plt.fill_between(range(1,len(mean)+1), mean-std, mean+std, alpha=0.3)

plt.yscale('log')
