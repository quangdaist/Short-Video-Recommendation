import matplotlib.pyplot as plt
import numpy as np

labels = ['OMoE', 'MMoE', 'MMoE_A', 'MMoE_AT', 'PLE', 'MetaHeac', 'AITM']
men_means = [0.20, 0.34, 0.30, 0.35, 0.27, 0.2, 0.9]
women_means = [0.25, 0.32, 0.34, 0.20, 0.25, 0.2, 0.9]
gay_means = [0.25, 0.32, 0.34, 0.20, 0.25, 0.2, 0.9]
hi_means = [0.25, 0.32, 0.34, 0.20, 0.25, 0.2, 0.9]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, men_means, width, label='Men')
rects2 = ax.bar(x, women_means, width, label='Women')
rects3 = ax.bar(x + width, women_means, width, label='gay')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3, fontsize=7)
ax.bar_label(rects2, padding=3, fontsize=7)
ax.bar_label(rects3, padding=3, fontsize=7)

fig.tight_layout()
plt.show()

##
import matplotlib.pyplot as plt
import numpy as np

labels = ['OMoE', 'MMoE', 'MMoE_A', 'MMoE_AT', 'PLE', 'MetaHeac', 'AITM']
men_means = [0.20, 0.34, 0.30, 0.35, 0.27, 0.2, 0.9]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, men_means, width, label='Men')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3, fontsize=7)
# fig.tight_layout()
plt.show()
