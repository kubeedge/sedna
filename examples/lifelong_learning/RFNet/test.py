import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

CPA_results = np.load("./cpa_results.npy").T
ratios = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
ratio_counts = np.zeros((len(CPA_results), len(ratios)), dtype=float)

for i in range(len(CPA_results)):
    for j in range(len(ratios)):
        result = CPA_results[i]
        result = result[result <= ratios[j]]

        ratio_counts[i][j] = len(result) / 275

plt.figure(figsize=(45, 10))
ratio_counts = pd.DataFrame(data=ratio_counts.T, index=ratios)
sns.heatmap(data=ratio_counts, annot=True, cmap="YlGnBu", annot_kws={'fontsize': 15})
plt.xticks(fontsize=20)
plt.yticks(fontsize=25)
plt.xlabel("Test images", fontsize=25)
plt.ylabel("Ratio of PA ranges", fontsize=25)
plt.savefig("./figs/ratio_count.png")
plt.show()


# data = pd.DataFrame(CPA_results.T)
#
# plt.figure(figsize=(30, 15))
# cpa_result = pd.DataFrame(data=data)
# sns.heatmap(data=cpa_result)
# plt.savefig("./figs/heatmap_pa.png")
# plt.show()
#
# plt.figure(figsize=(30, 15))
# cpa_result = pd.DataFrame(data=data[:15])
# sns.heatmap(data=cpa_result)
# plt.savefig("./figs/door_heatmap_pa.png")
# plt.show()
#
# plt.figure(figsize=(30, 15))
# cpa_result = pd.DataFrame(data=data[15:31])
# sns.heatmap(data=cpa_result)
# plt.savefig("./figs/garden1_heatmap_pa.png")
# plt.show()
#
# plt.figure(figsize=(30, 15))
# cpa_result = pd.DataFrame(data=data[31:])
# sns.heatmap(data=cpa_result)
# plt.savefig("./figs/garden2_heatmap_pa.png")
# plt.show()
