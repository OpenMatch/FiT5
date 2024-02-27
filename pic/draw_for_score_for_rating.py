import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


score_path = "/command/layer_score_20.json"
max_score = 0
min_score = 10
score_layer = {
    "9": {},
    "10": {},
    "11": {},
}
with open(score_path, 'r') as f:
    data = json.load(f)
    for k, v in data.items():
        for i in ['9', '10', '11']:
            for j in ["0", "1", "2", "3"]:
                for k in ["0", "1", "2", "3"]:
                    l = "{}_{}_score".format(j, k)
                    if v[i][l] != "NaN":
                        max_score = max(v[i][l], max_score)
                        min_score = min(v[i][l], min_score)
    data_min_max = {}
    for k, v in data.items():
        for i in ['9', '10', '11']:
            for j in ["0", "1", "2", "3"]:
                for k in ["0", "1", "2", "3"]:
                    l = "{}_{}_score".format(j, k)
                    if v[i][l] != "NaN":
                        v[i][l] = (v[i][l] - min_score) / (max_score - min_score)
                        if l not in score_layer[i]:
                            score_layer[i][l] = []
                        score_layer[i][l].append(v[i][l])
                    

#print(score_layer)
    
plt.rc('text', usetex=True)
plt.style.use('default')
fig, ax = plt.subplots(figsize=(25, 25))
#plt.rcParams["font.sans-serif"]=["WenQuanYi Micro Hei"] #设置字体
#plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
#plt.rcParams["font.weight"] = "bold"
#for i in ['9', '10', '11']:
"""for i in ['11']:
    for j in ["3"]:
        for k in ["0", "1", "2", "3"]:
            l = "{}_{}_score".format(j, k)
            ax.hist(score_layer[i][l], bins=16, alpha = 0.7, density=True, label="{}_{}".format(i, l))"""
for i in ['11']:
   for j in ["3"]:
       for k in ["0", "1", "2", "3"]:
           l = "{}_{}_score".format(j, k)
           sns.kdeplot(score_layer[i][l], shade=True, ax=ax, label=r"{}$\rightarrow${}".format(j, k))
#for i in ['9', '10', '11']:
#    score_i = []
#    for j in ["3"]:
#        for k in ["0", "1", "2"]:
#            l = "{}_{}_score".format(j, k)
#            score_i = score_i + score_layer[i][l]
#    if i != '11':
#        sns.kdeplot(score_i, shade=True, ax=ax, label="Layer "+str(int(i)+1))
#    else:
#        sns.kdeplot(score_i, shade=True, ax=ax, label="Layer "+str(int(i)+1)+" (last layer)")
size=105
ax.legend(prop = {'size':80}, bbox_to_anchor=(0.395,1.0245), framealpha=0.3)

ax.set_xlim(xmin = -0.1, xmax = 1)
ax.set_ylim(ymin = 0, ymax = 3.0)
ax.set_xticks(np.arange(0, 1.01, 0.5))
ax.set_yticks(np.arange(0, 3.01, 1))
ax.set_xlabel('Attention Value',fontsize=size)
ax.set_ylabel('Probability Denisity Function',fontsize=size)
ax.spines['bottom'].set_linewidth(3);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(3);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(3);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(3)
ax.tick_params(labelsize=size, width=4, size=25)

plt.show()
#plt.savefig('./img/score_distribution_each_layer_20.pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('./img/score_distribution_20.pdf', bbox_inches='tight', pad_inches=0)
