import json
import matplotlib.pyplot as plt
import seaborn as sns

score_path = "/command/layer_score.json"
max_score = 0
min_score = 10
score_layer = {
    "9": {"p_score": [], "n_score": [], "pn_score": []},
    "10": {"p_score": [], "n_score": [], "pn_score": []},
    "11": {"p_score": [], "n_score": [], "pn_score": []},
}
with open(score_path, 'r') as f:
    data = json.load(f)
    for k, v in data.items():
        for i in ['9', '10', '11']:
            for j in ["p_score", "n_score", "pn_score"]:
                max_score = max(v[i][j], max_score)
                min_score = min(v[i][j], min_score)
    data_min_max = {}
    for k, v in data.items():
        for i in ['9', '10', '11']:
            for j in ["p_score", "n_score", "pn_score"]:
                v[i][j] = (v[i][j] - min_score) / (max_score - min_score)
                if(v[i][j] > 1):
                    print(i, j)
                score_layer[i][j].append(v[i][j])

#print(score_layer)
    

fig, ax = plt.subplots(figsize=(12, 9))
#plt.rcParams["font.sans-serif"]=["WenQuanYi Micro Hei"] #设置字体
#plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

for i in ['9', '10', '11']:
    #for j in ["p_score", "n_score", "pn_score"]:
    for j in ["n_score"]:
        ax.hist(score_layer[i][j], bins=16, alpha = 0.7, density=True, label="{}_{}".format(i, j))
for i in ['9', '10', '11']:
    #for j in ["p_score", "n_score", "pn_score"]:
    for j in ["n_score"]:
        sns.kdeplot(score_layer[i][j], ax=ax)

ax.legend()
plt.show()
plt.savefig('./score_distribution_n.png')
