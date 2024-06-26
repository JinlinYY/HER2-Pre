import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
N = 37
df = pd.read_excel('data-CDK3-encoding-PCC.xlsx')
x = df.iloc[:, 1:N].values
y = df.iloc[:, 36].values
y = y.reshape(-1, 1)


df.fillna(df.mean(), inplace=True)

corr_matrix = np.corrcoef(x, y, rowvar=False)
feature_names = df.columns[1:N]
correlations = pd.Series(corr_matrix[-1, :-1], index=feature_names)



plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm')

plt.title('Correlation Matrix Heatmap')
plt.xticks(range(len(df.columns[1:N])), df.columns[1:N])
plt.yticks(range(len(df.columns[1:N])), df.columns[1:N])
plt.xticks(rotation=45, ha="right")

plt.yticks(rotation=0, ha="right")
plt.tight_layout()
plt.savefig('correlation_matrix_heatmap.png')
plt.show()




import matplotlib.pyplot as plt
import numpy as np



plt.figure(figsize=(10, 8))


values = list(reversed(correlations.values))

indices = list(reversed(correlations.index))
for i in range(36):
    print(indices[i],"\t",values[i])


values = np.where(np.isnan(values), 0, values)
values = np.where(np.isinf(values), 0, values)
if values.ptp() == 0:
    values = values - values.min() + 0.001


cmap = plt.cm.coolwarm
colors = cmap((values - values.min()) / (values.max() - values.min()))


bars = plt.barh(indices, values, color=colors)


plt.title('Feature Correlation with Target y')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')


cax = plt.axes([0.92, 0.1, 0.02, 0.8])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.75, 1))
sm.set_array([])
plt.colorbar(sm, cax=cax)


plt.savefig('feature_correlation_with_target_y.png')


plt.show()
