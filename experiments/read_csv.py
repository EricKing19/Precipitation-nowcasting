import numpy as np
import pandas as pd

csv_file = pd.read_csv("MultiModal_1e-4/result/19.csv")
acc = csv_file.loc[6, ['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4']].values
hist = csv_file.loc[0:4, ['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4']].values
# acc = csv_file.loc[3, ['pred_0', 'pred_1']].values
# hist = csv_file.loc[0:1, ['pred_0', 'pred_1']].values
# recall = csv_file[['recall']].values[:2]
iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
# f1 = [2*(acc[i]*recall[i][0])/(acc[i]+recall[i][0]) for i in range(len(recall))]
mean_iu = np.nanmean(iu)
mean_acc = np.diag(hist).sum()/hist.sum()

print('iou:')
print(iu)
print('mean_iou:')
print(mean_iu)

# print('f1:')
# print(f1)
# print('f1:')
# print(sum(f1)/2)

# print('acc:')
# print(acc)
# print('mean_acc:')
# print(mean_acc)