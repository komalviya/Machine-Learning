import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
from sklearn.metrics import mean_squared_error

# Read in the data as inefficiently as possible :)
df = pd.read_csv("vibes.csv", comment='#')
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X3 = df.iloc[:, 2]
X4 = df.iloc[:, 3]
X5 = df.iloc[:, 4]
X6 = df.iloc[:, 5]
X7 = df.iloc[:, 6]
X8 = df.iloc[:, 7]
X9 = df.iloc[:, 8]
X10 = df.iloc[:, 9]
X11 = df.iloc[:, 10]
X12 = df.iloc[:, 11]
X13 = df.iloc[:, 12]
X14 = df.iloc[:, 13]
X15 = df.iloc[:, 14]
X16 = df.iloc[:, 15]
X17 = df.iloc[:, 16]
X18 = df.iloc[:, 17]
X19 = df.iloc[:, 18]
X20 = df.iloc[:, 19]
X21 = df.iloc[:, 20]
X22 = df.iloc[:, 21]
X23 = df.iloc[:, 22]
X24 = df.iloc[:, 23]
X25 = df.iloc[:, 24]
X26 = df.iloc[:, 25]
X27 = df.iloc[:, 26]
y = df.iloc[:, 27]
X = np.column_stack((X1, X2, X3, X4, X5, X6, X7, X8, X9, X9, X10,
                     X11, X12, X13, X14, X15, X16, X17, X18, X19, X20,
                     X21, X22, X23, X24, X25, X26, X27))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=73)

# Create model with k = 2 & "distance" weights

knn = KNeighborsClassifier(n_neighbors=1, weights='uniform').fit(X_train, y_train)
y_pred = knn.predict(X_test)

# macro_roc_auc_ovo = roc_auc_score(y_test, y_pred, multi_class="ovo",
#                                   average="macro")
# weighted_roc_auc_ovo = roc_auc_score(y_test, y_pred, multi_class="ovo",
#                                      average="weighted")
# macro_roc_auc_ovr = roc_auc_score(y_test, y_pred, multi_class="ovr",
#                                   average="macro")
# weighted_roc_auc_ovr = roc_auc_score(y_test, y_pred, multi_class="ovr",
#                                      average="weighted")
# print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#       "(weighted by prevalence)"
#       .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
# print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#       "(weighted by prevalence)"
#       .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

fpr = dict()
tpr = dict()
roc_auc = dict()
b_fpr = dict()
b_tpr = dict()
b_roc_auc = dict()

y_true = [[0 for i in range(len(y_test))] for j in range(9)]
y_score = [[0 for i in range(len(y_test))] for j in range(9)]
y_true = np.array(y_true)
y_score = np.array(y_score)
for i in range(9):
    n = 0
    for x in y_test:
        if x == (i + 1):
            y_true[i, n] = 1
        else:
            y_true[i, n] = 0
        n += 1
    n = 0
    for x in y_pred:
        if x == (i + 1):
            y_score[i, n] = 1
        else:
            y_score[i, n] = 0
        n += 1

y_baseline = np.empty(len(y_test), dtype=object)
for i in range(len(y_test)):
    y_baseline[i] = random.randint(1, 9)

for i in range(9):
    fpr[i], tpr[i], _ = roc_curve(y_true[i, :], y_score[i, :])
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(9):
    b_fpr[i], b_tpr[i], _ = roc_curve(y_test, y_baseline, pos_label=i+1)
    b_roc_auc[i] = auc(b_fpr[i], b_tpr[i])


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(9)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(9):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= 9
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

b_all_fpr = np.unique(np.concatenate([b_fpr[i] for i in range(9)]))
b_mean_tpr = np.zeros_like(b_all_fpr)
for i in range(9):
    b_mean_tpr += np.interp(b_all_fpr, b_fpr[i], b_tpr[i])

b_mean_tpr /= 9
b_fpr["macro"] = b_all_fpr
b_tpr["macro"] = b_mean_tpr
b_roc_auc["macro"] = auc(b_fpr["macro"], b_tpr["macro"])

# Plot all ROC curves
fig = plt.figure()

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (AUC = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

plt.plot(b_fpr["macro"], b_tpr["macro"],
         label='Baseline ROC curve (AUC = {0:0.2f})'
               ''.format(b_roc_auc["macro"]),
         color='k', linestyle='--', linewidth=3)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'magenta', 'green', 'black', 'red', 'violet', 'gray'])
for i, color in zip(range(9), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label='ROC curve of class {0} (AUC = {1:0.2f})'
             ''.format(i+1, roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
fig.set_size_inches(18, 10)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC for kNN model')
plt.legend(loc="lower right")
plt.show()

print("                                 ACTUAL VALUE")
print("               Celtic       Indian       South American       Japanese        Mexican           French          German           African         Australian")
print("Celtic           " + str(confusion_matrix(y_test, y_pred)[0, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[0, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[0, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[0, 3]) + "                " + str(confusion_matrix(y_test, y_pred)[0, 4]) + "                " + str(confusion_matrix(y_test, y_pred)[0, 5]) + "                " + str(confusion_matrix(y_test, y_pred)[0, 6]) + "                " + str(confusion_matrix(y_test, y_pred)[0, 7]) + "                " + str(confusion_matrix(y_test, y_pred)[0, 8]))
print("Indian           " + str(confusion_matrix(y_test, y_pred)[1, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[1, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[1, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[1, 3]) + "                " + str(confusion_matrix(y_test, y_pred)[1, 4]) + "                " + str(confusion_matrix(y_test, y_pred)[1, 5]) + "                " + str(confusion_matrix(y_test, y_pred)[1, 6]) + "                " + str(confusion_matrix(y_test, y_pred)[1, 7]) + "                " + str(confusion_matrix(y_test, y_pred)[1, 8]))
print("South American   " + str(confusion_matrix(y_test, y_pred)[2, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[2, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[2, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[2, 3]) + "                " + str(confusion_matrix(y_test, y_pred)[2, 4]) + "                " + str(confusion_matrix(y_test, y_pred)[2, 5]) + "                " + str(confusion_matrix(y_test, y_pred)[2, 6]) + "                " + str(confusion_matrix(y_test, y_pred)[2, 7]) + "                " + str(confusion_matrix(y_test, y_pred)[2, 8]))
print("Japanese         " + str(confusion_matrix(y_test, y_pred)[3, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[3, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[3, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[3, 3]) + "                " + str(confusion_matrix(y_test, y_pred)[3, 4]) + "                " + str(confusion_matrix(y_test, y_pred)[3, 5]) + "                " + str(confusion_matrix(y_test, y_pred)[3, 6]) + "                " + str(confusion_matrix(y_test, y_pred)[3, 7]) + "                " + str(confusion_matrix(y_test, y_pred)[3, 8]))
print("Mexican          " + str(confusion_matrix(y_test, y_pred)[4, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[4, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[4, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[4, 3]) + "                " + str(confusion_matrix(y_test, y_pred)[4, 4]) + "                " + str(confusion_matrix(y_test, y_pred)[4, 5]) + "                " + str(confusion_matrix(y_test, y_pred)[4, 6]) + "                " + str(confusion_matrix(y_test, y_pred)[4, 7]) + "                " + str(confusion_matrix(y_test, y_pred)[4, 8]))
print("French           " + str(confusion_matrix(y_test, y_pred)[5, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[5, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[5, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[5, 3]) + "                " + str(confusion_matrix(y_test, y_pred)[5, 4]) + "                " + str(confusion_matrix(y_test, y_pred)[5, 5]) + "                " + str(confusion_matrix(y_test, y_pred)[5, 6]) + "                " + str(confusion_matrix(y_test, y_pred)[5, 7]) + "                " + str(confusion_matrix(y_test, y_pred)[5, 8]))
print("German           " + str(confusion_matrix(y_test, y_pred)[6, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[6, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[6, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[6, 3]) + "                " + str(confusion_matrix(y_test, y_pred)[6, 4]) + "                " + str(confusion_matrix(y_test, y_pred)[6, 5]) + "                " + str(confusion_matrix(y_test, y_pred)[6, 6]) + "                " + str(confusion_matrix(y_test, y_pred)[6, 7]) + "                " + str(confusion_matrix(y_test, y_pred)[6, 8]))
print("African          " + str(confusion_matrix(y_test, y_pred)[7, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[7, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[7, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[7, 3]) + "                " + str(confusion_matrix(y_test, y_pred)[7, 4]) + "                " + str(confusion_matrix(y_test, y_pred)[7, 5]) + "                " + str(confusion_matrix(y_test, y_pred)[7, 6]) + "                " + str(confusion_matrix(y_test, y_pred)[7, 7]) + "                " + str(confusion_matrix(y_test, y_pred)[7, 8]))
print("Australian       " + str(confusion_matrix(y_test, y_pred)[8, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[8, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[8, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[8, 3]) + "                " + str(confusion_matrix(y_test, y_pred)[8, 4]) + "                " + str(confusion_matrix(y_test, y_pred)[8, 5]) + "                " + str(confusion_matrix(y_test, y_pred)[8, 6]) + "                " + str(confusion_matrix(y_test, y_pred)[8, 7]) + "                " + str(confusion_matrix(y_test, y_pred)[8, 8]))

# BELOW WAS FOR PICKING BEST PARAMATERS. UNCOMMENT IF NEEDED

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_xlabel('# neighbours')
# ax1.set_ylabel('Accuracy')
# ax1.set_title('Number of Neighbours vs Model Accuracy using <distance> weights')
# fig.set_size_inches(18, 10)
#
# neighbours = 1
# while neighbours <= len(y_train):
#
#     knn = KNeighborsClassifier(n_neighbors=neighbours, weights='distance').fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#
#     if neighbours == 1:
#         ax1.scatter(neighbours, accuracy_score(y_test, y_pred), s=10, c='b', marker="o", label="Distance Weights")
#     else:
#         ax1.scatter(neighbours, accuracy_score(y_test, y_pred), s=10, c='b', marker="o")
#
#     neighbours += 1
#
# neighbours = 1
# while neighbours <= len(y_train):
#
#     knn = KNeighborsClassifier(n_neighbors=neighbours, weights='uniform').fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#
#     if neighbours == 1:
#         ax1.scatter(neighbours, accuracy_score(y_test, y_pred), s=10, c='r', marker="o", label="Uniform Weights")
#     else:
#         ax1.scatter(neighbours, accuracy_score(y_test, y_pred), s=10, c='r', marker="o")
#
#     neighbours += 1
#
# ax1.legend()
# plt.show()

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# ax1.set_xlabel('# neighbours')
# ax1.set_ylabel('MSE')
# ax1.set_title('Number of Neighbours vs MSE using 5-Fold Cross-validation (0 < k < 10)')
# fig1.set_size_inches(18, 10)
#
# neighbours = 1
# while neighbours <= 10:
#
#     kf = KFold(n_splits=5)
#
#     mse = []
#
#     for train, test in kf.split(X):
#         knn = KNeighborsClassifier(n_neighbors=neighbours, weights='distance').fit(X[train], y[train])
#         y_pred = knn.predict(X[test])
#         mse.append(mean_squared_error(y[test], y_pred))
#
#     mean = np.mean(mse)
#     variance = sum((mse - mean) ** 2) / len(mse)
#
#     if neighbours == 1:
#         ax1.scatter(neighbours, mean, s=10, c='b', marker="o", label="Distance Weights")
#     else:
#         ax1.scatter(neighbours, mean, s=10, c='b', marker="o")
#     #ax1.errorbar(neighbours, mean, yerr=variance, c='b', marker="o", ecolor="r")
#     #print("Distance " + str(neighbours) + "-NN Mean: " + str(mean) + " +-" + str(variance))
#
#     mse = []
#
#     for train, test in kf.split(X):
#         knn = KNeighborsClassifier(n_neighbors=neighbours, weights='uniform').fit(X[train], y[train])
#         y_pred = knn.predict(X[test])
#         mse.append(mean_squared_error(y[test], y_pred))
#
#     mean = np.mean(mse)
#     variance = sum((mse - mean) ** 2) / len(mse)
#
#     if neighbours == 1:
#         ax1.scatter(neighbours, mean, s=10, c='r', marker="o", label="Uniform Weights")
#     else:
#         ax1.scatter(neighbours, mean, s=10, c='r', marker="o")
#
#     neighbours += 1
#
# ax1.legend()
# plt.show()
