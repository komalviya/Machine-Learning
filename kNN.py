import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Read in the data
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
                     X21, X22, X23, X24, X25, X26))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_xlabel('# neighbours')
# ax1.set_ylabel('Accuracy')
# ax1.set_title('Number of Neighbours vs Model Accuracy using <uniform> weights')
# fig.set_size_inches(18, 10)
#
# neighbours = 1
# while neighbours <= len(y_train):
#
#     knn = KNeighborsClassifier(n_neighbors=neighbours, weights='uniform').fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#
#     ax1.scatter(neighbours, accuracy_score(y_test, y_pred), s=10, c='b', marker="o")
#     #print("Accuracy : " + str(accuracy_score(y_test, y_pred)))
#
#     #print("ACTUAL VALUE")
#     #print("               Celtic       Indian       South American       Japanese")
#     #print("Celtic           " + str(confusion_matrix(y_test, y_pred)[0, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[0, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[0, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[0, 3]))
#     #print("Indian           " + str(confusion_matrix(y_test, y_pred)[1, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[1, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[1, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[1, 3]))
#     #print("South American   " + str(confusion_matrix(y_test, y_pred)[2, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[2, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[2, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[2, 3]))
#     #print("Japanese         " + str(confusion_matrix(y_test, y_pred)[3, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[3, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[3, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[3, 3]))
#
#     neighbours += 1
#
# plt.show()
#
# neighbours = 1
# while neighbours <= 3:
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
#     print("Distance " + str(neighbours) + "-NN Mean: " + str(mean) + " +-" + str(variance))
#
#     neighbours += 1
#
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('# neighbours')
ax1.set_ylabel('Accuracy')
ax1.set_title('Number of Neighbours vs Model Accuracy using <uniform> weights')
fig.set_size_inches(18, 10)

knn = KNeighborsClassifier(n_neighbors=2, weights='distance').fit(X_train, y_train)
y_pred = knn.predict(X_test)


print("Accuracy : " + str(accuracy_score(y_test, y_pred)))

print("                                 ACTUAL VALUE")
print("               Celtic       Indian       South American       Japanese")
print("Celtic           " + str(confusion_matrix(y_test, y_pred)[0, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[0, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[0, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[0, 3]))
print("Indian           " + str(confusion_matrix(y_test, y_pred)[1, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[1, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[1, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[1, 3]))
print("South American   " + str(confusion_matrix(y_test, y_pred)[2, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[2, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[2, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[2, 3]))
print("Japanese         " + str(confusion_matrix(y_test, y_pred)[3, 0]) + "            " + str(confusion_matrix(y_test, y_pred)[3, 1]) + "               " + str(confusion_matrix(y_test, y_pred)[3, 2]) + "                " + str(confusion_matrix(y_test, y_pred)[3, 3]))

