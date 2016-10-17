import torchfile
from sklearn import svm
from sklearn.model_selection import cross_val_score

features_0 = torchfile.load('data/origin.t7')
features_20 = torchfile.load('data/ite_new_20.t7')
features_100 = torchfile.load('data/ite_new_100.t7')
labels = torchfile.load('data/label.t7')

clf1 = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf1, features_0, labels, cv=5)
print scores
print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

clf2 = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf2, features_20, labels, cv=5)
print scores
print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


clf1 = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf1, features_100, labels, cv=5)
print scores
print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
