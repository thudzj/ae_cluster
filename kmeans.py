import torchfile
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

images = torchfile.load('data/image.t7')
labels = torchfile.load('data/label.t7')
features = torchfile.load('data/feature.t7')
features_learnt = torchfile.load('data/feature_100.t7')
n_points = 1000
images = images.reshape(n_points, -1)
pca = PCA(n_components=10)
pca.fit(images)
images_10d = pca.transform(images)

cls_cnt = [0] * 10
for i in range(10):
    cls_cnt[i] = np.sum(labels == i)
h_c = 0
for i in range(10):
    h_c -= 1.0*(cls_cnt[i])/n_points*math.log(1.0*(cls_cnt[i])/n_points)/math.log(2)

def metric(labels_):
    I = 0
    h_o = 0

    tp_fp = 0
    tp = 0
    fn_tn = 0
    fn = 0

    pre_cnt = {}
    pre_cls_cnt = {}

    for i in range(10):
        cnt = 0
        cls_cnt_one = [0] * 10
        for j in range(n_points):
            if labels_[j] == i:
                cnt += 1
                cls_cnt_one[labels[j]] += 1
        for j in range(10):
            if cls_cnt_one[j] > 0:
                I += 1.0*(cls_cnt_one[j]) / n_points * math.log(1.0*(cls_cnt_one[j] * n_points) / cnt / cls_cnt[j])/math.log(2)
        h_o -= 1.0*(cnt)/n_points*math.log(1.0*(cnt)/n_points)/math.log(2)

        tp_fp += cnt*(cnt-1)/2
        for j in range(10):
            tp = cls_cnt_one[j] * (cls_cnt_one[j] - 1) / 2
        for j in range(0, i):
            fn_tn += cnt * pre_cnt[j]
            for k in range(10):
                fn += cls_cnt_one[k] * pre_cls_cnt[j][k]
        pre_cnt[i] = cnt
        pre_cls_cnt[i] = cls_cnt_one

    RI = (tp + fn_tn - fn) * 1.0 / (fn_tn + tp_fp)
    print("Images, MI: %f, NMI: %f, RI: %f" % (I, 2 * I / (h_c + h_o), RI))

kmeans = KMeans(n_clusters=10, random_state=0, max_iter=500, n_jobs=-1).fit(images)
metric(kmeans.labels_)

kmeans = KMeans(n_clusters=10, random_state=0, max_iter=500, n_jobs=-1).fit(images_10d)
metric(kmeans.labels_)

kmeans = KMeans(n_clusters=10, random_state=0, max_iter=500, n_jobs=-1).fit(features)
metric(kmeans.labels_)

kmeans = KMeans(n_clusters=10, random_state=0, max_iter=500, n_jobs=-1).fit(features_learnt)
metric(kmeans.labels_)