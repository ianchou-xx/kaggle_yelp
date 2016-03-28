from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

scaler = StandardScaler()
K, C, G = 200, 1, 1e-2  # parameters
x_raw, y_raw = load_svmlight_file("train.%d.feat" % K, multilabel=True)
x_raw = x_raw.toarray()
scaler.fit(x_raw)
X = scaler.transform(x_raw)
Y = MultiLabelBinarizer().fit_transform(y_raw)
clf = OneVsRestClassifier(SVC(C=C, gamma=G))
clf.fit(X, Y)

xt_raw, yt_raw = load_svmlight_file("test.sample.%d.feat" % K)
xt_raw = xt_raw.toarray()
Xt = scaler.transform(xt_raw)
Yt_p = clf.predict(Xt)

with open("submission.csv", "w") as fw, open("test_photo_to_biz.sample.csv") as f_map:
    fw.write("business_id,labels\n")
    f_map.readline()
    labels = {}
    for pred, line in zip(Yt_p, f_map):
        pid, bid = line.strip().split(",")
        if bid in labels:
            labels[bid].append(pred)
        else:
            labels[bid] = [pred,]

    for bid in labels:
        count = [0] * 9
        for label in labels[bid]:
            for idx in range(9):
                count[idx] += label[idx]
        on, n = [], len(labels[bid])
        for idx in range(9):
            if count[idx] > n/2:
                on += [str(idx)]
        fw.write("%s,%s\n" % (bid, " ".join(on)))
