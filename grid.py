#!/usr/bin/python

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import time

with open("grid.output", "w") as fw:
    scaler = StandardScaler()
    for k in [100, 200, 400]:
        x_raw, y_raw = load_svmlight_file("train.%d.feat" % k, multilabel=True)
        x_raw = x_raw.toarray()
        scaler.fit(x_raw)
        X = scaler.transform(x_raw)
        Y = MultiLabelBinarizer().fit_transform(y_raw)
        xv_raw, yv_raw = load_svmlight_file("val.%d.feat" % k, multilabel=True)
        xv_raw = xv_raw.toarray()
        Xv= scaler.transform(xv_raw)
        Yv = MultiLabelBinarizer().fit_transform(yv_raw)
        
        for c in [1e-2, 1, 1e2]:
            for g in [1e-2, 1, 1e2]:
                start_time = time.time()
                fw.write("k:%d, c:%f, g:%f\n" % (k, c, g))
                clf = OneVsRestClassifier(SVC(C=c, gamma=g))
                clf.fit(X, Y)
                Y_p = clf.predict(X)
                fw.write("training f1_score: %f\n" % f1_score(Y, Y_p, average='macro'))

                Yv_p = clf.predict(Xv)
                fw.write("validation f1_score: %f\n" % f1_score(Yv, Yv_p, average='macro'))
                fw.flush()
                print("--- %s seconds ---" % (time.time() - start_time))
