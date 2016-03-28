#!/bin/usr/python

import cv2, os, pickle, random, numpy
from sklearn.cluster import KMeans

surf = cv2.xfeatures2d.SURF_create()
surf.setExtended(True)
descriptors = None
for dn in ("train_photos", "test_photos"):
    for fn in os.listdir(dn):
        if os.path.isfile(dn + "/" + fn):
            if random.randint(1,10000) > 1:
                continue
            img = cv2.imread(dn + "/" + fn)
            kp, tmp = surf.detectAndCompute(img, None)
            if descriptors == None:
                descriptors = tmp
            else:
                if tmp == None:
                    print fn, dn
                else:
                    descriptors = numpy.concatenate((descriptors, tmp))
        else:
            print "ERROR!"

print "# of descriptors: " + str(len(descriptors))

kmeans = KMeans(n_clusters=400)
kmeans.fit(descriptors)
with open("kmeans.400.pkl", "wb") as handle:
    pickle.dump(kmeans, handle)
