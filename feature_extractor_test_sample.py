#!/usr/bin/python

import cv2, pickle, random
from collections import Counter

for n in [200, 100, 400]:
    with open('kmeans.%d.pkl' % n, 'rb') as handle, open("test_photo_to_biz.sample.csv") as f_map, open("test.sample.%d.feat" % n, "w") as fw:
        kmeans = pickle.load(handle)
        surf = cv2.xfeatures2d.SURF_create()
        surf.setExtended(True)

        f_map.readline()    # skip first line
        for line in f_map:
            pid, bid = line.strip().split(",")
            img = cv2.imread("test_photos/" + pid + ".jpg")
            kp, des = surf.detectAndCompute(img, None)
            if des == None:
                features = []
            else:
                counter = Counter(kmeans.predict(des)).items()
                counter.sort()
                features = [str(pair[0]) + ":" + str(pair[1]) for pair in counter]
            fw.write("0 " + " ".join(features) + "\n")
