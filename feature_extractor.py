#!/usr/bin/python

import cv2, pickle, random
from collections import Counter

for n in [100, 200, 400]:
    print n
    random.seed(123)
    with open("train.csv") as f_label: 
        # load labels
        labels = {}
        f_label.readline()    # skip first line
        for line in f_label:
            bid, label = line.strip().split(",")
            label = ",".join(label.split())
            labels[bid] = label

        with open('kmeans.%d.pkl' % n, 'rb') as handle, open("train_photo_to_biz_ids.csv") as f_map, open("train.%d.feat" % n, "w") as fw_t, open("val.%d.feat" % n, "w") as fw_v:
            kmeans = pickle.load(handle)
            surf = cv2.xfeatures2d.SURF_create()
            surf.setExtended(True)

            f_map.readline()    # skip first line
            for line in f_map:
                if random.randint(1, 10) > 1:   # sampling
                    continue
                pid, bid = line.strip().split(",")
                img = cv2.imread("train_photos/" + pid + ".jpg")
                kp, des = surf.detectAndCompute(img, None)
                if len(kp) < 10:    # probably broken images
                    continue
                counter = Counter(kmeans.predict(des)).items()
                counter.sort()
                features = [str(pair[0]) + ":" + str(pair[1]) for pair in counter]
                features.sort
                if int(bid) % 2 == 0:
                    fw = fw_t
                else:
                    fw = fw_v
                fw.write(labels[bid] + " " + " ".join(features) + "\n")
