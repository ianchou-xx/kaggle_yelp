#!/usr/bin/bash

import random

maps = {}
fr = open('test_photo_to_biz.csv')
fw = open('test_photo_to_biz.sample.csv', 'w')

fw.write(fr.readline())
for line in fr:
    pid, bid = line.strip().split(",")
    if bid in maps:
        maps[bid].append(pid)
    else:
        maps[bid] = [pid,]

for bid in maps:
    if len(maps[bid]) > 9:
        maps[bid] = random.sample(maps[bid], 9)
    for pid in maps[bid]:
        fw.write("%s,%s\n" % (pid, bid))
