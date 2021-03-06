#!/usr/bin/env python2

import matplotlib.pyplot as plt
import json

reg_file = "regular_stats2.json"
sup_file = "super_stats3.json"
bic_file = "bicubic_stats3.json"
with open(reg_file, 'r') as fp:
    reg_dict = json.load(fp)
with open(sup_file, 'r') as fp:
    sup_dict = json.load(fp)
with open(bic_file, 'r') as fp:
    bic_dict = json.load(fp)

def moving_avg(data, run=20):
    avg_list = []
    for i in range(run, len(data)):
        tot = 0.0
        for j in range(run):
            tot += data[i-j]
        avg = tot/run
        avg_list.append(avg)
    return avg_list
        
print "Max reg: {}".format(max(reg_dict["test"]))
print "Max sup: {}".format(max(sup_dict["test"]))
print "Max bic: {}".format(max(bic_dict["test"]))

plt.subplot(311)
plt.plot(reg_dict["train"], color='blue')
plt.plot(bic_dict["train"], color='green')
plt.plot(sup_dict["train"], color='red')
plt.subplot(312)
#plt.scatter(range(len(reg_dict["test"])), reg_dict["test"], color='blue')
#plt.scatter(range(len(bic_dict["test"])), bic_dict["test"], color='green')
#plt.scatter(range(len(sup_dict["test"])), sup_dict["test"], color='red')
plt.plot(reg_dict["test"], color='blue')
plt.plot(bic_dict["test"], color='green')
plt.plot(sup_dict["test"], color='red')
plt.subplot(313)
plt.plot(moving_avg(reg_dict["test"]), color='blue')
plt.plot(moving_avg(bic_dict["test"]), color='green')
plt.plot(moving_avg(sup_dict["test"]), color='red')
plt.show()
