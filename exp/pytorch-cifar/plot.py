#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json

acc_file = "exp1.json"
with open(acc_file, 'r') as fp:
    acc_dict = json.load(fp)

var_list, acc_list = [], []
for var in sorted(acc_dict):
    acc = acc_dict[var]["train"][-1]
    print("{}: {}".format(var, acc))
    var_list.append(var)
    acc_list.append(acc)

print(acc_dict.keys())
plt.figure(0)
for k in sorted(acc_dict):
    plt.plot(acc_dict[k]["train"])
plt.figure(1)
for k in sorted(acc_dict):
    plt.plot(acc_dict[k]["test"])
plt.show()
