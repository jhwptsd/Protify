import json
import sys

def parse_json(path, a, b, max_len=150):
    num = -1
    seqs = {}
    comps = []
    macros = []
    f = open(path)
    data = json.load(f)
    for i in data["train_set"]:
        for j in data["train_set"][i]:
            for k in data["train_set"][i][j]:
                num = num + 1
                if data["train_set"][i][j][k]["length"]>max_len:
                    continue
                if num>=a and num<=b:
                    seqs[k]=data["train_set"][i][j][k]["sequence"]
                    comps.append(i)
                    macros.append(j)
                if num>b:
                    break
    f.close()
    return seqs, comps, macros
