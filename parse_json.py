import json
import sys

def parse_json(path, max_len=150):
    seqs = {}
    comps = []
    macros = []
    f = open(path)
    data = json.load(f)
    for i in data["train_set"]:
        for j in data["train_set"][i]:
            for k in data["train_set"][i][j]:
                if data["train_set"][i][j][k]["length"]>max_len:
                    continue
                seqs[k]=data["train_set"][i][j][k]["sequence"]
                comps.append(i)
                macros.append(j)
    f.close()
    return seqs, comps, macros
