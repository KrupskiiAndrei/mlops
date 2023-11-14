import os
import sys
import pickle
import json

import pandas as pd

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py data-file model\n")
    sys.exit(1)

f_input = sys.argv[1]
f_model = sys.argv[2]

df = pd.read_csv(f_input, header=None)
X = df.iloc[:, [1, 2, 3]]
y = df.iloc[:, 0]

with open(f_model, "rb") as fd:
    clf = pickle.load(fd)

score = clf.score(X, y)

prc_file = os.path.join("evaluate", "score.json")
os.makedirs(os.path.join("evaluate"), exist_ok=True)

with open(prc_file, "w") as fd:
    json.dump({"score": score}, fd)
