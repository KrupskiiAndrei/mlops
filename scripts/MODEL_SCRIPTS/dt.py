import sys
import os
import yaml
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Проверка наличия двух аргументов командной строки: data-file и model
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython dt.py data-file model\n")
    sys.exit(1)

# Имя файла с данными на входе
f_input = sys.argv[1]

# Имя файла модели на выходе
f_output = os.path.join("models", sys.argv[2])
os.makedirs(os.path.join("models"), exist_ok=True)

# Загрузка параметров обучения из файла params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]
p_seed = params["seed"]
p_max_depth = params["max_depth"]

# Загрузка данных из файла data-file
df = pd.read_csv(f_input, header=None)
X = df.iloc[:, [1, 2, 3]]
y = df.iloc[:, 0]

# Обучение модели Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=p_max_depth, random_state=p_seed)
clf.fit(X, y)

# Сохранение обученной модели в файл model
with open(f_output, "wb") as fd:
    pickle.dump(clf, fd)
