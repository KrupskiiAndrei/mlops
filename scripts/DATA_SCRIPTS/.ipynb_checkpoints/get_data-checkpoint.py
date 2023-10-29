import pandas as pd

# Загрузка данных из файла
file_path = "/home/infor/project/mlops/data/RAW/Knal.csv"
data = pd.read_csv(file_path)

# Разделение данных
train_data = data.iloc[:-2]
test_data = data.iloc[-2:-1]

# Сохранение данных в соответствующие файлы
test_file_path = "/home/infor/project/mlops/data/RAW/test.csv"
train_file_path = "/home/infor/project/mlops/data/RAW/train.csv"

test_data.to_csv(test_file_path, index=False)
train_data.to_csv(train_file_path, index=False)
