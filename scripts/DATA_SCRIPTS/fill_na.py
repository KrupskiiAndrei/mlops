import sys
import os
import io
import numpy as np

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 fill_na.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("data", "stage2", "train.csv")
os.makedirs(os.path.join("data", "stage2"), exist_ok=True)

def fill_na(values):
    # Функция для заполнения пропущенных значений
    # Если список числовой, заполняем средним значением
    if all(isinstance(i, (int, float, np.number)) for i in values):
        mean_val = np.nanmean(values)
        return [mean_val if np.isnan(x) else x for x in values]
    # Если список категориальный, заполняем модой (наиболее частым значением)
    else:
        mode_val = max(set(values), key=values.count)
        return [mode_val if x is None else x for x in values]

def process_data(fd_in, fd_out):
    # Считывание данных из файла и транспонирование списка для удобства обработки
    data = list(zip(*(line.strip().split(',') for line in fd_in)))
    
    # Заполнение пропусков в каждом столбце
    filled_data = [fill_na([float(x) if x else np.nan for x in column]) if index != 0 else column
                   for index, column in enumerate(data)]
    
    # Запись обработанных данных в файл
    for row in zip(*filled_data):
        fd_out.write(','.join(map(str, row)) + '\n')

with io.open(f_input, encoding="utf8") as fd_in:
    with io.open(f_output, "w", encoding="utf8") as fd_out:
        fd_in.readline()  # Пропускаем заголовок
        process_data(fd_in, fd_out)
