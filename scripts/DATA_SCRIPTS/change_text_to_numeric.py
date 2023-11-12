import sys
import os
import io

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 change_text_to_numeric.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("data", "stage3", "train.csv")
os.makedirs(os.path.join("data", "stage3"), exist_ok=True)

def text_to_numeric(arr):
    # Пример функции для преобразования текстовых данных в числовые
    # Заменяет 'male' на 1, 'female' на 0, это нужно заменить на вашу логику
    return [1 if x == 'male' else 0 for x in arr]

def process_data(fd_in, fd_out):
    # Считывание данных из файла и транспонирование списка для удобства обработки
    data = list(zip(*(line.strip().split(',') for line in fd_in)))
    
    # Преобразование текстовых данных в числовые для всех столбцов
    numeric_data = [text_to_numeric(column) if index != 0 else column
                    for index, column in enumerate(data)]
    
    # Запись обработанных данных в файл
    for row in zip(*numeric_data):
        fd_out.write(",".join(map(str, row)) + "\n")

with io.open(f_input, encoding="utf8") as fd_in:
    with io.open(f_output, "w", encoding="utf8") as fd_out:
        process_data(fd_in, fd_out)