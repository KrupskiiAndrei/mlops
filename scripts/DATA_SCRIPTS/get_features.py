import sys
import os
import io

# Проверка количества аргументов командной строки
if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 get_features.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("data", "stage1", "train.csv")

# Создание каталога для выходного файла, если он не существует
os.makedirs(os.path.join("data", "stage1"), exist_ok=True)

# Функция для обработки данных
def process_data(fd_in, fd_out):
    fd_in.readline()  # Пропуск заголовка
    for line in fd_in:
        line = line.rstrip('\n').split(',')
        date = line[0]
        vpost = line[1]
        pic = line[2]
        value = line[3]
        ktum = line[4]
        ppp = line[5]
        dpm = line[6]
        aes = line[7]
        fd_out.write("{},{},{},{},{},{},{},{}\n".format(date, vpost, pic, value, ktum, ppp, dpm, aes))

# Открытие входного файла для чтения и выходного файла для записи
with io.open(f_input, encoding="utf8") as fd_in:
    with io.open(f_output, "w", encoding="utf8") as fd_out:
        process_data(fd_in, fd_out)
