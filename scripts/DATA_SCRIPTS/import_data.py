import csv
import psycopg2

# Установите параметры подключения
conn_params = {
    'host': '192.168.1.121',
    'database': 'infordb',
    'user': 'infor',
    'password': '123'
}

# Создаем подключение к базе данных
conn = psycopg2.connect(**conn_params)
cursor = conn.cursor()

# Открываем файл CSV для чтения
with open('/home/infor/project/mlops/data/RAW/Knal.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Пропустить заголовок
    for row in reader:
        # Преобразуем пустые строки в NULL и загружаем данные в таблицу
        cursor.execute(
            """
            INSERT INTO Knal(Date, Vpost, pic, Value, kTum, PPP, DPM, AES) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [None if x == '' else x for x in row]
        )

# Фиксируем транзакцию
conn.commit()

# Закрываем курсор и соединение
cursor.close()
conn.close()

