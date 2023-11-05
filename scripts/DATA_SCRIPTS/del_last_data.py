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

# Предполагая, что у вас есть столбец с датой, который можно использовать для определения последней строки
delete_query = "DELETE FROM Knal WHERE date = (SELECT MAX(date) FROM Knal);"

# Выполняем запрос на удаление
cursor.execute(delete_query)

# Фиксируем транзакцию
conn.commit()

# Закрываем курсор и соединение
cursor.close()
conn.close()

print("Последняя запись была успешно удалена из таблицы Knal.")
