import mysql.connector
import string
from getpass import getpass
import random


def get_connection(db_name=None):
    return mysql.connector.connect(
        host="localhost",
        user=get_user(),
        password=get_password(),
        database=db_name,
    )

def get_database_name_to_create():
    s = input("database name to create: ")
    assert all(x in string.ascii_lowercase for x in s)
    return s

def get_user():
    s = input("username: ")
    return s

def get_password():
    s = getpass("password: ")
    return s

def create_new_db():
    with get_connection() as connection:
        mycursor = connection.cursor()
        database_name = get_database_name_to_create()
        mycursor.execute(f"CREATE DATABASE {database_name}")


def show_dbs():
    show_db_query = "SHOW DATABASES"
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(show_db_query)
        for db in cursor:
            print(db)


def create_testdb_mapdata_table():
    with get_connection("testdb") as connection:
        cursor = connection.cursor()
        query = """
        CREATE TABLE mapdata(
            point_number INT,
            elevation FLOAT
        )
        """
        cursor.execute(query)
        connection.commit()


def describe_testdb_mapdata_table():
    with get_connection("testdb") as connection:
        cursor = connection.cursor()
        query = "DESCRIBE mapdata"
        cursor.execute(query)
        result = cursor.fetchall()  # results from last query
        for row in result:
            print(row)


def add_random_elevation_data():
    records = get_random_elevation_records()
    with get_connection("testdb") as connection:
        cursor = connection.cursor()
        insert_query = """
        INSERT INTO mapdata
            (point_number, elevation)
            VALUES (%s,%s)
        """
        cursor.executemany(insert_query, records)
        connection.commit()


def get_random_elevation_records():
    records = []
    for i in range(20):
        point_number = random.randint(0, 655362)
        elevation = random.uniform(-1000, 1000)
        record = (str(point_number), str(elevation))
        records.append(record)
    return records


def read_elevation_data():
    with get_connection("testdb") as connection:
        cursor = connection.cursor()
        select_query = "SELECT * FROM mapdata LIMIT 5"
        cursor.execute(select_query)
        result = cursor.fetchall()  # get all previous query results
        for row in result:
            print(row)


def select_underwater_points(connection):
    cursor = connection.cursor()
    query = """
    SELECT point_number, elevation
    FROM mapdata
    WHERE elevation < 0
    ORDER BY elevation DESC
    """
    cursor.execute(query)
    for row in cursor.fetchall():
        print(row)


def move_lowest_elevation_up(connection):
    min_query = """
    SELECT MIN(elevation) AS LowestElevation
    FROM mapdata
    """
    cursor = connection.cursor()
    cursor.execute(min_query)
    min_elevation_tup, = cursor.fetchall()
    min_elevation, = min_elevation_tup
    new_elevation = min_elevation + 100
    print(f"moving {min_elevation} to {new_elevation}")

    update_query = f"""
    UPDATE mapdata
    SET elevation = {new_elevation}
    WHERE elevation = {min_elevation}
    """
    cursor.execute(update_query)
    connection.commit()


if __name__ == "__main__":
    # create_new_db()
    # create_testdb_mapdata_table()
    # describe_testdb_mapdata_table()
    # add_random_elevation_data()
    # read_elevation_data()

    with get_connection("testdb") as connection:
        for i in range(10):
            move_lowest_elevation_up(connection)
        select_underwater_points(connection)
