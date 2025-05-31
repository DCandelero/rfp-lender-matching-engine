""" File containing function to help on data extraction. """

import mysql.connector
import pandas as pd
from config import *


class DBAccess:
    """Access and get data from MySQL database."""

    def __init__(self) -> None:
        self.api_key: str = ""
        self.tables: list = []

    def connect(self) -> None:
        """Connects to the MySQL database."""
        self.read_key()
        self.connect_db()

    def read_key(self) -> None:
        """
        Reads an API key or similar sensitive data from a file and returns it
        as a string.

        Args:
            file_path (str): The path to the file containing the API key or
            sensitive data.

        Returns:
            str: The API key or data read from the file, trimmed of any
            leading or trailing whitespace.
        """
        with open(DB_PASSWORD, "r", encoding='UTF-8') as file:
            self.api_key = file.read().strip()

    def connect_db(self) -> None:
        """
        Establishes a connection to a MySQL database and returns the connection
        object and cursor.

        Args:
            password (str): The password for the database user.
            database (str): The name of the database to connect to.

        Returns:
            tuple: A tuple containing the connection object and a cursor for the
            database.
        """

        config = {
            "user": "timedados",
            "password": self.api_key,
            "host": "laravel-forge-db.crxgofa2ytp0.sa-east-1.rds.amazonaws.com",
            "database": DATABASE,
            "raise_on_warnings": True,
        }

        self.conn = mysql.connector.connect(**config)
        self.cursor = self.conn.cursor()

    def get_tables(self) -> list:
        """Recover the name of the tables on the database.

        Returns:
            list: List of the table names.
        """
        self.cursor.execute("SHOW TABLES")
        for (table_name,) in self.cursor:
            self.tables.append(table_name)

        return self.tables

    def load_and_save_data_from_db(self, table_name) -> None:
        """
        Loads data from a specified database table and saves it as a CSV file
        at a given path.

        Args:
            table_name (str): The name of the database table from which to
            load data.
            conn (MySQLConnection): A connection object to the database.
            PATH_NAME_TO_SAVE_CSV (str): The file path where the CSV file
            should be saved.

        Note:
            This function requires the 'pandas' library for data handling and
            'mysql.connector' for database connection.
        """

        path_to_save = f"{DATA_PATH_RAW}/df_{table_name}.csv"

        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, self.conn)
        df.to_csv(path_to_save, index=False)
