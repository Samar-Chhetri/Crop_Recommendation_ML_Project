import os, sys

import pandas as pd
import numpy as np
import pickle
import mysql.connector
from mysql.connector import Error
from sklearn.metrics import accuracy_score

from src.exception import CustomException


# To create server connection
def create_server_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
        host = host_name,
        user = user_name,
        passwd = user_password
        )
        print("MySQL server connection successful")
    
    
    except Error as e:
        print(f" Error: '{e}'")
        
    return connection





# To connect to database
def create_db_connection(host_name, user_name, user_password, db_name):
    connection =None
    try:
        connection = mysql.connector.connect(
        host = host_name,
        user = user_name,
        passwd = user_password,
        database = db_name
        )
        print("MySQL databse connection successfully")
    
    except Error as e:
        print(f"Error : '{e}'")
    return connection



# To execute query
def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query was successful")
    
    
    except Error as e:
        print(f"Error : '{e}'")


# To read query
def read_query(connection, query):
    cursor =connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"Error : '{e}'")




def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)


    except Exception as e:
        raise CustomException(e, sys)