import os, sys
from src.logger import logging
from src.exception import CustomException

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mysql.connector
from mysql.connector import Error
from src.utils import create_server_connection, create_db_connection, read_query
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion path")
        try:
            pw= 'Mysql#11'
            db = 'crops'

            q1 = """
            select * from crop_details;
            """


            connection = create_server_connection(host_name= 'localhost', user_name='root', user_password=pw)
            connection = create_db_connection(host_name='localhost', user_name='root', db_name=db, user_password=pw)

            results = read_query(connection, q1)

            from_db = []

            for i in results:
                res = list(i)
                from_db.append(res)

            df = pd.DataFrame(from_db, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])

            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test data initiated')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=23)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    preprocessor = DataTransformation()
    train_arr, test_arr,_ = preprocessor.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    acc_sco = modeltrainer.initiate_model_trainer(train_arr, test_arr)
    print(acc_sco)