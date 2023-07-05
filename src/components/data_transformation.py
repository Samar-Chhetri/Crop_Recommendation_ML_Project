import os,sys

import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

            num_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='median')),
                ('ss', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns)
            ])

            logging.info(f"Numerical columns : {numerical_columns}")

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read data as dataframe completed")

            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = ['label']

            input_features_train_df = train_df.drop(columns=target_column_name)
            target_feature_train_df = train_df['label']

            input_features_test_df = test_df.drop(columns=target_column_name)
            target_feature_test_df = test_df['label']

            logging.info("Applying preprocesor obj to train and test dataset")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            target_feature_train_new = target_feature_train_df.map({'mungbean':0, 'muskmelon':1, 'watermelon':2, 'lentil':3, 'jute':4, 'grapes':5, 
                                                                    'mothbeans':6,'banana':7, 'rice':8, 'kidneybeans':9, 'orange':10, 'coffee':11, 
                                                                    'pigeonpeas':12, 'cotton':13,'pomegranate':14, 'coconut':15, 'mango':16, 
                                                                    'maize':17, 'blackgram':18, 'chickpea':19, 'apple':20, 'papaya':21})
            
            target_feature_test_new = target_feature_test_df.map({'mungbean':0, 'muskmelon':1, 'watermelon':2, 'lentil':3, 'jute':4, 'grapes':5, 
                                                                  'mothbeans':6,'banana':7, 'rice':8, 'kidneybeans':9, 'orange':10, 'coffee':11, 
                                                                  'pigeonpeas':12, 'cotton':13,'pomegranate':14, 'coconut':15, 'mango':16, 
                                                                  'maize':17, 'blackgram':18, 'chickpea':19, 'apple':20, 'papaya':21})
            

            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_new)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_new)]

            logging.info("Saved preprocessor object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)

