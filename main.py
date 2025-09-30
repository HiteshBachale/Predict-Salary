"""
In the Main file we are going to load data and call respected functions for model
development
"""
import numpy as np
import pandas as pd
import sklearn
import sys
import matplotlib.pyplot as plt
import logging
import pickle
from SLR_Log import setup_logging
logger = setup_logging('main')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
class SLR_INFO:
    try:
        def __init__(self,path):
            self.df = pd.read_csv(path)
            logger.info(f"Data Loaded Successfull : {self.df.shape}")
            #self.df = self.df.drop(['Id'],axis=1)
            self.X = self.df.iloc[: , 0] # independent data
            self.y = self.df.iloc[: , 1] # dependent data
            # checking if the data is clean or not:
            logger.info(f"Missing Value in the data : {self.df.isnull().sum()}")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.3,random_state=42)
            logger.info(f'length of X_train : {len(self.X_train)}')
            logger.info(f'length of y_train : {len(self.y_train)}')

    except Exception as e:
        er_ty,er_msg,er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def algo_p1(self):
        try:
            # Give the data to Linear Regression Algorithm
            logger.info("Linear Regression Algorithm")
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            # Ensure X is 2D
            self.X_train = self.X_train.to_frame() if isinstance(self.X_train, pd.Series) else self.X_train
            self.X_test = self.X_test.to_frame() if isinstance(self.X_test, pd.Series) else self.X_test
            self.sli_reg = LinearRegression()
            self.sli_reg.fit(self.X_train, self.y_train)
            y_test_pred = self.sli_reg.predict(self.X_test)
            # Regression metrics
            logger.info(f"Mean Squared Error : {mean_squared_error(self.y_test, y_test_pred)}")
            logger.info(f"Mean Absolute Error: {mean_absolute_error(self.y_test, y_test_pred)}")
            logger.info(f"R2 Score           : {r2_score(self.y_test, y_test_pred)*100}")

            # Give the data to Train Performance
            # Creating dataframe for traning data -> X_train as X_Train_Values and y_train as Y_Train_Values
            training_data = pd.DataFrame()
            training_data['X_Train_Values'] = self.X_train.copy()  # Shallow Copy
            training_data['Y_Train_Values'] = self.y_train.copy()  # Shallow Copy
            # Calling dataframe for traning data -> X_train as X_Train_Values and y_train as Y_Train_Values
            logger.info(f'{training_data}')
            # Storing X_train Prediction data using Linear Regression As 'reg' In Y_Train_Predictions
            Y_Train_Predictions = self.sli_reg.predict(self.X_train)
            # Dataframe for traning data and predicted data comparison
            training_data['Answers_From_Model'] = Y_Train_Predictions
            # Checking dataframe for traning data and predicted data comparison
            logger.info(f'{training_data}')
            # Regression metrics
            logger.info(f"Training Loos : {mean_squared_error(self.y_train,Y_Train_Predictions)}")
            #logger.info(f"Mean Absolute Error: {mean_absolute_error(self.y_train,Y_Train_Predictions)}")
            logger.info(f"Training Accuracy :  {r2_score(self.y_train,Y_Train_Predictions)*100}")

            # Give the data to Test Performance
            # Creating dataframe for testing data -> X_test as X_Test_Values and y_test as Y_Test_Values
            testing_data = pd.DataFrame()
            testing_data['X_Test_Values'] = self.X_test.copy()  # Shallow Copy
            testing_data['Y_Test_Values'] = self.y_test.copy()  # Shallow Copy
            # Calling dataframe for testing data -> X_test as X_Test_Values and y_test as Y_Test_Values
            logger.info(f'{testing_data}')
            # Storing X_test Prediction data using Linear Regression As 'li_reg' In Y_Test_Predictions
            Y_Test_Predictions = self.sli_reg.predict(self.X_test)
            # Dataframe for testing data and predicted data comparison
            testing_data['Answers_From_Model'] = Y_Test_Predictions
            # Checking dataframe for testing data and predicted data comparison
            logger.info(f'{testing_data}')
            # Regression metrics
            logger.info(f"Testing Loos : {mean_squared_error(self.y_test, Y_Test_Predictions)}")
            # logger.info(f"Mean Absolute Error: {mean_absolute_error(self.y_test,Y_Test_Predictions)}")
            logger.info(f"Testing Accuracy :  {r2_score(self.y_test, Y_Test_Predictions) * 100}")

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def li_pred(self):
        try:
            # Checking predictions for new data points externally
            # Checking salary for 15 years experianced professional
            outcome = 9339.08172382 * 15 + 25918.438334893202
            logger.info(f'{outcome}')

            logger.info(f'********** Checking Predictions For New Data Points Externally **********')
            import warnings
            warnings.filterwarnings('ignore')
            outcome = self.sli_reg.predict([[15]])
            # As outcome having only zero (0 th) index -> [166004.66419212]
            # So in print statement we will print outcome as outcome[0] index for result as shown below
            #logger.info(f'{outcome}')
            logger.info(f'The Salary Expected for 15 years Experience was : {outcome[0]} with 94% Assurace')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def sli_pkl(self):
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            # Writing the file
            # w -> write format, wb -> write with binary format
            with open('SLR_Model.pkl', 'wb') as f:
                pickle.dump(self.sli_reg, f)

            # Reading the file
            # Checking the file
            with open('SLR_Model.pkl', 'rb') as f:
                m = pickle.load(f)

            # Predicting values based on user inputs
            logger.info(f'***** Predicting values based on user inputs pickle files *****')
            import warnings
            warnings.filterwarnings('ignore')
            outcome = self.sli_reg.predict([[16]])
            logger.info(f'The Salary Expected for 16 years Experience was : {outcome[0]} with 94% Assurace')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


if __name__ == "__main__":
    try:
        path = 'E:\\Aspire Tech Academy Bangalore\\Data Science Tools\\Machine Learning\\Machine Learning Projects\\Simple Linear Regression\\Salary_Data.csv'
        obj = SLR_INFO(path)
        obj.algo_p1()
        obj.li_pred()
        obj.sli_pkl()
    except Exception as e:
        er_ty,er_msg,er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')













