import logging

def setup_logging(script_name):
    # Create a logger object
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler for the script
    handler = logging.FileHandler(f'E:\\Aspire Tech Academy Bangalore\\Data Science Tools\\Machine Learning\\Machine Learning Projects\\Simple Linear Regression\\Log_Files\\{script_name}.log',mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

