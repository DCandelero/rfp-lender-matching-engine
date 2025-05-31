""" File containing function to help on data extraction. """

import pandas as pd
from config import *


class DataTransform:
    """
    A class responsible for loading and transforming raw data files.
    """

    def __init__(self) -> None:
        """
        Initializes the DataTransform object by loading raw data files into pandas DataFrames.
        """
        #self.df_programa_jornada = pd.read_csv(PATH_RAW_PROGRAMA_JORNADA)

    def join_all_in_resp_interacao(self) -> pd.DataFrame:
        """
        Joins multiple DataFrames to consolidate data related to interaction responses.

        Returns:
            pd.DataFrame: A DataFrame containing the merged data from the specified DataFrames.
        """
        df = pd.merge(self.df_resposta_interacao, self.df_interacao, left_on = 'interacao_id', right_on = 'id', how = 'left', suffixes=('_resp_int', '_int'))

        return df
    
    def save_dataframe_processed(self, df, table_name) -> None:
        """
        receive data from a specified database table and saves it as a CSV file
        at a given path.

        Args:
            pd.DataFrame: Dataframe to be stored.
            table_name (str): The name of the database table from which to
            load data.

        Note:
            This function requires the 'pandas' library for save dataframe
        """

        path_to_save = f"{DATA_PATH_WRANGLE}/df_{table_name}.csv"
        df.to_csv(path_to_save, index=False)

