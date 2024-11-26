import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

class CustomDataProcessor:
    def __init__(self, imputation_strategy='mean', imputation_fill_value=None):
        self.imputation_strategy = imputation_strategy
        self.imputation_fill_value = imputation_fill_value
        self.imputer = SimpleImputer(strategy=imputation_strategy, fill_value=imputation_fill_value)
    
    def drop_all_na_columns(self, data):
        df = pd.DataFrame(data)
        all_na_columns = df.columns[df.isna().all()]
        df_dropped = df.drop(columns=all_na_columns)
        return df_dropped
    
    def fit(self, data):
        df_dropped = self.drop_all_na_columns(data)
        self.imputer.fit(df_dropped)
    
    def impute(self, new_data):
        df_new = pd.DataFrame(new_data)
        df_dropped = self.drop_all_na_columns(new_data)
        imputed_data = self.imputer.transform(df_dropped)
        df_imputed = pd.DataFrame(imputed_data, columns=df_dropped.columns)
        return df_imputed

# Example data with some variables containing NA values
original_data = np.array([[1, np.nan, 3, 4],
                          [4, np.nan, np.nan, 7],
                          [7, 8, np.nan, 10]])

new_data = np.array([[np.nan, 2, 3, 4],
                     [4, np.nan, 6, np.nan],
                     [7, 8, 9, 10]])

data_processor = CustomDataProcessor(imputation_strategy='mean')

# Fit the data processor on the original data
data_processor.fit(original_data)

# Apply the same data processor to the new data
imputed_new_data = data_processor.impute(new_data)

print("Original Data:")
print(new_data)

print("Imputed Data:")
print(imputed_new_data)