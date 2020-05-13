#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import scipy.stats
from sklearn.preprocessing import MinMaxScaler


class Model():
    """
    Performs data transformations, executes linear regression model by calling the model_output method
    
    Arguments:
    data(df) : raw data 
    """
    
    def __init__(self,data):
        self.data = data
        
    def impute_columns(self,data,columns_impute,columns_regress,target):
        """
        Imputes the x variables based on the last 8-digit geoid and stores all the data in all_data variable to multiply the 
        weights with all data and stroes th data by dropping the nulls in y variables for regressing.

        Arguments:
            data(df) : Raw data
            columns_impute(list) : columns to be imputed
            columns_regress(list) : columns to be regressed
            target(string) : target variables

        Returns:
            all_data : scaled data which needs to be multiplied with our weights.(all census tracts)
            data : data which needs to be regressed
        """
        # imputing columns based on group means 
        if columns_impute:
            data['geoid_div'] = data['geoid'].apply(lambda x: int(str(x)[:8])) #first 8-digit in geoid
            for col in columns_impute:
                data[col] = data.groupby('geoid_div')[col].apply(lambda col : col.fillna(col.mean()))
            data = data[columns_regress + [target]] #to drop the rows only based on columns we are using
            all_data = data[columns_regress].copy() #to store all x values which are imputed
            data = data.dropna(how = 'any', axis = 0) #dropping the rows bassed on y values cuz we do not impute those
        else:
            all_data = data[columns_regress].copy()
            data = data.dropna(how = 'any', axis = 0)
        return all_data, data
 


    def scaling(self,data,all_data,columns_regress,target):
        """
        Standardize the data and target 

        Arguments:
            data(df) : Clean data (imputed and dropped rows based on target)
            all_data(df) : All rows with only columns to be regressed(output of impute_columns)
            columns_regress(list) : list of columns to be regressed
            target(string) : target variable

        Returns:
            data_transformed(df) : standardized x - variables dataframe
            all_data(df) : standardized data with all census tracts
            y_transformed(np array) : standardized target variable
        """
        # scaling x and y
        scaler = StandardScaler()
        X_transformed = scaler.fit_transform(data[columns_regress].values)
        data_transformed = pd.DataFrame(data = X_transformed, columns = columns_regress)

        all_data = scaler.fit_transform(all_data[columns_regress].values)
        all_data = pd.DataFrame(data = all_data, columns = columns_regress)

        y = data[target].values.reshape(-1,1)
        y_transformed = StandardScaler().fit_transform(y) #put multiplier in next function

        return data_transformed, all_data, y_transformed
 

    def rescale(self,data,data_transformed,all_data,multiply_cols,target,target_multiplier):
        """
        rescales the data by multiplying with -1,1 to rescale everything as high is good and return that data

        Arguments:
            data(df) : raw data
            data_transformed(df) : standardized x-variables data to be regressed
            all_data(df) : standardized data with all census tracts
            multiiply_cols(dict) : dictionary with columns and thier multiplier(-1,1) as key-value pairs
            target(string) : target variable
            target_multiplier(int) : multiplier(-1,1) for target variable

        Returns:
            data_transformed(df) : rescaled regression data
            all_data(df) : rescaled all census tracts data
            y_transformed(np array) : rescaled target variable 
        """
        for col,value in multiply_cols.items():
            data_transformed[col] = data_transformed[col] * value #only these data will go to our regression 
            all_data[col] = all_data[col] * value # all census tracts        

        y = data[target].values.reshape(-1,1)
        y_transformed = StandardScaler().fit_transform(y) * target_multiplier

        return data_transformed, all_data, y_transformed
    
    
    
    def winsorize_data(self,data_transformed,winsorize_outliers,winsorize_with_95):
        """
        Winsorize the data 

        Argument:
            winsorize_outliers(dict) : dictionary of limits for the respective columns{'col' : limit}
            winsorize_with_95(boolean) : winsorize all columns with 95 percentile(True or False) default - False

        Returns:
            X_transformed(np array) : Data after winsorization.
        """
        # winsorize based on the list or 95%
        if winsorize_outliers:
            X_transformed = np.zeros((data_transformed.shape[0],1))
            for col,limit in winsorize_outliers.items():
                x = data_transformed[col].values.reshape(-1,1)
                x = scipy.stats.mstats.winsorize(x, limits = limit) #by EDA we should come up with the percentile 
                X_transformed = np.hstack((X_transformed,x))
            X_transformed = X_transformed[:,1:] #dropping all zeros column 

        elif winsorize_with_95:
            X_transformed = np.zeros((data_transformed.shape[0],1))
            for col in columns_regress:
                x = data_transformed[col].values.reshape(-1,1)
                x = scipy.stats.mstats.winsorize(x, limits=[0, 0.05]) 
                X_transformed = np.hstack((X_transformed,x))
            X_transformed = X_transformed[:,1:]

        else:
            X_transformed = data_transformed.values

        return X_transformed
    
    
    
    def model_output(self,columns_regress,target,multiply_cols,columns_impute = None,winsorize_outliers = None,winsorize_with_95 = False,target_multiplier=1):
        '''
        Performs data imputation,scaling,winsorize and executes linear regression on specified columns and returns weights and
        all census tracts transformed data which needs to be multiplied with final weights to get the health scores.

        Arguments:
            columns_regress(list) : x variables in the regression model
            target(string) : target variable(y) in the regression model
            multiply_cols(dict) : dictionary with columns and thier multiplier(-1,1) as key-value pairsto rescale all variables
            as high is good.
            columns_impute(list) : columns_impute to be imputed(default is None)
            winsorize_outliers(dict) : dictionary of limits for the respective columns{'col' : limit} (default is None)
            winsorize_with_95(boolean) : winsorize all columns with 95 percentile(True or False) (default - False)
            target_multiplier(int) : to change the direction of y variable if needed(default is 1)

        IMPORTANT : Follow the arguments order or specify the argument name when calling the function.

        Returns :returns all census tracts transformed data(high is good)
               :prints model summary
               :returns model coefficients  
        '''
        data = self.data.copy()

        # imputing columns based on group means 
        all_data, data = self.impute_columns(data,columns_impute,columns_regress,target)

        # scaling x and y
        data_transformed, all_data, y_transformed = self.scaling(data,all_data,columns_regress,target)

        # multiply with -1,1 to rescale everything as high is good and return that data
        data_transformed, all_data, y_transformed = self.rescale(data,data_transformed,all_data,multiply_cols,target,target_multiplier)

        # winsorize based on the list or 95%
        X_transformed = self.winsorize_data(data_transformed,winsorize_outliers,winsorize_with_95)

        # linear model
        X2 = sm.add_constant(X_transformed)
        lm_model = sm.OLS(y_transformed, X2)
        res = lm_model.fit()
        print(res.summary(xname = ['const'] + columns_regress))

        params = res.params[1:] #returns without constant coefficient

        return all_data, params

