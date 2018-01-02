import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn import ensemble

          
# function to merge data for one specific store
train = pd.read_csv('train.csv')
weather = pd.read_csv('weather.csv')
key = pd.read_csv('key.csv')

# function to merge data for one specific store
def merge_store_nbr(train, weather, key, store_nbr):
    train = train[train['store_nbr'] == store_nbr]
    df = weather.merge(key, how = 'outer', on = ['station_nbr'])
    df = df.merge(train, how = 'outer', on = ['store_nbr','date'])
    df = df[df['units'] > 0]
    return df

df = merge_store_nbr(train, weather, key, 15)

# Clean method 1
def clean_data(df):
    WeatherCode = ['FC', 'TS', 'GR', 'RA', 'DZ', 'SN', 'SG', 'GS', 'PL', 'IC', 'FG', 'BR', 'UP', 'HZ', 'FU', 'VA', 
               'DU', 'DS', 'PO', 'SA', 'SS', 'PY', 'SQ', 'DR', 'SH', 'FZ', 'MI', 'PR', 'BC', 'BL', 'VC', '+']
    df['Weather'] = pd.Series(list(map(lambda x: sum(list(map(lambda y: x.find(y) >= 0, WeatherCode))), df.codesum)), index = df.index)
    df = df.replace(('-', ' '), np.nan)
    df = df.replace('M', np.nan)
    df = df.replace(('T', '  T'), 0.001)
    num_attr = ['tmax', 'tmin', 'tavg', 'dewpoint', 'wetbulb', 'heat', 'cool', 'snowfall', 
                'preciptotal', 'stnpressure','sealevel', 'resultspeed', 'resultdir', 'avgspeed', 'units']
    df[num_attr] = df[num_attr].astype(float)
    df = df.loc[:,df.isnull().sum()< df.shape[0]*.5]
    return df

df1 = clean_data(df)

## Clean method 2
#def clean_data1(df):
#    WeatherCode = ['FC', 'TS', 'GR', 'RA', 'DZ', 'SN', 'SG', 'GS', 'PL', 'IC', 'FG', 'BR', 'UP', 'HZ', 'FU', 'VA', 
#               'DU', 'DS', 'PO', 'SA', 'SS', 'PY', 'SQ', 'DR', 'SH', 'FZ', 'MI', 'PR', 'BC', 'BL', 'VC']
#    for i in WeatherCode:
#        df[i] = df.codesum.str.contains(i).astype(int) + df.codesum.str.contains(i+'\+').astype(int) # number of certain weather plus status
#    df = df.replace('-', np.nan) # replace no record to np.nan
#    df = df.replace('M', np.nan) # replace missing value to np.nan
#    df = df.replace(('T', '  T'), 0.001) # replace trace to a very small number
#    num_attr = ['tmax', 'tmin', 'tavg', 'dewpoint', 'wetbulb', 'heat', 'cool', 'snowfall', 
#                'preciptotal', 'stnpressure','sealevel', 'resultspeed', 'resultdir', 'avgspeed', 'units']
#    df[num_attr] = df[num_attr].astype(float)
#    df = df.loc[:,df.isnull().sum()< df.shape[0]*.5] # delete columns with missing value more than 50%
#    return df

#df2 = clean_data1(df)

# Study distributions for each numerical attribute

df_num = df1.select_dtypes(include = ['float64'])
for i in df_num.columns.values:
    #plt.boxplot(df_num[i].dropna())
    plt.figure()
    plt.hist(df_num[i].dropna())   
    plt.title(i)
    plt.show()
    print('The mean of variable', i,'is', np.mean(df_num[i]))
    print('The standard deviation of variable',i,'is', np.std(df_num[i]))




# Perform machine learning methods
item_nbr = df['item_nbr'].unique()

for i in item_nbr:
    # Decision tree
    def decisionTree_item_nbr(df, item_nbr):
        df = df[df['item_nbr'] == item_nbr]
        num_attr = ['tmax', 'tmin', 'tavg', 'dewpoint', 'wetbulb', 'heat', 'cool', 'preciptotal', 
                    'stnpressure','sealevel', 'resultspeed', 'resultdir', 'avgspeed', 'units']
        df = df[num_attr]
        
        df = df.dropna()
        
        n_df = pd.DataFrame(preprocessing.normalize(df))
        
        valueArray = df.values
        NewValueArray = n_df.values
        X = NewValueArray[:,0:13]
        Y = valueArray[:,13]
        test_size = 0.1
        seed = 5
        X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)
        
        # Fit regression model   
        regr = DecisionTreeRegressor(max_depth=5, min_samples_split = 2)
        regr.fit(X_train, Y_train)
        # Predict
        y = regr.predict(X_validate)
        return y,Y_validate
    
    pred_y1,true_y1 = decisionTree_item_nbr(df1,i)
    
    # Gradientboosting
    def Gradientboosting_item_nbr(df, item_nbr):
        df = df[df['item_nbr'] == item_nbr]
        num_attr = ['tmax', 'tmin', 'tavg', 'dewpoint', 'wetbulb', 'heat', 'cool', 'preciptotal', 
                    'stnpressure','sealevel', 'resultspeed', 'resultdir', 'avgspeed', 'units']
        df = df[num_attr]
        df = df.dropna()
        n_df = pd.DataFrame(preprocessing.normalize(df))
        
        valueArray = df.values
        NewValueArray = n_df.values
        X = NewValueArray[:,0:13]
        Y = valueArray[:,13]
        test_size = 0.1
        seed = 5
        X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)
        
        # Fit regression model
        #params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 1, 'learning_rate': 0.01, 'loss': 'ls'}
        params = {'max_depth': 5}
        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(X_train, Y_train)
        # Predict
        y = clf.predict(X_validate)
        
        return y,Y_validate
    
    pred_y2,true_y2 = Gradientboosting_item_nbr(df1,i)
    
    # Computer prediction accuracy score
    def accuracy_score(x,y):
        print('fake sd :  ' ,np.std(x-y))
        cor = np.corrcoef(x,y)
        print("Correlation is")
        print(cor)
        test_score = r2_score(x,y)
        print('R-2: ', test_score)
        spearman = spearmanr(x,y)
        print('spearman rank: ', spearman)
        pearson = pearsonr(x,y)
        print('pearson coef: ', pearson)
   
    
    # Draw the plot
    def draw_plot(m,v):
        x = range(len(m))
        plt.figure()
        plt.plot(x, v, color="cornflowerblue",
                 label="predict", linewidth=2)
        plt.plot(x, m, color="yellowgreen", label="true", linewidth=2)
        plt.legend()
        plt.title(i)
     
    
    # Computer prediction accuracy score
    #accuracy_score(true_y1, pred_y1)
    accuracy_score(true_y2, pred_y2)
    
    # Draw the plot
    #draw_plot(true_y1, pred_y1)
    draw_plot(true_y2,pred_y2)
