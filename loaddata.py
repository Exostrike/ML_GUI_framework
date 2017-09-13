import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from collections import Counter
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

def loaddata(file_location,target_class,removed_boundary,class_boundary,secondary_class,secondary_boundary,dropped):
    
    #Loads data
    months_df = pd.read_csv(file_location, header=0 )

    #filters data down
    months_df = months_df[months_df[target_class] >= removed_boundary]
    months_df = months_df[months_df[secondary_class] >= secondary_boundary]
    #months_df = months_df[months_df.Dead == 0]

    #makes classification
    survival_result = months_df[target_class]
    y = np.where(survival_result <= class_boundary,1,0)
    #y = np.where(np.logical_and(survival_result <= 183,remedial_result == 1),1,0)


	
    # Drops unwanted columns
    features= months_df.drop(dropped,axis=1)

    features['classification'] = y 
    features.to_csv('dataset.csv', sep='\t')

    features= features.drop('classification',axis=1)
	
    # Pull out features for future use
    features_names = features.columns.tolist()
	
	
    print ("Column names:")
    print (features_names)

    X = features.as_matrix().astype(np.int)

    print(X.shape)
    
    print('Original dataset shape {}'.format(Counter(y)))
    print('')

    #oversampling to improve accuracy
    ada = ADASYN()
    sm = SMOTE()
    ros = RandomOverSampler()
    X, y = ada.fit_sample(X, y)

    print('Resampled dataset shape {}'.format(Counter(y)))
    print('')

    y = np.array([[1,0] if l==0 else [0,1] for l in y])
    n_classes = 2
    
	#Scaler to improve results for some models               
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
	
    #Kfolds to check accuracy across multiple runs
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
 
    return X_train, X_test, y_train, y_test,n_classes