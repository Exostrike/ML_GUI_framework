from __future__ import division
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC, ExtraTreeClassifier
from sklearn.metrics import roc_curve, auc,precision_score,matthews_corrcoef
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier,RandomForestClassifier, GradientBoostingClassifier,IsolationForest
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier,BernoulliRBM
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE 
from collections import Counter
from mlxtend.classifier import StackingClassifier,Adaline,SoftmaxRegression
from statsmodels.sandbox.stats.runs import mcnemar
from sklearn.feature_selection import chi2
from scipy.stats import ttest_rel,ttest_1samp,ttest_ind,chisquare
from sklearn.pipeline import make_pipeline, make_union
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler

#note this is presently configured for 4 models, you 
def basic_averaging(X_train, X_test, y_train, y_test,n_classes,models):

    y_scorearray = []

    for i in models:
        classifier1 = OneVsRestClassifier(i)
        y_score = classifier1.fit(X_train, y_train).predict(X_test)
        y_scorearray.append(y_score)
     
    newyscore = (y_scorearray[0]+y_scorearray[1]+y_scorearray[2]+y_scorearray[3])/len(y_scorearray)
    
    return newyscore

def weighted_averaging(X_train, X_test, y_train, y_test,n_classes,models,weights):
    
    y_scorearray = []

    for i in models:
        classifier1 = OneVsRestClassifier(i)
        y_score = classifier1.fit(X_train, y_train).predict(X_test)
        y_scorearray.append(y_score)

    weight_score = ((y_scorearray[0]*weights[0])+(y_scorearray[1]*weights[1])+(y_scorearray[2]*weights[2])+(y_scorearray[3]*weights[3]))/len(y_scorearray)
    
    return weight_score

def basic_averaging_prob(X_train, X_test, y_train, y_test,n_classes,models):

    y_scorearray = []

    for i in models:
        classifier1 = OneVsRestClassifier(i)
        y_score = classifier1.fit(X_train, y_train).predict_proba(X_test)
        y_scorearray.append(y_score)
     
    newyscore = (y_scorearray[0]+y_scorearray[1]+y_scorearray[2]+y_scorearray[3])/len(y_scorearray)
    
    return newyscore

def weighted_averaging_prob(X_train, X_test, y_train, y_test,n_classes,models,weights):
    
    y_scorearray = []

    for i in models:
        classifier1 = OneVsRestClassifier(i)
        y_score = classifier1.fit(X_train, y_train).predict_proba(X_test)
        y_scorearray.append(y_score)

    weight_score = ((y_scorearray[0]*weights[0])+(y_scorearray[1]*weights[1])+(y_scorearray[2]*weights[2])+(y_scorearray[3]*weights[3]))/len(y_scorearray)
    
    return weight_score
