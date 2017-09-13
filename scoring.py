from loaddata import loaddata
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def accuracy(y_test, y_score):
    #print(classification_report(y_test, y_score.round()))
    print("overall Accuracy:",accuracy_score(y_test, y_score.round()))
    
    print("all  precision  recall  specificity  f1")
    
    cm=(confusion_matrix(y_test.flatten(), y_score.round().flatten()))
    #print(cm)
    FP = cm[0][1]
    TP = cm[0][0]
    TN = cm[1][1]
    FN = cm[1][0]
    precision=1-((TN/TN+FP)/len(y_test))
    recall=1-((TP/TP+FN)/len(y_test))
    specificity=1-((TN/TN+FP)/len(y_test))
    f1=1-((TP/TP+FP+FN)/len(y_test))    
    print("all  ",round(precision,2),"  ",round(recall,2),"  ",round(specificity,2),"  ",round(f1,2))
    print("all Accuracy:",round(accuracy_score(y_test, y_score.round()),2))
    
    
    cm2=(confusion_matrix(y_test[:, 0], y_score.round()[:, 0]))
    #print(cm2)
    FP2 = cm2[0][1]
    TP2 = cm2[0][0]
    TN2= cm2[1][1]
    FN2 = cm2[1][0]
    precision2=1-((TN2/TN2+FP2)/len(y_test))
    recall2=1-((TP2/TP2+FN2)/len(y_test))
    specificity2=1-((TN2/TN2+FP)/len(y_test))
    fone2=1-((TP/TP+FP+FN)/len(y_test))    
    #print("Precision:",precision2,)
    #print("Recall:",recall2)
    #print("Specificity:",specificity2)
    #print("F1 score:",fone2)
    print("0   ",round(precision2,2),"   ",round(recall2,2),"   ",round(specificity2,2),"   ",round(fone2,2))
    print("0 Accuracy:",round(accuracy_score(y_test[:, 0], y_score.round()[:, 0]),2))
    
    
    cm3=(confusion_matrix(y_test[:, 1], y_score.round()[:, 1]))
    #print(cm3)
    FP3 = cm3[0][1]
    TP3 = cm3[0][0]
    TN3 = cm3[1][1]
    FN3 = cm3[1][0]
    precision3=1-((TN3/TN3+FP3)/len(y_test))
    recall3=1-((TP3/TP3+FN3)/len(y_test))
    specificity3=1-((TN3/TN3+FP3)/len(y_test))
    fone3=1-((TP3/TP3+FP3+FN3)/len(y_test))    
    #print("Precision:",precision3)
    #print("Recall:",recall3)
    #print("Specificity:",specificity3)
    #print("F1 score:",fone3)
    print("1   ",round(precision3,2),"  ",round(recall3,2),"  ",round(specificity3,2),"  ",round(fone3,2))
    print("1 Accuracy:",round(accuracy_score(y_test[:, 1], y_score.round()[:, 1]),2))
