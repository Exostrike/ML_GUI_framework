from __future__ import division
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC, ExtraTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier,RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import RidgeClassifier,PassiveAggressiveClassifier,SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis

from scoring import accuracy
import sys

# Learn to predict each class against the other
model1 = DTC(max_depth=4) #80%
model1a = DTC(max_depth=10) #88%
model1b = DTC(max_depth=100) #84%
model1c = DTC() #83
model1d = DTC(criterion='entropy') #84%
model1e = DTC(criterion='entropy',max_depth=10) #88%
model1f = DTC(class_weight='balanced')
model2 = KNN() #81%
model2a = KNN(algorithm='auto')
model2b = KNN(algorithm='ball_tree')
model2c = KNN(algorithm='kd_tree')
model2d = KNN(algorithm='brute')
model3 = KNN(weights="distance") #86%
model3a = KNN(weights="distance",algorithm='auto')
model3b = KNN(weights="distance",algorithm='ball_tree')
model3c = KNN(weights="distance",algorithm='kd_tree')
model3d = KNN(weights="distance")
model6 = KNN(n_neighbors=7) #79%
model6a = KNN(weights="distance",n_neighbors=7)#85%
model6b = KNN(n_neighbors=10)
model6c = KNN(n_neighbors=4)
model4 = GaussianNB() #78% accuracy
model5 = svm.LinearSVC()
model5a = svm.LinearSVC(penalty='l1')
model5b = svm.LinearSVC(loss='hinge')
model5c = svm.LinearSVC(penalty='l1',loss='hinge')
model8 = MLPClassifier() # 82% accuracy
model8a = MLPClassifier() 
model9 = BernoulliNB() #gives poor result 60% accuracy
model9a = BernoulliNB(alpha=10.0, fit_prior=False) #gives poor result
model9b = BernoulliNB(alpha=10.0) #gives poor result
model10 = ExtraTreeClassifier(max_depth=6) #70% accuracy
model10a = ExtraTreeClassifier()
model10b = ExtraTreeClassifier(max_depth=10)
MNB_model = MultinomialNB()
MNB_model2 = MultinomialNB(fit_prior=False)
LDA_model = LinearDiscriminantAnalysis()
QDA_model = QuadraticDiscriminantAnalysis()
model11 = RidgeClassifier()
model12 = PassiveAggressiveClassifier()
model13 = SGDClassifier()

voting_model = VotingClassifier(estimators=[('knn', model1), ('svc', model9a), ('test', model4),('fail',model3)],
                                            voting='soft', weights=[2,2,1,1])

voting_modela = VotingClassifier(estimators=[('knn', model1), ('svc', model9a), ('test', model4),('fail',model2)],
                                            voting='soft', weights=[2,1,1,2])

voting_modelb = VotingClassifier(estimators=[('knn', model1b), ('svc', model9a),('fail',model3)],
                                            voting='soft', weights=[2,2,1])

voting_modelc = VotingClassifier(estimators=[('knn', model1d), ('svc', model9a),('fail',model3)],
                                            voting='soft', weights=[2,3,1])

voting_modeld = VotingClassifier(estimators=[('knn', model1c), ('svc', model9a), ('test', model4),('fail',model3)],
                                            voting='soft', weights=[2,2,1,1])

voting_modele = VotingClassifier(estimators=[('knn', model1a), ('svc', model3), ('test', model8)],
                                            voting='soft', weights=[1,1,1])
    
voting_modelf = VotingClassifier(estimators=[('knn', model4), ('svc', model2), ('test', model8)],
                                            voting='soft', weights=[1,1,1]) 
    
voting_modelf2 = VotingClassifier(estimators=[('knn', model4), ('svc', model2), ('test', model8)],
                                            voting='soft', weights=[1,3,2])
    
voting_modelg = VotingClassifier(estimators=[('knn', model4), ('svc', model2), ('test', model8),('svc2', model1), ('test1', model9)],
                                            voting='soft', weights=[1,3,2,1,1])
    
voting_modelh = VotingClassifier(estimators=[('knn', model2), ('svc', model4), ('test', model8),('svc2', model1), ('test1', model9)],
                                            voting='soft', weights=[1,1,1,1,1])
    
voting_model2 = VotingClassifier(estimators=[('knn', model10), ('svc', model8), ('test', model1), ('test', LDA_model), ('test', QDA_model)],
                                            voting='soft', weights=[1,1,1,1,1])
    
voting_model2a = VotingClassifier(estimators=[('knn', model10), ('svc', model8), ('test', model1), ('test', LDA_model), ('test', QDA_model),('knn2', model2b), ('svc2', model1a), ('test2', model4), ('test2', model2b), ('test2', model10b)],
                                            voting='soft', weights=[1,1,1,1,1,1,1,1,1,1])

voting_model2b = VotingClassifier(estimators=[('knn', model1a), ('svc', model1d), ('test', model2), ('test', LDA_model), ('test', model3a),('knn2', model2b), ('svc2', model8), ('test2', model4), ('test2', model1f), ('test2', model10b)],
                                            voting='soft', weights=[1,1,1,1,1,1,1,1,1,1])
    
voting_model2c = VotingClassifier(estimators=[('knn', model1a), ('svc', model1d), ('test', model2), ('test', LDA_model), ('test', model3a),('knn2', model2b), ('svc2', model8), ('test2', model4), ('test2', model1f), ('test2', model10b)],
                                            voting='soft', weights=[3,2,1,2,1,1,1,2,1,1])

voting_model2d = VotingClassifier(estimators=[('knn', model1a), ('svc', model9a),('fail',model10)],
                                            voting='soft', weights=[2,2,1])

stacking_model1 = StackingClassifier(classifiers=[model1, model9, model2,model10], 
        meta_classifier=model8, use_probas=True)
    
stacking_model1a = StackingClassifier(classifiers=[model1, model9, model2,model10,model1d,LDA_model,model3a,model8,model4], 
        meta_classifier=model8, use_probas=True)
    
stacking_model1b = StackingClassifier(classifiers=[model1a, model9, model2,model10,LDA_model,model4], 
        meta_classifier=model8, use_probas=True)

stacking_model2 = StackingClassifier(classifiers=[model1, model9, model2,model10], 
        meta_classifier=model1a, use_probas=True)
    
stacking_model2a = StackingClassifier(classifiers=[model1, model9, model2,model10,model1d,LDA_model,model3a,model8,model4], 
        meta_classifier=model1a, use_probas=True)
    
stacking_model2b = StackingClassifier(classifiers=[model2c, model1e,model9, QDA_model,model8], 
        meta_classifier=model1a, use_probas=True)
    
stacking_model3 = StackingClassifier(classifiers=[model1, model9, model2,model10], 
        meta_classifier=model4, use_probas=True)
    
stacking_model3a = StackingClassifier(classifiers=[model1, model9, model2,model10,model1d,LDA_model,model3a,model8,QDA_model], 
        meta_classifier=model4, use_probas=True)
    
stacking_model4 = StackingClassifier(classifiers=[model1a, model3, model8], 
        meta_classifier=model2, use_probas=True)
    
stacking_model4a = StackingClassifier(classifiers=[model1, model9, model2,model10,model1d,LDA_model,model3a,model8,model4], 
        meta_classifier=model2, use_probas=True)
    
stacking_model5 = StackingClassifier(classifiers=[model1a, model3, model8], 
        meta_classifier=model3, use_probas=False)
    
stacking_model5a = StackingClassifier(classifiers=[model1, model9, model2,model10,model1d,LDA_model,model3a,model8,model4], 
        meta_classifier=model3, use_probas=True)
    
stacking_model6 = StackingClassifier(classifiers=[model11,model12, model11,LDA_model,QDA_model], 
        meta_classifier=model9, use_probas=False)
    
stacking_model6a = StackingClassifier(classifiers=[model1, model9, model2,model10,model1d,LDA_model,model3a,model8,model4], 
        meta_classifier=model9, use_probas=True)
    
stacking_model6b = StackingClassifier(classifiers=[model1, model9, model2,model10,model1d,LDA_model,model3a,model8,model4], 
        meta_classifier=model9, use_probas=False)
    
boost_model = AdaBoostClassifier()
boost_modela = AdaBoostClassifier(n_estimators=100)
boost_modelb1 = AdaBoostClassifier(base_estimator=model1f)#doesn't work
boost_modelb2 = AdaBoostClassifier(base_estimator=model4)#55% accuracy
boost_modelb3 = AdaBoostClassifier(algorithm='SAMME',base_estimator=model10a)#doesn't work
boost_modelb4 = AdaBoostClassifier(learning_rate=10)#64%
boost_modelb5 = AdaBoostClassifier(base_estimator=model10)#64%
boost_modelb6 = AdaBoostClassifier(algorithm='SAMME',base_estimator=model1c)
boost_modelb7 = AdaBoostClassifier(base_estimator=model4,n_estimators=100)
boost_modelb8 = AdaBoostClassifier(base_estimator=model10a)
boost_modelb8 = AdaBoostClassifier(base_estimator=model1d)
boost_modelc = AdaBoostClassifier(algorithm='SAMME') #88%
boost_modelc1 = AdaBoostClassifier(algorithm='SAMME',base_estimator=model1d)
boost_modelc2 = AdaBoostClassifier(algorithm='SAMME',base_estimator=model10a)
boost_modelc3 = AdaBoostClassifier(algorithm='SAMME',n_estimators=100)
boost_modelc4 = AdaBoostClassifier(algorithm='SAMME',base_estimator=model4)
    
    
bag_model1 = BaggingClassifier()
bag_model1a = BaggingClassifier(n_estimators=100)
bag_model2 = BaggingClassifier(base_estimator=model2)
bag_model2 = BaggingClassifier(base_estimator=model4)
bag_model3 = BaggingClassifier(base_estimator=model9)
bag_model4 = BaggingClassifier(base_estimator=model10)
bag_model5 = BaggingClassifier(base_estimator=model5)
bag_model6 = BaggingClassifier(base_estimator=LDA_model)
bag_model7 = BaggingClassifier(base_estimator=model3)
    
RF_model1 = RandomForestClassifier()
RF_model1a = RandomForestClassifier(n_estimators=100)
RF_model2 = RandomForestClassifier(criterion='entropy')
    
GB_model1 = GradientBoostingClassifier()
GB_model2 = GradientBoostingClassifier(loss='exponential')
    
ET_model1 = ExtraTreeClassifier()
ET_model2 = ExtraTreeClassifier(criterion='entropy')

hybrid_1 = AdaBoostClassifier(base_estimator=voting_model2d)
    
hybrid_2 = AdaBoostClassifier(base_estimator=voting_model2d,n_estimators=100)
    
hybrid_2a = AdaBoostClassifier(base_estimator=boost_modelc,n_estimators=100)
    
hybrid_2b = AdaBoostClassifier(base_estimator=bag_model1)
    
hybrid_2c = AdaBoostClassifier(base_estimator=bag_model4)
    
hybrid_3 = BaggingClassifier(base_estimator=boost_model)
    
hybrid_3a = BaggingClassifier(base_estimator=bag_model4)
    
hybrid_3b = BaggingClassifier(base_estimator=boost_modelc2)
    
hybrid_4 = BaggingClassifier(base_estimator=boost_modelb1)
    
hybrid_5 = BaggingClassifier(base_estimator=boost_modelb2)
    
hybrid_6 = BaggingClassifier(base_estimator=boost_modelb7)
    
hybrid_7 = BaggingClassifier(base_estimator=boost_modelc)
    
hybrid_8 = BaggingClassifier(base_estimator=boost_modelc2)
    
hybrid_9 = BaggingClassifier(base_estimator=boost_modela)
    
hybrid_10 = BaggingClassifier(base_estimator=boost_modelb4)
    
hybrid_11 = BaggingClassifier(base_estimator=boost_modela)
    
hybrid_12 = BaggingClassifier(base_estimator=voting_model2d)
    
hybrid_13 = BaggingClassifier(base_estimator=voting_model2d)
    
hybrid_14 = VotingClassifier(estimators=[('knn', boost_model), ('svc', boost_modelb6), ('test', bag_model1),('fail',voting_modela)],
                                            voting='soft', weights=[2,2,1,1])
    
hybrid_15 = VotingClassifier(estimators=[('knn', boost_model), ('svc', boost_modelb6), ('test', bag_model1),('fail',voting_modela), ('test1', stacking_model1b)],
                                            voting='soft', weights=[1,1,1,1,1])
    
hybrid_16 = VotingClassifier(estimators=[('knn', bag_model3), ('svc', boost_modelc1),('fail',stacking_model6b)],
                                            voting='soft', weights=[2,2,1])
    
hybrid_17 = VotingClassifier(estimators=[('knn', voting_modelf2), ('svc', voting_model), ('test', stacking_model6a),('fail',boost_modelb7)],
                                            voting='soft', weights=[2,2,1,1])
    
hybrid_18 = VotingClassifier(estimators=[('knn', stacking_model2b), ('svc', voting_modele), ('test', bag_model7)],
                                            voting='soft', weights=[1,1,1])
    
hybrid_19 = stacking_model1 = StackingClassifier(classifiers=[stacking_model2b, voting_modele, boost_modelc1,bag_model1], 
        meta_classifier=model8, use_probas=True)
    
hybrid_20 = StackingClassifier(classifiers=[stacking_model5a, voting_model2c, stacking_model1,bag_model1,bag_model3,boost_modelb4,boost_modelb6,bag_model2], 
        meta_classifier=model2, use_probas=True)
    
hybrid_21 = StackingClassifier(classifiers=[stacking_model2b, voting_modele, boost_modelc1,bag_model1], 
        meta_classifier=model9, use_probas=False)
    
hybrid_22 = StackingClassifier(classifiers=[boost_model, stacking_model3a, voting_modelg,bag_model1a], 
        meta_classifier=model1a, use_probas=True)
    
hybrid_23 = StackingClassifier(classifiers=[stacking_model5, voting_model2c,bag_model6], 
        meta_classifier=model1a, use_probas=True)
    
hybrid_24 = StackingClassifier(classifiers=[hybrid_1, hybrid_3,hybrid_20,hybrid_15], 
        meta_classifier=model1a, use_probas=True)

#(https://stackoverflow.com/questions/11325019/output-on-the-console-and-file-using-python)
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self) :
        for f in self.files:
            f.flush()

f = open('results.txt', 'w')
sys.stdout = Tee(sys.stdout, f)


def base(X_train, X_test, y_train, y_test,n_classes,base1,base2,base3):

    print("")
    print('base models')
    print("")
    
    DCT_models = [model1,model1a,model1b,model1c,model1d,model1e,
                  model1f]
    
    KNN_models = [model2,model2a,model2b,model2c ,model2d,
                  model3,model3a ,model3b ,model3c ,model3d,model6,
                  model6a,model6b,model6c]
    
    other_models = [model4,model8,model8a,model9,model9a,
                    model9b,model10,model10a,model10b,LDA_model,QDA_model]

    base_models = []

    if (base1 == True):
        base_models.extend(DCT_models)

    if (base2 == True):    
        base_models.extend(KNN_models)
        
    if (base3 == True):    
        base_models.extend(other_models)        

    classifiers = []
    y_scores = []
    fprs = []
    tprs = []
    roc_aucs = []
    
    for i in base_models:
            classifier = OneVsRestClassifier(i)
            classifiers.append(classifier)
            y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
            y_scores.append(y_score)
            print('')
            print (base_models.index(i)+1,' out of ',len(base_models))
            print('')
            print(i)
            print('')
            accuracy(y_test, y_score)
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])    

            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            fprs.append(fpr["micro"])
            tprs.append(tpr["micro"])
            roc_aucs.append(roc_auc["micro"])
            print('ROC area = %0.5f' % roc_auc["micro"])    
    
    plt.figure()
    plt.figure(figsize=(10,10))
    lw = 2
    for x in range(len(fprs)):
        plt.plot(fprs[x], tprs[x], color=np.random.random(3),
                 lw=lw, label=f'model{x} (area = %0.5f)' % roc_aucs[x])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.savefig('basemodels.png', bbox_inches='tight')
    
 
def hetro(X_train, X_test, y_train, y_test,n_classes,hetro1,hetro2):
    
    print("")
    print('Heterogeneous models')
    print("")
    
    voting_models = [voting_model, voting_modela, voting_modelb,voting_modelc, 
                    voting_modeld, voting_modele, voting_modelf,voting_modelf2,
                    voting_modelg, voting_modelh, voting_model2, voting_model2a,
                    voting_model2b,voting_model2c,voting_model2d]
    
    stacking_models = [stacking_model1,stacking_model1a,stacking_model1b,
                       stacking_model2,stacking_model2a,stacking_model2b,
                       stacking_model3,stacking_model3a,stacking_model4,
                       stacking_model4a,stacking_model5,stacking_model5a,
                       stacking_model6,stacking_model6a,stacking_model6b]

    hetro_models = []

    if (hetro1 == True):
        hetro_models.extend(voting_models)

    if (hetro2 == True):    
        hetro_models.extend(stacking_models)

    classifiers = []
    y_scores = []
    fprs = []
    tprs = []
    roc_aucs = []
    
    for i in hetro_models:
            classifier = OneVsRestClassifier(i)
            classifiers.append(classifier)
            y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
            y_scores.append(y_score)
            print('')
            print (hetro_models.index(i)+1,' out of ',len(hetro_models))
            print('')
            print(i)
            print('')
            accuracy(y_test, y_score)
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])    

            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            fprs.append(fpr["micro"])
            tprs.append(tpr["micro"])
            roc_aucs.append(roc_auc["micro"])
            print('ROC area = %0.5f' % roc_auc["micro"])    
    
    plt.figure()
    plt.figure(figsize=(10,10))
    lw = 2
    for x in range(len(fprs)):
        plt.plot(fprs[x], tprs[x], color=np.random.random(3),
                 lw=lw, label=f'model{x} (area = %0.5f)' % roc_aucs[x])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.savefig('hetromodels.png', bbox_inches='tight')   

def homo(X_train, X_test, y_train, y_test,n_classes,homo1,homo2,homo3):	
	
    print("")
    print('Homogeneous models')
    print("")

    boosting_models = [boost_model,boost_modela,boost_modelb1,boost_modelb4,
                       boost_modelb5,boost_modelb6,boost_modelb7,boost_modelb8,
                       boost_modelb8,boost_modelc,boost_modelc1,boost_modelc2,
                       boost_modelc3,boost_modelc4]
    
    bagging_models = [bag_model1,bag_model1a,bag_model2,bag_model2,
                      bag_model3,bag_model4,bag_model5,
                      bag_model6,bag_model7]
    
    other_models = [RF_model1,RF_model1a,RF_model2,GB_model1,GB_model2]

    homo_models = []

    if (homo1 == True):
        homo_models.extend(boosting_models)

    if (homo2 == True):    
        homo_models.extend(bagging_models)
        
    if (homo3 == True):    
        homo_models.extend(other_models)        

    classifiers = []
    y_scores = []
    fprs = []
    tprs = []
    roc_aucs = []
    
    for i in homo_models:
            classifier = OneVsRestClassifier(i)
            classifiers.append(classifier)
            y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
            y_scores.append(y_score)
            print('')
            print (homo_models.index(i)+1,' out of ',len(homo_models))
            print('')
            print(i)
            print('')
            accuracy(y_test, y_score)
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])    

            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            fprs.append(fpr["micro"])
            tprs.append(tpr["micro"])
            roc_aucs.append(roc_auc["micro"])
            print('ROC area = %0.5f' % roc_auc["micro"])    
    
    plt.figure()
    plt.figure(figsize=(10,10))
    lw = 2
    for x in range(len(fprs)):
        plt.plot(fprs[x], tprs[x], color=np.random.random(3),
                 lw=lw, label=f'model{x} (area = %0.5f)' % roc_aucs[x])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.savefig('homomodels.png', bbox_inches='tight')

def hybrid(X_train, X_test, y_train, y_test,n_classes,hybrid1,hybrid2):	
	
    print("")
    print('Hybrid models')
    print("")
    
    hybrid_hetro_models = [hybrid_14,hybrid_15,    
                           hybrid_16,hybrid_17,hybrid_18,hybrid_19,hybrid_20,     
                           hybrid_21,hybrid_22,hybrid_23,hybrid_24]
    
    hybrid_homo_models = [hybrid_1,hybrid_2,hybrid_2,hybrid_2b,hybrid_2c,    
                          hybrid_3,hybrid_3a,hybrid_3b,hybrid_4,hybrid_5,
                          hybrid_6,hybrid_7,hybrid_8,hybrid_9,hybrid_10,    
                          hybrid_11,hybrid_12,hybrid_13]

    hybrid_models = []

    if (hybrid1 == True):
        hybrid_models.extend(hybrid_hetro_models)

    if (hybrid2 == True):    
        hybrid_models.extend(hybrid_homo_models)
    
    classifiers = []
    y_scores = []
    fprs = []
    tprs = []
    roc_aucs = []
    
    for i in hybrid_models:
            classifier = OneVsRestClassifier(i)
            classifiers.append(classifier)
            y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
            y_scores.append(y_score)
            print('')
            print (hybrid_models.index(i)+1,' out of ',len(hybrid_models))
            print('')
            print(i)
            print('')
            accuracy(y_test, y_score)
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])    

            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            fprs.append(fpr["micro"])
            tprs.append(tpr["micro"])
            roc_aucs.append(roc_auc["micro"])
            print('ROC area = %0.5f' % roc_auc["micro"])    
    
    plt.figure()
    plt.figure(figsize=(10,10))
    lw = 2
    for x in range(len(fprs)):
        plt.plot(fprs[x], tprs[x], color=np.random.random(3),
                 lw=lw, label=f'model{x} (area = %0.5f)' % roc_aucs[x])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.savefig('hybridmodels.png', bbox_inches='tight')
    
def custom_hetro(X_train, X_test, y_train, y_test,n_classes,method,model):	
	
    print("")
    print('Custom model')
    print("")

    
    custom_models = []
    custom_weights =[]
    models = []
    
    for i in model:
        if (i == "Decision_Tree"):
            custom_models.append(DTC())
        elif (i == "K_nearest_neighbor"):
            custom_models.append(KNN())        
        elif (i == "GaussianNB"):
            custom_models.append(GaussianNB())    
        elif (i == "MLPClassifier"):
            custom_models.append(MLPClassifier())    
        elif (i == "BernoulliNB"):
            custom_models.append(BernoulliNB())    
        elif (i == "Extra_Tree"):
            custom_models.append(ExtraTreeClassifier())    
        elif (i == "Linear_Discriminant_Analysis"):
            custom_models.append(LinearDiscriminantAnalysis())    
        elif (i == "Quadratic_Discriminant_Analysis"):
            custom_models.append(QuadraticDiscriminantAnalysis())
            
    for i in custom_models:
        custom_weights.append(1)        

    if (method == 'Voting'):
        voting_models =[]
        for i in custom_models:
            voting_models.append(('model',i))  
        classifier = OneVsRestClassifier(VotingClassifier(estimators=voting_models,
                                            voting='soft', weights=custom_weights))
        models.append(classifier)

    if (method == "Stacking"):    
        classifier = OneVsRestClassifier(StackingClassifier(classifiers=custom_models, 
        meta_classifier=DTC(), use_probas=True))
        models.append(classifier)
    
    classifiers = []
    y_scores = []
    fprs = []
    tprs = []
    roc_aucs = []
    
    for i in models:
            classifier = OneVsRestClassifier(i)
            classifiers.append(classifier)
            y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
            y_scores.append(y_score)
            print('')
            print (models.index(i)+1,' out of ',len(models))
            print('')
            print(i)
            print('')
            accuracy(y_test, y_score)
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])    

            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            fprs.append(fpr["micro"])
            tprs.append(tpr["micro"])
            roc_aucs.append(roc_auc["micro"])
            print('ROC area = %0.5f' % roc_auc["micro"])    
    
    plt.figure()
    lw = 2
    for x in range(len(fprs)):
        plt.plot(fprs[x], tprs[x], color=np.random.random(3),
                 lw=lw, label=f'model{x} (area = %0.5f)' % roc_aucs[x])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.savefig('custom_model.png', bbox_inches='tight')

def custom_homo(X_train, X_test, y_train, y_test,n_classes,method,model):	
	
    print("")
    print('Custom model')
    print("")
    
    custom_model = None
    models = []
    
    if (model == "Decision_Tree"):
        custom_model = DTC()
    elif (model == "K_nearest_neighbor"):
        custom_model = KNN()        
    elif (model == "GaussianNB"):
        custom_model = GaussianNB()    
    elif (model == "MLPClassifier"):
        custom_model = MLPClassifier()    
    elif (model == "BernoulliNB"):
        custom_model = BernoulliNB()    
    elif (model == "Extra_Tree"):
        custom_model = ExtraTreeClassifier()    
    elif (model == "Linear_Discriminant_Analysis"):
        custom_model = LinearDiscriminantAnalysis()    
    elif (model == "Quadratic_Discriminant_Analysis"):
        custom_model = QuadraticDiscriminantAnalysis()     
    
    if (method == 'AdaBoost'):
        classifier = OneVsRestClassifier(AdaBoostClassifier(algorithm='SAMME',base_estimator=custom_model))
        models.append(classifier)

    if (method == 'Bagging'):
        classifier = OneVsRestClassifier(BaggingClassifier(base_estimator=custom_model))
        models.append(classifier)
        
    if (method == 'Random_Forest'):
        classifier = OneVsRestClassifier(RandomForestClassifier())
        models.append(classifier)
        
    if (method == 'Gradient_Boostingt'):
        classifier = OneVsRestClassifier(GradientBoostingClassifier())         
        models.append(classifier)
        
    classifiers = []
    y_scores = []
    fprs = []
    tprs = []
    roc_aucs = []
    
    for i in models:
            classifier = OneVsRestClassifier(i)
            classifiers.append(classifier)
            y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
            y_scores.append(y_score)
            print('')
            print (models.index(i)+1,' out of ',len(models))
            print('')
            print(i)
            print('')
            accuracy(y_test, y_score)
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])    

            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            fprs.append(fpr["micro"])
            tprs.append(tpr["micro"])
            roc_aucs.append(roc_auc["micro"])
            print('ROC area = %0.5f' % roc_auc["micro"])    
    
    plt.figure()
    lw = 2
    for x in range(len(fprs)):
        plt.plot(fprs[x], tprs[x], color=np.random.random(3),
                 lw=lw, label=f'model{x} (area = %0.5f)' % roc_aucs[x])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.savefig('custom_model.png', bbox_inches='tight')    