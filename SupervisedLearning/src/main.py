import pandas as pd
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from plot_learning_curve import plot_learning_curve
#from sklearn.metrics import recall_score


data = 'scalar'
datatype = 'telecom'
datasetnumber = data + ' ' + datatype
#https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
def plot_search_results(grid, params, name):
    """
    Params:
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_accuracy']
    stds_test = results['std_test_accuracy']
    means_train = results['mean_train_accuracy']
    stds_train = results['std_train_accuracy']
    means_testr = results['mean_test_recall']
    stds_testr = results['std_test_recall']
    means_trainr = results['mean_train_recall']
    stds_trainr = results['std_train_recall']
    #means_testf = results['mean_test_f1_score']
    #stds_testf = results['std_test_f1_score']
    #means_trainf = results['mean_train_f1_score']
    #stds_trainf = results['std_train_f1_score']


    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))
    #print(type(grid.param_grid))
    #params=grid.param_grid
    #print(type(params))
    ## Ploting results
    plt.figure()
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(8,4))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        print(x)
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        y_3 = np.array(means_testr[best_index])
        e_3 = np.array(stds_testr[best_index])
        y_4 = np.array(means_trainr[best_index])
        e_4 = np.array(stds_trainr[best_index])
        #y_5 = np.array(means_testf[best_index])
        #e_5 = np.array(stds_testf[best_index])
        #y_6 = np.array(means_trainf[best_index])
        #e_6 = np.array(stds_trainf[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test a')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train a' )
        ax[i].errorbar(x, y_3, e_3, linestyle='--', marker='o', label='test r')
        ax[i].errorbar(x, y_4, e_4, linestyle='-', marker='^',label='train r' )
        #ax[i].errorbar(x, y_5, e_5, linestyle='--', marker='o', label='test f')
        #ax[i].errorbar(x, y_6, e_6, linestyle='-', marker='^',label='train f' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.savefig('./' + datatype + "/" + name + datasetnumber + '.png')
    plt.show()


def rundt(xTrain,yTrain,xTest,yTest):
    #test_parms = [{'ccp_alpha' :[.001, .002, .003, .005, .01, .015, .02], 'min_samples_split': [2,4,6,8,10,12,14,16] ,'min_samples_leaf': [1,2,3,4,6,8,10] }]
    test_parms = dict({'ccp_alpha' :[.0005,.001, .002, .003, .005, .01, .015, .02, .05, .1] , 'max_depth' :[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25] })
    scores = dict({ 'accuracy':make_scorer(accuracy_score), 'recall': make_scorer(recall_score) })
    #print(dhsada)
    clf = GridSearchCV(
        tree.DecisionTreeClassifier(), test_parms, scoring=scores, return_train_score=True, refit='accuracy'
    )
    clf.fit(xTrain,yTrain)
    print(clf.best_params_)
    dt = tree.DecisionTreeClassifier()
    dt.fit(xTrain,yTrain)
    print("dtune train res " + str(clf.score(xTrain,yTrain)))
    print("dtune test res " + str(clf.score(xTest,yTest)))
    print("dtfault train res " + str(dt.score(xTrain,yTrain)))
    print("dtfault test res " + str(dt.score(xTest,yTest)))
    #print(cross_val_score(dt, xTrain, yTrain, cv=5))
    print(clf.cv_results_.keys())
    plot_search_results(clf, test_parms,'Recall DecisionTree ' + datasetnumber )
    title = "Learning Curve Decision Tree " + datasetnumber
    plot = plot_learning_curve(clf,title,xTrain,yTrain)
    plot.savefig('./'+ datatype + "/" + title + ".png")
    plot.show()
    print(clf.best_params_)
    return clf

    #for i in range(1,2):
    #    print(xTrain)
    #    dt = tree.DecisionTreeClassifier( )
        #clf = make_pipeline(preprocessing.StandardScaler(), dt)
        #print((.01*i))
    #    dt.fit(xTrain,yTrain)
    #    dtr = dt.score(xTest,yTest)
    #    dtt = dt.score(xTrain,yTrain)
    #    print(cross_val_score(dt, xTrain, yTrain, cv=5))
    #    print("dt test res " + str(dtr))
    #    print("dt train res " + str(dtt))
    #    print(dt.get_params())
    #clf = tree.DecisionTreeClassifier()

    #path = clf.cost_complexity_pruning_path(xTrain, yTrain)
    #ccp_alphas, impurities = path.ccp_alphas, path.impurities
    #print(str(min(ccp_alphas)) + " max " + str(max(ccp_alphas)) )

    #print(len(ccp_alphas))
    #for ccp_alpha in ccp_alphas:
        #clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        #clf.fit(xTrain, yTrain)
        #print("train results of " + str(ccp_alpha) + str(clf.score(xTrain,yTrain)))
        #print("test results of " + str(ccp_alpha) + str(clf.score(xTest,yTest)))
        #clfs.append(clf)


def runnn(xTrain,yTrain,xTest,yTest, layers):
    #(8,), (12,), (12,2), (8,2) (5,2),
    #(100,),(15,), (15,2),(12,2)

    test_parms = dict({'hidden_layer_sizes' :[(4,layers), (12,layers), (36, layers), (48,layers), (96,layers) ], 'activation' : ['logistic', 'tanh', 'relu']})
    #test_parms = dict({'hidden_layer_sizes' :[(12,layers) ],'max_iter' : [200, 500, 800], 'activation' : [ 'logistic', 'relu']})
    scores = dict({ 'accuracy':make_scorer(accuracy_score), 'recall': make_scorer(recall_score) })
    #print(dhsada)
    clf = GridSearchCV(
        MLPClassifier(solver='lbfgs'), test_parms, scoring=scores, return_train_score=True, refit='accuracy'
    )
    clf.fit(xTrain,yTrain)
    nn = MLPClassifier(solver='lbfgs')
    nn.fit(xTrain,yTrain)
    #plot_search_results(clf, test_parms, 'NeuralNet')
    title = "Learning Curve Neural Net " + str(layers) + datasetnumber
    plot = plot_learning_curve(clf,title,xTrain,yTrain)
    plot.savefig('./'+ datatype + "/" + title + ".png")
    plot.show()
#    print(clf.best_params_)

    return clf


def runk(xTrain,yTrain,xTest,yTest):
    test_parms = dict({'n_neighbors' : [1,2,3,4,5,6,7,8,9,10,11,12], 'p' : [1,2] })
    scores = dict({ 'accuracy':make_scorer(accuracy_score), 'recall': make_scorer(recall_score) })
    #print(dhsada)
    clf = GridSearchCV(
        KNeighborsClassifier(), test_parms, scoring=scores, return_train_score=True, refit='accuracy'
    )
    clf.fit(xTrain,yTrain)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(xTrain,yTrain)
    print(clf.best_params_)
    plot_search_results(clf, test_parms,'Recall KNN ' + datasetnumber )
    #title = "Learning Curve KNN " + datasetnumber
    #plot = plot_learning_curve(clf,title,xTrain,yTrain)
    #plot.savefig('./'+ datatype + "/" + title + ".png")
    #plot.show()
    print("knntune train res " + str(clf.score(xTrain,yTrain)))
    print("knntune test res " + str(clf.score(xTest,yTest)))
    print("knnfault train res " + str(knn.score(xTrain,yTrain)))
    print("knnfault test res " + str(knn.score(xTest,yTest)))
    return clf

def runb(xTrain,yTrain,xTest,yTest):
    test_parms = dict({'n_estimators' : [100,150,200,250], 'max_depth' : [1,2,3,4] , 'learning_rate' : [ .005, .01, .05, 0.1, .5, 1 ]})
    scores = dict({ 'accuracy':make_scorer(accuracy_score), 'recall': make_scorer(recall_score) })
    #print(dhsada)
    clf = GridSearchCV(
        GradientBoostingClassifier(), test_parms, scoring=scores, return_train_score=True, refit='accuracy'
    )
    #'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 50
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=5, random_state=0)
    #gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
    #max_depth=4)
    clf.fit(xTrain,yTrain)
    gbc.fit(xTrain,yTrain)
    print(clf.best_params_)
    print("here")
    plot_search_results(clf, test_parms, 'Recall Gradient Booster ' + datasetnumber)
    title = "Learning Curve Gradient Booster " + datasetnumber
    plot = plot_learning_curve(gbc,title,xTrain,yTrain)
    plot.savefig('./'+ datatype + "/" + title + ".png")
    plot.show()
    #print("boostune train res " + str(clf.score(xTrain,yTrain)))
    #print("boostune test res " + str(clf.score(xTest,yTest)))
    print("boosfault train res " + str(gbc.score(xTrain,yTrain)))
    print("boosfault test res " + str(gbc.score(xTest,yTest)))
    return clf
    #print(clf)
    #return clf

def runsvl(xTrain,yTrain,xTest,yTest):
    test_parms = dict({'C' : [2,5,8,12,16,33], 'max_iter' : [200,300,400,500,600,700]})
    scores = dict({ 'accuracy':make_scorer(accuracy_score), 'recall': make_scorer(recall_score) })
    #print(dhsada)
    clf = GridSearchCV(
        LinearSVC(), test_parms, scoring=scores, return_train_score=True, refit='accuracy'
    )
    sv = LinearSVC(random_state=0, tol=1e-3)
    sv.fit(xTrain, yTrain)
    clf.fit(xTrain,yTrain)
    #svres = sv.score(xTest, yTest)
    print(sv.score(xTrain,yTrain))
    print(sv.score(xTest,yTest))
    print(clf.best_params_)
    plot_search_results(clf, test_parms,'Recall LinearSVC ' + datasetnumber)
    title = "Learning Curve LinearSVC " + datasetnumber
    plot = plot_learning_curve(clf,title,xTrain,yTrain)
    plot.savefig('./'+ datatype + "/" +title + ".png")
    plot.show()
    #print("svtune train res " + str(clf.score(xTrain,yTrain)))
    #print("svtune test res " + str(clf.score(xTest,yTest)))
    #print("svfault train res " + str(sv.score(xTrain,yTrain)))
    #print("svfault test res " + str(sv.score(xTest,yTest)))
    return clf

def runsv(xTrain,yTrain,xTest,yTest):
    test_parms = dict({'C' : [2,5,8,12,16,33], 'max_iter' : [200,300,400,500,600,700], 'kernel' : ['rbf', 'sigmoid']})
    scores = dict({ 'accuracy':make_scorer(accuracy_score), 'recall': make_scorer(recall_score) })
    #print(dhsada)
    clf = GridSearchCV(
        SVC(), test_parms, scoring=scores, return_train_score=True, refit='accuracy'
    )
    sv = SVC()
    sv.fit(xTrain, yTrain)
    clf.fit(xTrain,yTrain)
    #svres = sv.score(xTest, yTest)
    print(sv.score(xTrain,yTrain))
    print(sv.score(xTest,yTest))
    #print(clf.best_params_)
    plot_search_results(clf, test_parms,'Recall SVC ' + datasetnumber)
    title = "Learning Curve SVM " + datasetnumber
    plot = plot_learning_curve(sv,title,xTrain,yTrain)
    plot.savefig('./'+ datatype + "/" + title + ".png")
    plot.show()
    #print("svtune train res " + str(clf.score(xTrain,yTrain)))
    #print("svtune test res " + str(clf.score(xTest,yTest)))
    #print("svfault train res " + str(sv.score(xTrain,yTrain)))
    #print("svfault test res " + str(sv.score(xTest,yTest)))
    return clf

def load_cardio():
    data2 = pd.read_csv("../data/cardio_train.csv")
    #x2 = data2[data2.columns[2:12]] #telecom
    x2 = data2[data2.columns[1:12]]
    #print(data2['age'])
    data2['age'] = data2['age']/365
    #print(data2['age'])
    #print(x2)
    #y2 = data2['Churn']
    x2Train, x2Test, y2Train, y2Test = train_test_split(x2, y2, test_size=0.50, random_state=42)
    y2 = data2['cardio']
    return x2,y2

def load_telecom():
    data2 = pd.read_csv("../data/telecom_churn.csv")
    x2 = data2[data2.columns[2:12]] #telecom
    y2 = data2['Churn']
    print(data2[data2.Churn == 1].shape[0])
    print(data2[data2.Churn == 0].shape[0])
    #randomly reduce dataset size by half
    #x2Train, x2Test, y2Train, y2Test = train_test_split(x2, y2, test_size=0.50)
    return x2,y2



file1 = "original.csv" #credit data
#file1 = "drug200.csv" #drug identification set
#file2 = "cardio_train.csv"
data1 = pd.read_csv("../data/" + file1)
#data2 = pd.read_csv("../data/telecom_churn.csv")
#data2 = pd.read_csv("../data/cardio_train.csv")
#print(data[data.columns[0:5]].head())
data1 = data1.dropna()
x1 = data1[data1.columns[1:4]]
y1 = data1['default']
print(data1[data1.default == 1].shape[0])
print(data1[data1.default == 0].shape[0])
#print(x1)
#print(y1)
#print(data2)
#x2 = data2[data2.columns[2:12]] #telecom
#x2 = data2[data2.columns[1:12]]
#print(data2['age'])
#data2['age'] = data2['age']/365
#print(data2['age'])
#print(x2)
#y2 = data2['Churn']
#y2 = data2['cardio']
x2,y2 = load_telecom()
#print(x2)
#print(y2)
#print(x2)
#print(y2)
#drug data
#data = pd.read_csv("../data/drug200.csv")
#x1 = data[data.columns[0:5]]
#y1 = data[['Drug']]
#y1['Drug'].replace(['drugA','drugB','drugC','drugx1','Drugy1'], [0,1,2,3,4], inplace=True)
#x1['Sex1'].replace(['M','F'], [0,1], inplace=True)
#x1['BP'].replace(['LOW','NORMAL','HIGH'], [0,1,2], inplace=True)
#x1['Cholesterol'].replace(['NORMAL','HIGH'], [0,1], inplace=True)
#print(data)
#y1 = y1[['Drug']].replace({'drugB': 1})
#print(y1)

#print(x1)
#print(y1)
if data == 'normalize':
    x2 = normalize(x2)
    x1 = normalize(x1)
    #y2 = normalize(y2)

x1Train, x1Test, y1Train, y1Test = train_test_split(x1, y1, test_size=0.22)
x2Train, x2Test, y2Train, y2Test = train_test_split(x2, y2, test_size=0.22, random_state = 20)
if data == 'scalar':
    scaler = StandardScaler()
    scaler.fit(x2Train)
    x2Train = scaler.transform(x2Train)
    # apply same transformation to test data
    x2Test = scaler.transform(x2Test)
    scaler.fit(x1Train)
    x1Train = scaler.transform(x1Train)
    # apply same transformation to test data
    x1Test = scaler.transform(x1Test)

layers = 2
#clfd = rundt(x2Train,y2Train,x2Test,y2Test)
#clfk = runk(x2Train,y2Train,x2Test,y2Test)
clfn = runnn(x2Train,y2Train,x2Test,y2Test,layers)
#clfb = runb(x2Train,y2Train,x2Test,y2Test)
#clfm = runsv(x2Train,y2Train,x2Test,y2Test)
#clfl = runsvl(x2Train,y2Train,x2Test,y2Test)
#clfd = rundt(x1Train,y1Train,x1Test,y1Test)
#clfk = runk(x1Train,y1Train,x1Test,y1Test)
#clfn = runnn(x1Train,y1Train,x1Test,y1Test,layers)
#clfb = runb(x1Train,y1Train,x1Test,y1Test)
#clfm = runsv(x1Train,y1Train,x1Test,y1Test)
#clfl = runsvl(x1Train,y1Train,x1Test,y1Test)
#print("svm train res " + str(clfm.score(x2Train,y2Train)))
#print("svm test res " + str(clfm.score(x2Test,y2Test)))
#print("svl train res " + str(clfl.score(x2Train,y2Train)))
#print("svl test res " + str(clfl.score(x2Test,y2Test)))
#print("boo train res " + str(clfb.score(x2Train,y2Train)))
#print("boo test res " + str(clfb.score(x2Test,y2Test)))
#print("knn train res " + str(clfk.score(x2Train,y2Train)))
#print("knn test res " + str(clfk.score(x2Test,y2Test)))
print("nn train res " + str(clfn.score(x2Train,y2Train)))
print("nn test res " + str(clfn.score(x2Test,y2Test)))
#print("dt train res " + str(clfd.score(x2Train,y2Train)))
#print("dt test res " + str(clfd.score(x2Test,y2Test)))

#print("svm train res " + str(clfm.score(x1Train,y1Train)))
#print("svm test res " + str(clfm.score(x1Test,y1Test)))
#print("svl train res " + str(clfl.score(x1Train,y1Train)))
#print("svl test res " + str(clfl.score(x1Test,y1Test)))
#print("boo train res " + str(clfb.score(x1Train,y1Train)))
#print("boo test res " + str(clfb.score(x1Test,y1Test)))
#print("knn train res " + str(clfk.score(x1Train,y1Train)))
#print("knn test res " + str(clfk.score(x1Test,y1Test)))
#print("nn train res " + str(clfn.score(x1Train,y1Train)))
#print("nn test res " + str(clfn.score(x1Test,y1Test)))
#print("dt train res " + str(clfd.score(x1Train,y1Train)))
#print("dt test res " + str(clfd.score(x1Test,y1Test)))
#print(clfm.best_params_)
#print(clfl.best_params_)
#print(clfb.best_params_)
#print(clfk.best_params_)
print(clfn.cv_results_)
#print(clfn.best_params_)
#print("CV Score Decision Tree Tuned " + datasetnumber + str(cross_val_score(dt, xTrain, yTrain, cv=5)))
#print("Test Score Decision Tree Tuned " + datasetnumber + dt.score(xTest, yTest))

