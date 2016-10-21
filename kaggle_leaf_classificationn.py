import numpy as np 
import pandas as pd 
import sklearn as sk 
from collections import defaultdict, Counter
from sklearn.cross_validation import train_test_split
data_train = pd.read_csv('C:/work/leaf_classification/train.csv', sep=',')

data_test = pd.read_csv('C:/work/leaf_classification/test.csv')
ID_test = data_test['id']
del data_test['id']
del data_train['id']

X_body = data_train.iloc[:,1:194]

Target = np.array(data_train['species'])

Max_values = 100
dummy_features =['species']
def select_values_dum(dataset,lis_to_dum) : 
    dummy_values={}
    for feat in lis_to_dum :
        values = [value for (value,_) in Counter(dataset[feat]).most_common(Max_values)]
        dummy_values[feat]=values
    return dummy_values
def D_encode_df(dataset):
    for (feat,dummy_values) in dum_values.items() : 
        for dv in dummy_values : 
            dummy_name ='{}_{}'.format(feat,dv)
            dataset[dummy_name] = (dataset[feat] == dv).astype(float)
        del dataset[feat]
        print('Dummy done for %s' % feat)

#dummy the column 'species' in order to get one column by type
dum_values = select_values_dum(data_train,dummy_features)
D_encode_df(data_train)


ListSpecies = list(x for x in np.unique(Target))

#print logloss for each species 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
probas_test ={}
lgloss={}
def prediction_RF_Test(List,Dataset):

    for x in List : 
        Ytrain = np.array(data_train['{}_{}'.format('species',x)])
        clfRF = RandomForestClassifier(n_estimators=250,
                n_jobs=1,
                max_depth=30,
                random_state=1337,
                min_samples_leaf=1)
        Body_train, Body_test, Target_train, Target_test = train_test_split(Dataset, Ytrain, train_size=0.7)
        clfRF.fit(Body_train, Target_train)
        probas_test[x] = clfRF.predict_proba(Body_test)[:,1]
        lgloss[x] = log_loss(Target_test, probas_test[x])
        yield probas_test[x],lgloss[x]
        
#Using gradient boosting
from sklearn import ensemble

def prediction_GB(List,Dataset):
    probasGB={}
    lglossGB={}
    for x in List : 
        Ytrain = np.array(data_train['{}_{}'.format('species',x)])   
        params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5,
              'learning_rate': 0.1, 'loss': 'deviance'}
        clf_GB = ensemble.GradientBoostingClassifier(**params)
        Body_train, Body_test, Target_train, Target_test = train_test_split(Dataset, Ytrain, train_size=0.7)
        clf_GB.fit(Body_train,Target_train)
        probasGB[x]=clf_GB.predict_proba(Body_test)[:,1]
        lglossGB[x]=log_loss(Target_test,probasGB[x])
        yield probasGB[x],lglossGB[x]

        
#call the function .. case 1 
genr_RF=prediction_RF_Test(ListSpecies,X_body)
for i in genr_RF : 
    print(i)
    
genr_GB = prediction_GB(ListSpecies,X_body)
for i in genr_GB : 
    print(i)


#compare results of both algorithms 
lgl_RF= pd.DataFrame(list(lgloss.items()), columns=['species', 'lgloss'])
lgl_GB = pd.DataFrame(list(lglossGB.items()), columns['species','lgloss'])
print('Mediane du RF est {0} avec une déviance de {1}'.format(lgl_RF['lgloss'].mean(), lgl_RF['lgloss'].std()))
print('Mediane du GB est {0} avec une déviance de {1}'.format(lgl_GB['lgloss'].mean(), lgl_GB['lgloss'].std()))


#generate the prediction for the submission 
probas={}
def prediction_RF(List):
    
    for x in List : 
        Ytrain = np.array(data_train['{}_{}'.format('species',x)])
        clfRF = RandomForestClassifier(n_estimators=250,
                n_jobs=1,
                max_depth=30,
                random_state=1337,
                min_samples_leaf=1)
        clfRF.fit(X_body,Ytrain)
        probas[x] = clfRF.predict_proba(data_test)[:,1]
        
        yield probas[x]
        
        
#call the function .. case 2
gen=prediction_RF(ListSpecies)
for i in gen : 
    print(i)
    

#generate the file for submission
probas_all = pd.DataFrame.from_dict(probas)
probas_all.set_index(ID_test,inplace=True)
probas_all.index.name=None
probas_all.to_csv('C:/work/leaf_classification/subfile.csv')