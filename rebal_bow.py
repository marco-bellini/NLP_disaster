
# coding: utf-8

import numpy as np
import pandas as pd
import pylab as plt
import random as rd

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.util import minibatch, compounding
from nltk.stem.porter import *  

import re
import urllib.request
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_columns = None
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

nlp = spacy.load("en_core_web_md")

from spacy import displacy
import df_helper as dfh

from sklearn.utils.random import sample_without_replacement
import importlib




if 1:
    url = 'https://datasets.figure-eight.com/figure_eight_datasets/disaster_response_data/disaster_response_messages_training.csv'  
    urllib.request.urlretrieve(url, 'disaster_response_messages_training.csv')
    url = 'https://datasets.figure-eight.com/figure_eight_datasets/disaster_response_data/disaster_response_messages_test.csv'  
    urllib.request.urlretrieve(url, 'disaster_response_messages_test.csv')
    url = 'https://datasets.figure-eight.com/figure_eight_datasets/disaster_response_data/disaster_response_messages_validation.csv'  
    urllib.request.urlretrieve(url, 'disaster_response_messages_validation.csv')



from sklearn.base import TransformerMixin


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, validation_curve
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report, confusion_matrix

import sklearn.metrics as met
from sklearn.metrics import precision_recall_curve, roc_curve, auc, matthews_corrcoef
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer, average_precision_score, auc,     accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report,     brier_score_loss, roc_auc_score

from scipy.stats import randint as sp_randint , uniform
import eli5


# In[3]:


def transform_sentence(bow, message):
    # for eli5
    
    print("original: ",message)
    
    words=np.array(bow.get_feature_names())
    TR=bow.transform([ message ])
    g,ind=TR.nonzero()
    transformed=",".join(words[ind]) 
    print("transformed: ",transformed)
    print()
    
def explain_message(pos,fn_messages,pipe,dataset,top=10):
    """
    explain_message(pos,X_test[y_fn],pipe,bow1k_bal)
    """
    
    message=fn_messages.loc[pos]
    transform_sentence(pipe.steps[0][1],message)

    print('Predicted class:', pipe.predict([message ])[0] )
    display(eli5.show_prediction(pipe.steps[2][1],   dataset['Xtest'][pos,:] , target_names=[0,1],
                         feature_names= pipe.steps[0][1].get_feature_names(),top=top) )
    
    
def spacy_tokenizer_stemmer(message):
    message=re.sub("[1-9#@$'!*+%\".()!,]?;",'',message).replace('','').replace('-','')
    message=' '.join(message.split())
    doc=nlp(message)
    words=[]

    stemmer = PorterStemmer()  
    
    remove_ent=[]
    for ent in doc.ents:
        if ent.label_ in ['GPE','LOC','NORP','FAC','ORG','LANGUAGE']:
            remove_ent.append(ent.text)

    # remove punctuation etc
    for token in doc:
        if ( (~token.is_stop)   & (token.pos_!='NUM') & (token.pos_!='PUNCT') & (token.pos_!='SYM') &
           ~(token.text in (remove_ent)) & (len(token.text)>1) ):
            words.append( stemmer.stem(token.text) )
    return(words)


# In[9]:




def imbalanced_undersample(X_train,Y_train,n_false_sample, classes=[False,True]):
    """
    resamples the false class classes[0] with n_false_sample to correct for imbalanced data
    
    """
    
    n_false=Y_train.loc[Y_train==classes[0]].shape[0]
    n_true=Y_train.loc[Y_train==classes[1]].shape[0]
    ind_false=Y_train.loc[Y_train==classes[0]].index
    ind_true=Y_train.loc[Y_train==classes[1]].index

    print('Original n_true, n_false:' ,n_true,n_false)
    
    ind_s=sample_without_replacement(n_false,n_false_sample)
    ind=np.hstack((ind_true,ind_false[ind_s]))
    np.random.shuffle(ind)

    X_train2=X_train.iloc[ind].copy()
    Y_train2=Y_train.iloc[ind].copy()
    
    n_false2=Y_train2.loc[Y_train==classes[0]].shape[0]
    n_true2=Y_train2.loc[Y_train==classes[1]].shape[0]
    
    print('Resampled n_true, n_false:' ,n_true2,n_false2)
    
    return(X_train2,Y_train2)


# In[10]:


def train_bow(X_train,Y_train, X_test,Y_test,n_false_sample ,  max_df=0.9,min_df=1, max_features=2000 ):
    """
    transforms the training and test set with a reduced B.O.W. with only n_false_sample 
    usage:
    
    bow1k_bal, bow_bal, tfidf = train_bow(X_train,Y_train, X_test,Y_test,n_false_sample ,  max_df=0.9,min_df=1, max_features=2000 )
    """
    
    
    X_train2,Y_train2=imbalanced_undersample(X_train,Y_train,n_false_sample, classes=[False,True])

    bow_bal=CountVectorizer(tokenizer = spacy_tokenizer_stemmer, max_df=max_df,min_df=min_df, max_features=max_features)
    tfidf=TfidfTransformer()     

    bow_bal.fit(X_train2)
    Xbow_train=bow_bal.transform(X_train)
    X_train_tdidf2 = tfidf.fit_transform(Xbow_train)

    Xbow_test = bow_bal.transform(X_test)
    X_test_tdidf2 = tfidf.transform(Xbow_test)
    
    bow1k_bal={'Xtrain':X_train_tdidf2, 'Ytrain':Y_train, 'Xtest':X_test_tdidf2, 'Ytest':Y_test}
    return( bow1k_bal, bow_bal, tfidf )
    


# In[40]:


def clf_estimate(cl_dict,bow1k_bal,DOE):
    """
    cl_dict= {label:clf}

    """
    
    c=0
    for  label,clf in cl_dict.items():
        print(label)
        
        clf.fit(bow1k_bal['Xtrain'], bow1k_bal['Ytrain'])
        y_score = clf.predict_proba(bow1k_bal['Xtest'] )[:,1]
        y_pred = clf.predict(bow1k_bal['Xtest'])
        Y_test=bow1k_bal['Ytest']

        if c==0:
            rf=dfh.add_metrics(Y_test,y_score,y_pred,label)
        else:
            rf=dfh.add_metrics(Y_test,y_score,y_pred,label,df=rf)        
            
        c+=1
        
    return(rf)


# In[48]:


def make_doe(DOE,bow1k_bal):
    cl_dict=dict()

    for N_est in [10, 20, 50, 100, 200, 500, 1000]:
        cl_dict['RF%d' % N_est]=RandomForestClassifier(  criterion="entropy",class_weight="balanced", n_estimators=N_est)

    N_est=200


    for max_features in [50,100,200,300,500, 1000, 1500, 2000]:
        cl_dict['RF%d_mf%d' % (N_est,max_features) ]=RandomForestClassifier(  criterion="entropy",class_weight="balanced",
                                                                            n_estimators=N_est,max_features=max_features)

    max_features=1500    
    for max_depth in [4,8, 12, 16, 20, 25, 30, 40]:
        cl_dict['RF%d_mf%d_md%d' % (N_est,max_features,max_depth) ]=RandomForestClassifier(  criterion="entropy",class_weight="balanced",
                                                                            n_estimators=N_est,max_features=max_features, max_depth=max_depth)

    for C in [1e-4,1e-3,1e-2,1e-1,0.2,0.5,0.75,1.]:
        cl_dict['LR_C%g' % C]=LogisticRegression(random_state=0, solver='liblinear',penalty='l1',max_iter=200,class_weight='balanced', C=C )
    
    rf=clf_estimate(cl_dict,bow1k_bal,DOE)
    
    rf.to_pickle('DOE_%s.p' % DOE)


# In[ ]:


df=pd.read_csv('disaster_response_messages_training.csv')
df=df[['message','food']]

test_df=pd.read_csv('disaster_response_messages_test.csv')
test_df=test_df[['message','food']]

valid_df=pd.read_csv('disaster_response_messages_validation.csv')
valid_df=valid_df[['message','food']]


X_train=df['message']
Y_train=df['food']

X_test=test_df['message']
Y_test=test_df['food']

X_valid=valid_df['message']
Y_valid=valid_df['food']


# In[51]:


n_false_sample= 1000
n_features=2000

DOE='false_%s_nwords_%s_mindf_1' % (n_false_sample,n_features)
print(DOE)

bow1k_bal, bow_bal, tfidf = train_bow(X_train,Y_train, X_test,Y_test,n_false_sample ,  max_df=0.9,min_df=1, max_features=n_features )
make_doe(DOE,bow1k_bal)


# In[44]:


n_false_sample= 500
n_features=2000

DOE='false_%s_nwords_%s_mindf_1' % (n_false_sample,n_features)
print(DOE)

bow1k_bal, bow_bal, tfidf = train_bow(X_train,Y_train, X_test,Y_test,n_false_sample ,  max_df=0.9,min_df=1, max_features=n_features )
make_doe(DOE,bow1k_bal)


# In[ ]:


n_false_sample= 500
n_features=1000

DOE='false_%s_nwords_%s_mindf_1' % (n_false_sample,n_features)
print(DOE)

bow1k_bal, bow_bal, tfidf = train_bow(X_train,Y_train, X_test,Y_test,n_false_sample ,  max_df=0.9,min_df=1, max_features=n_features )
make_doe(DOE,bow1k_bal)


# In[ ]:


n_false_sample= 500
n_features=500

DOE='false_%s_nwords_%s_mindf_1' % (n_false_sample,n_features)
print(DOE)

bow1k_bal, bow_bal, tfidf = train_bow(X_train,Y_train, X_test,Y_test,n_false_sample ,  max_df=0.9,min_df=1, max_features=n_features )
make_doe(DOE,bow1k_bal)


# In[ ]:


n_false_sample= 0
n_features=500

DOE='false_%s_nwords_%s_mindf_1' % (n_false_sample,n_features)
print(DOE)

bow1k_bal, bow_bal, tfidf = train_bow(X_train,Y_train, X_test,Y_test,n_false_sample ,  max_df=0.9,min_df=1, max_features=n_features )
make_doe(DOE,bow1k_bal)

