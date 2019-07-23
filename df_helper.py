import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.impute import SimpleImputer

from sklearn import svm, datasets 
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer, average_precision_score, auc, \
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    brier_score_loss

from sklearn.model_selection import learning_curve

from scipy.stats import randint as sp_randint , uniform

from scipy.stats import chi2_contingency
from inspect import signature
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# importlib.reload(dfh)


def find_alpha_in_columns(X):
    """
        returns the non numeric elements in all the columns as a dict

    """
    
    for col in X.columns: 

        X_no_nan=X[col].dropna()
        ind_non_num=pd.to_numeric(X_no_nan, errors='coerce').isnull()

        X_alpha=X_no_nan.loc[ind_non_num]
        if X_alpha.shape[0]>0:
            col_alpha[col]=np.unique(X_alpha)
    return(col_alpha)


def na_columns_plot(df,figsize=(16,8)):
    # plots the percentage of na by columns 

    plt.figure(figsize=figsize)
    null_values_count=df.isnull().sum()
    perc_missing_values=100.*null_values_count/df.shape[0]
    
    plt.subplot(211)
    perc_missing_values.hist()
    plt.xlabel('% missing data by column');
    plt.ylabel('counts')
    
    plt.subplot(212)
    perc_missing_values[perc_missing_values>0].plot(kind='bar')

    plt.ylabel('% missing data by column');
    plt.gca().tick_params(axis='x', which='major', labelsize=8)
    plt.gca().tick_params(axis='x', which='minor', labelsize=8)
    
def na_columns_list(df,threshold=20):
    # returns the columns with more than threshold % of na 
    
    null_values_count=df.isnull().sum()
    perc_missing_values=100.*null_values_count/df.shape[0]

    ind_mv=perc_missing_values>threshold

    high_missing_values=perc_missing_values[ind_mv]
    
    print('found %d columns on %d' % (len(high_missing_values) , len(df.columns) ))
    
    return(list(high_missing_values.index),perc_missing_values)


def na_rows_plot(df):
    # plots the percentage of na by rows
    
    number_of_columns=df.shape[1]
    null_values_rows=df.isnull().sum(axis=1)

    perc_missing_values_rows=100.*null_values_rows/number_of_columns
    perc_missing_values_rows.hist()
    plt.xlabel('Percentage of missing data in rows');
    
def na_rows_list(df, threshold_rows=12):
    # returns the rows with more than threshold % of na 
    
    number_of_columns=df.shape[1]
    null_values_rows=df.isnull().sum(axis=1)

    perc_missing_values_rows=100.*null_values_rows/number_of_columns    
    missing_ind  =perc_missing_values_rows >=threshold_rows
    
    missing_rows=df.index[missing_ind]
    print('found %d rows on %d' % (len(missing_rows) , len(df) ))
    return(missing_rows)
    
def find_categorical_columns(df, threshold=21):
    # find the floats that are categorical values
    
    categorical_columns=[]
    cardinality=[]
    
    other_columns=[]
    other_cardinality=[]
    
    
    for col in df.columns:
        unique=np.unique(df[col].dropna() )
        if unique.shape[0]<threshold:
            categorical_columns.append(col)
            cardinality.append(unique.shape[0])
        else:
            other_columns.append(col)
            other_cardinality.append(unique.shape[0])
        
    return(categorical_columns,cardinality,other_columns, other_cardinality)

def find_string_columns(nf):
    colnames_numerics_only = nf.select_dtypes(include=[np.number]).columns.tolist()
    col_cat=list( set(nf.columns)- set(colnames_numerics_only) )
    return(col_cat)

def plot_correlation_matrix(df,figsize=(10,10)):
    # plots the correlation matrix 
    plt.figure(figsize=figsize)
    corr=df.corr()
    plt.matshow(corr)

    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns, cmap="gnuplot2")
    return(corr)

def filter_correlations(dataCorr, threshold=0.7,**kwargs):
    # returns the highest correlation pairs from the correlation matrix dataCorr
    
    dataCorr = dataCorr[abs(dataCorr) >= threshold].stack().reset_index()
    dataCorr = dataCorr[dataCorr['level_0'].astype(str)!=dataCorr['level_1'].astype(str)]

    # filtering out lower/upper triangular duplicates 
    dataCorr['ordered-cols'] = dataCorr.apply(lambda x: '-'.join(sorted([x['level_0'],x['level_1']])),axis=1)
    dataCorr = dataCorr.drop_duplicates(['ordered-cols'])
    dataCorr.drop(['ordered-cols'], axis=1, inplace=True)
    
    return(dataCorr)

def find_str_in_cols(nf,search_str):
    # finds the string in the columns, for debugging
    for col in nf.columns:
        found_str=nf[nf[col]==search_str]
        if len(found_str)>0:
            print(col)


def large_mean_columns(df,threshold=5):
    # finds the columns with a large mean
    # e.g. to normalize before PCA
    
    nf=df.mean()
    nf=nf[nf>threshold]
    return(list(nf.index),nf)


def normalize_df(nf):
    scaler= MaxAbsScaler()
    nf_s=scaler.fit_transform(nf)
    nf_s=pd.DataFrame(nf_s, columns=nf.columns)
    return(nf_s)



def plot_pca(df,frac=0.2,fillna=None):
    # PCA plot to find optimal number of components
    
    pca = PCA(n_components='mle', svd_solver='full')
    if fillna is None:
        X=df.sample(frac=frac).dropna()
    else:
        X=df.sample(frac=frac).fillna(fillna)
    fit_pca=pca.fit_transform(X)    

    y_expl=100.*np.cumsum(pca.explained_variance_ratio_)
    plt.plot( y_expl,'.-' );
    plt.xlabel('number of components')
    plt.ylabel('explained variance');
    return(fit_pca)

def reduce_dim_pca(df,n_components):
    pca = PCA(n_components=n_components, svd_solver='full')
    fit_pca=pca.fit_transform(df)    

    return(fit_pca)


def cluster(X,nc,method='minibatch',frac=0.2):
    # helper function
    
    if method=='minibatch':
        kmeans = MiniBatchKMeans(n_clusters=nc, random_state=0).fit(X)
    else:
        kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)
    
    dist=np.abs(kmeans.score(X))
    centers=kmeans.cluster_centers_

    return(dist,centers)

def find_n_clusters(df,ncl,verbose=False,**kwargs):
    # fast clustering to assess how many clusters can be found
    
    dist=[]
    centers=[]

    for nc in ncl:
        d,c=cluster(df,nc,**kwargs)
        dist.append(d)
        centers.append(c)
        if verbose:
            print(nc,d)

    return(np.array(dist),np.array(centers))

def plot_pr_mat(df):
  
    for n in df.index:
  
        y_score=df.loc[n,'y_score']
        y_test=df.loc[n,'y_test']
        name=df.loc[n,'model']
        average_precision = average_precision_score(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision,  alpha=0.5,
                 where='post', label ='{0:}  AP={1:0.2f}'.format(name,average_precision))
    plt.xlabel('Recall = TP/CP = (TP/(TP+FN))')
    plt.ylabel('Precision = TP/PP = (TP/(TP+FP))')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.legend()
    
def plot_roc_mat(df):

    for n in df.index:
        y_score=df.loc[n,'y_score']
        y_test=df.loc[n,'y_test']

        name=df.loc[n,'model'] 

        fpr, tpr, thresholds = roc_curve(y_test,  y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=1, label='%s (AUC = %0.2f)' % (name,roc_auc));
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                   label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR = FP/(FP+TN)')
    plt.ylabel('TPR = Recall = (TP/(TP+FN)) ')
    plt.title('ROC viewed')
    plt.legend(loc="lower right");
    plt.axis('square');

def add_metrics(y_test,y_score,y_pred,model,df=None,DOE=None):
    # records the scores in a DataFrame
    """

    Example:
    y_score = clf.predict_proba(X_test)[:,1]
    y_pred = clf.predict(X_test)
    rf=dfh.add_metrics(y_test,y_score,y_pred,'label')

    DOE is an optional DOE vector
    
    """  
    average_precision = average_precision_score(y_test, y_score)
    cm=confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp =cm.ravel()

    if DOE is None:
        df2=pd.DataFrame([[model,tn, fp, fn, tp,average_precision,y_score,y_pred,y_test]],
                       columns=['model','tn', 'fp', 'fn', 'tp','AP','y_score','y_pred','y_test'])
    else:
        df2=pd.DataFrame([[model,tn, fp, fn, tp,average_precision,y_score,y_pred,y_test,DOE]],
                       columns=['model','tn', 'fp', 'fn', 'tp','AP','y_score','y_pred','y_test','DOE'])
        
    if df is None:
        df=df2
    else:
        df=pd.concat((df,df2),ignore_index=True)
    return(df)

def print_cm(y_test, y_pred):
    """

    Example:
    y_score = clf.predict_proba(X_test)[:,1]
    y_pred = clf.predict(X_test)
    rf=dfh.add_metrics(y_test,y_score,y_pred,'label')

    
    """
    
    
    print('Confusion Matrix')
    print('C true,predicted')
    print()
    cm=confusion_matrix(y_test, y_pred)
    print(cm)
    print()
    tn, fp, fn, tp =cm.ravel()

    print('true negatives  : true 0, predicted 0: ',tn)
    print('false positives : true 0, predicted 1: ',fp)
    print('false negatives : true 1, predicted 0: ',fn)
    print('true positives  : true 1, predicted 1: ',tp)
    

    
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring='roc_auc'):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def multinomial_chi2_independence(X,Y):
    # returns the chi2_contingency for all columns in X by separating in Y=0 and Y=1
    
    df=X.copy()
    df['Y']=Y

    signific=[]

    columns=X.columns
    for col in columns:
        contingency_table=df.groupby(by='Y')[col].value_counts().unstack()
        chi2, p, dof,expect=chi2_contingency(contingency_table.values)
        signific.append(p)
    return( np.array(signific), columns)

def plot_feature_importance(X_train,clf,figsize=(16,4),n_features=30):
    # plots the feature important for the classifier clf
    
    feat=pd.DataFrame.from_dict(dict(zip(X_train.columns,clf.feature_importances_) ), orient='index',columns=['importance'])
    feat=feat.sort_values(by=['importance'],ascending=False)

    feat.plot(kind='bar',figsize=figsize);
    plt.xlim(0,n_features);
    
def dict_clf_estimate(cl_dict,data_set):
    """
    loops over dictionary of classifiers / pipelines
    
    data_set: data_set['Xtrain'] data_set['Ytrain'] data_set['Xtest'] data_set['Ytest']
    cl_dict= {label:clf}
    
    returns df add_metrics

    """
    
    c=0
    fit_cl=[]
    
    for  label,clf in cl_dict.items():
        print(label)
        
        clf.fit(data_set['Xtrain'], data_set['Ytrain'])
        y_score = clf.predict_proba(data_set['Xtest'] )[:,1]
        y_pred = clf.predict(data_set['Xtest'])
        Y_test=data_set['Ytest']

        if c==0:
            rf=add_metrics(Y_test,y_score,y_pred,label)
        else:
            rf=add_metrics(Y_test,y_score,y_pred,label,df=rf)        
        fit_cl.append(clf)
        c+=1
        
    return(rf,fit_cl)

def plot_cm_mat(df,levels=200):

    fig = plt.figure(1, figsize=(10, 10))
    ax1=plt.subplot(2,2,1)
    ax2=plt.subplot(2,2,2)
    ax3=plt.subplot(2,2,3)
    ax4=plt.subplot(2,2,4)
    
    
    for n in df.index:
  
        y_score=df.loc[n,'y_score']
        y_test=df.loc[n,'y_test']
        name=df.loc[n,'model']

        tn, fp, fn, tp,t=confusion_matrix_threshold(y_test,y_score,levels=levels,normalize_class=False);

        
        ax1.plot(t,tn,'-',label=name);
        ax1.set_ylabel('TN')
        ax1.set_xlabel('threshold')

        ax4.plot(t,tp,'-',label=name);
        ax4.set_ylabel('TP')
        ax4.set_xlabel('threshold')

        ax3.plot(t,fn,'--',label=name);
        ax3.set_ylabel('FN')
        ax3.set_xlabel('threshold')

        ax2.plot(t,fp,'--',label=name);
        ax2.set_ylabel('FP')
        ax2.set_xlabel('threshold')

    plt.legend()
        
    return(ax1,ax2,ax3,ax4)


def confusion_matrix_threshold(y_test,y_prob,levels=200,normalize_all=False,normalize_class=True):
    
    thresholds=np.linspace(0,1,levels)
    tp=0*thresholds
    tn=0*thresholds
    fp=0*thresholds
    fn=0*thresholds

    n=0
    for threshold in thresholds:
        y_pred_threshold=y_prob > threshold 
        tp[n]=np.sum( y_test & y_pred_threshold)
        tn[n]=np.sum( ~y_test & ~y_pred_threshold)
        fp[n]=np.sum( ~y_test & y_pred_threshold)
        fn[n]=np.sum( y_test & ~y_pred_threshold)

        n+=1
        
    if normalize_all:
        n=y_test.shape[0]
        return(tn/n, fp/n, fn/n, tp/n,thresholds )
    elif normalize_class:
        npos=np.sum(y_test)
        nn=y_test.shape[0]-npos
        
        return(tn/nn, fp/nn, fn/npos, tp/npos,thresholds )
    else:
        return(tn, fp, fn, tp,thresholds )
    
def plot_prob_cal(df,bins=10):

    fig = plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for n in df.index:

        prob_pos=df.loc[n,'y_score']
        name=df.loc[n,'model']
        y_test=df.loc[n,'y_test']

        prob_pos =   (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=bins)
        brier_score = brier_score_loss(y_test, prob_pos, pos_label=1)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s  (%1.3f)" % (name,brier_score ))

        ax2.hist(prob_pos, range=(0, 1), bins=bins, label=name,
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)