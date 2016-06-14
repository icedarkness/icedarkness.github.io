# Importing a few necessary libraries

import numpy as np

import math  as math

import pandas as pd

import matplotlib.pyplot as pl

from sklearn import datasets

from sklearn.tree import DecisionTreeRegressor

from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

from sklearn import cross_validation

from sklearn import linear_model

from sklearn import preprocessing



from IPython.display import display



in_file = 'G:/onedrive/document/study/Udaicity/kaggle/data/train.csv'

full_data = pd.read_csv(in_file)



display(full_data.head())



#variable selection



housing_features.shape

selector=SelectPercentile(f_regression)

selector.fit_transform(housing_features,housing_prices)

score=-np.log10(selector.pvalues_)

all_score=np.random.rand(2,len(score))

all_score[0][:]=-np.log10(selector.pvalues_)

all_score[1][:]=np.arange(0,len(score),1)

all_score_trans=all_score.transpose()

E_avg_rec=1000000.0

bestpoly=0

bestalpha=0.0

E_avg_in=0

alpha_power_list = np.arange(-5,5,0.01)

alpha_select_list=[math.exp(alpha_select_list) for alpha_select_list in alpha_power_list]

for featuren in range(1,14):

    #n feature want to keep

    #featuren=13



    housing_features_new=SelectKBest(f_regression,k=featuren).fit_transform(housing_features,housing_prices)



    #Regression Model with Ridge Regularization

    #10-fold Cross Validation Involved

    #selection of Polynomial

    #selection of regularization alpha



    foldern=10

    polyn=0

    for maxpoly in range(1,4):

        if math.factorial(maxpoly+featuren-1)/(math.factorial(maxpoly)*math.factorial(featuren-1)) >= len(housing_prices)/2 and math.factorial(maxpoly+featuren-2)/(math.factorial(maxpoly-1)*math.factorial(featuren-1)) <= len(housing_prices)/2:

            polyn=maxpoly-1

    if polyn==0:

        polyn=3

    E_out=np.zeros((polyn,len(alpha_select_list),foldern),dtype=float)

    E_in =np.zeros((polyn,len(alpha_select_list),foldern),dtype=float)

    for k in range(1,polyn+1):

        if k == 1:

            housing_features_poly=housing_features_new

        else:

            poly=preprocessing.PolynomialFeatures(k)

            housing_features_poly=poly.fit_transform(housing_features_new)

        kf=cross_validation.KFold(len(housing_features_poly), n_folds=foldern)

        i=0

        for alpha_select in alpha_select_list:

            j=0

            for train, test in kf:

                X_train, X_test, Y_train, Y_test = housing_features_poly[train], housing_features_poly[test], housing_prices[train], housing_prices[test]

                #Fitting use Ridge regression model

                clf = linear_model.Ridge (alpha = alpha_select)

                clf.fit(X_train, Y_train)

                E_in[k-1][i][j] =np.dot((Y_train-clf.predict(X_train)),(Y_train-clf.predict(X_train)).transpose())/len(Y_train)

                E_out[k-1][i][j]=np.dot((Y_test-clf.predict(X_test)),(Y_test-clf.predict(X_test)).transpose())/len(Y_test)

                j+=1

            #searching for best model

            #print(k,",",i,",:",np.mean(E_out[k-1][i][:]))

            print("Feature number:",featuren,", Polynomial:",k,", Alpha:",alpha_select,", E_out:",np.mean(E_out[k-1][i][:]))

            if E_avg_rec > np.mean(E_out[k-1][i][:]):

                bestfeaturen=featuren

                bestpoly = k

                bestalpha = alpha_select

                E_avg_rec = np.mean(E_out[k-1][i][:])

                E_avg_in  = np.mean(E_in[k-1][i][:])

            i+=1



print("Feature number:",bestfeaturen,", Best Polynomial:",bestpoly,", Best Alpha:",bestalpha,", E_out:",E_avg_rec,", E_in:",E_avg_in)



#fit the selected model with full data

best_model = linear_model.Ridge (alpha = bestalpha)

poly=preprocessing.PolynomialFeatures(bestpoly)

housing_features_new=SelectKBest(f_regression,k=bestfeaturen).fit_transform(housing_features,housing_prices)

housing_features_poly=poly.fit_transform(housing_features_new)

best_model.fit(housing_features_poly, housing_prices)

best_model.score(housing_features_poly, housing_prices)

print(best_model.score(housing_features_poly, housing_prices))

E_in_best=np.dot((housing_prices-best_model.predict(housing_features_poly)),(housing_prices-best_model.predict(housing_features_poly)).transpose())/len(housing_prices)

print(E_in_best)





#Process the Prediction Data

best_feature_index=list(all_score_trans[all_score_trans[:,0].argsort()][:][(len(all_score_trans)-bestfeaturen):][:,1].astype(int))

best_client_feature=np.array(CLIENT_FEATURES[0])[sorted(best_feature_index)]

print(best_client_feature)

best_client_feature_poly=poly.fit_transform(best_client_feature)

print(best_client_feature_poly)



#Prediction

best_model.predict(best_client_feature_poly)









