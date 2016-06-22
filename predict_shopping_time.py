# Importing a few necessary libraries

import numpy as np

import math  as math

import pandas as pd

from sklearn import linear_model

from sklearn import preprocessing



from IPython.display import display



in_file_train = 'G:/onedrive/document/python_projects/instacart/instacart-picking-time-challenge-data/train_trips.csv'

train_trip = pd.read_csv(in_file_train)



in_file_test = 'G:/onedrive/document/python_projects/instacart/instacart-picking-time-challenge-data/test_trips.csv'

test_trip = pd.read_csv(in_file_test)



in_file_items = 'G:/onedrive/document/python_projects/instacart/instacart-picking-time-challenge-data/order_items.csv'

order_items = pd.read_csv(in_file_items)



display(train_trip.head())

display(test_trip.head())

display(order_items.head())



# combine test and train data for preprocessing

train_trip['is_train'] = 1

test_trip['is_train'] = 0

full_data = train_trip

full_data = full_data.append(test_trip)



full_data.index = range(len(full_data))



# reformat columns

full_data['store_id'] = full_data['store_id'].apply(str)



# process time variable and create weekday and hour of day variable

full_data['start_dt'] = pd.to_datetime(full_data.shopping_started_at)

full_data['end_dt'] = pd.to_datetime(full_data.shopping_ended_at)

full_data['shopping_time'] = (full_data['end_dt'] - full_data['start_dt']).astype('timedelta64[s]')





def hr_func(ts):

    return ts.hour





full_data['start_hour'] = full_data['start_dt'].apply(hr_func)

full_data['start_weekday'] = full_data['start_dt'].apply(lambda x: x.weekday())



full_data['is_weekend'] = [1 if x >= 5 else 0 for x in full_data['start_weekday']]

full_data['is_morning'] = [1 if (x >= 6 and x < 11) else 0 for x in full_data['start_hour']]

full_data['is_noon'] = [1 if (x >= 11 and x < 13) else 0 for x in full_data['start_hour']]

full_data['is_afternoon'] = [1 if (x >= 13 and x < 18) else 0 for x in full_data['start_hour']]

full_data['is_night'] = [1 if (x >= 18 and x <= 24) else 0 for x in full_data['start_hour']]

full_data['is_midnight'] = [1 if x < 6 else 0 for x in full_data['start_hour']]



# generate dummies for stores

new_col1 = pd.get_dummies(full_data['store_id'], prefix='store')

new_col2 = pd.get_dummies(full_data['fulfillment_model'], prefix='model')

full_data_processed = pd.concat([full_data, new_col1], axis=1)

full_data_processed = pd.concat([full_data_processed, new_col2], axis=1)



# process items bought

# total item by trip

total_items = pd.DataFrame(order_items.groupby(['trip_id'])['quantity'].sum())

total_items['trip_id'] = total_items.index

full_data_processed = pd.merge(full_data_processed, total_items, on='trip_id')

# average item by shopper_id

average_items_by_shopper = pd.DataFrame(full_data_processed.groupby(['shopper_id'])['quantity'].mean())

average_items_by_shopper['shopper_id'] = average_items_by_shopper.index

average_items_by_shopper.rename(columns={'shopper_id': 'shopper_id', 'quantity': 'avg_quant'}, inplace=True)

full_data_processed = pd.merge(full_data_processed, average_items_by_shopper, on='shopper_id')



# average item by time to buy

average_items_by_time = pd.DataFrame(

    full_data_processed.groupby(['is_weekend', 'is_morning', 'is_noon', 'is_afternoon', 'is_night', 'is_midnight'],as_index=False)[

        'quantity'].mean())

average_items_by_time.rename(columns={'quantity': 'avg_time_quant'}, inplace=True)

full_data_processed = pd.merge(full_data_processed, average_items_by_time, how='left',

                               on=['is_weekend', 'is_morning', 'is_noon', 'is_afternoon', 'is_night', 'is_midnight'])



# process missing variables



imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)

full_data_imputed = pd.DataFrame(imp.fit_transform(full_data_processed[['trip_id', 'shopper_id', 'avg_quant']]),

                                 columns=['trip_id', 'shopper_id', 'avg_quant_nonull'])

full_data_processed = pd.merge(full_data_processed, full_data_imputed[['trip_id', 'avg_quant_nonull']], on='trip_id')



full_data_processed.drop(

    ['shopping_ended_at', 'shopping_started_at', 'quantity', 'avg_quant', 'start_hour', 'start_weekday', 'store_id',

     'fulfillment_model', 'start_dt', 'end_dt'],

    axis=1, inplace=True)



# split train and test data

full_data_train = full_data_processed[full_data_processed.is_train == 1]

full_data_test = full_data_processed[full_data_processed.is_train == 0]



# RANDOM SHUFFLE DATA

new_order = np.random.permutation(len(full_data_train))

full_data_train = full_data_train.iloc[new_order]



columns_to_run = [x for x in full_data_train.columns if x not in ('is_train', 'trip_id', 'shopper_id','shopping_time''')]

x_all = full_data_train[columns_to_run]

y_all = full_data_train['shopping_time']



# x_all.dtypes



# using ridge regression



from sklearn import grid_search

from sklearn.metrics import make_scorer, mean_squared_error

from sklearn import cross_validation

from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

from sklearn import cross_validation

from sklearn import linear_model



# tree_score= make_scorer(output_restric ,greater_is_better=False)



# variable selection



x_all.shape

selector = SelectPercentile(f_regression)

selector.fit_transform(x_all, y_all)

score = -np.log10(selector.pvalues_)

all_score = np.random.rand(2, len(score))

all_score[0][:] = -np.log10(selector.pvalues_)

all_score[1][:] = np.arange(0, len(score), 1)

all_score_trans = all_score.transpose()

bestpoly = 0

bestalpha = 0.0

alpha_power_list = np.arange(-3, 5, 0.1)

alpha_select_list = [math.exp(alpha_select_list) for alpha_select_list in alpha_power_list]

score_record = pd.DataFrame(columns=['feature_num', 'poly', 'alpha_value', 'e_out', 'e_in'])

for featuren in range(24, len(x_all.columns) + 1):

    # n feature want to keep

    # featuren=13



    full_data_processed_new = SelectKBest(f_regression, k=featuren).fit_transform(x_all, y_all)



    # Regression Model with Ridge Regularization

    # 10-fold Cross Validation Involved

    # selection of Polynomial

    # selection of regularization alpha



    polyn = 2

    for k in range(2, polyn + 1):

        if k == 1:

            full_data_processed_poly = full_data_processed_new

        else:

            poly = preprocessing.PolynomialFeatures(k)

            full_data_processed_poly = poly.fit_transform(full_data_processed_new)

        kf = cross_validation.KFold(len(full_data_processed_poly), n_folds=10)

        for alpha_select in alpha_select_list:

            E_in = 0.0

            E_out = 0.0

            for train, test in kf:

                X_train, X_test, Y_train, Y_test = full_data_processed_poly[train], full_data_processed_poly[

                    test], \

                                                   y_all.iloc[train], y_all.iloc[test]

                # Fitting use Ridge regression model

                clf = linear_model.Ridge(alpha=alpha_select, max_iter=1000)

                clf.fit(X_train, Y_train)

                E_in += mean_squared_error(Y_train, clf.predict(X_train))

                E_out += mean_squared_error(Y_test, clf.predict(X_test))

            score_record = score_record.append(

                pd.DataFrame([[featuren, k, alpha_select, E_out / 10.0, E_in / 10.0]],

                             columns=['feature_num', 'poly', 'alpha_value', 'e_out', 'e_in']))



print('finished modeling')



# TEST RUN DATA

# clf = linear_model.LogisticRegression(C=15)

# clf.fit(full_data_processed, target_combine)

# accuracy_score(target_combine, clf.predict(full_data_processed))



score_record[score_record.e_out == score_record.e_out.min()]

best_estimation = score_record[score_record.e_out == score_record.e_out.min()]



best_estimation

clf = linear_model.Ridge(alpha=0.548812, max_iter=1000)

full_data_processed_new = SelectKBest(f_regression, k=best_estimation.feature_num[0]).fit_transform(x_all, y_all)

poly = preprocessing.PolynomialFeatures(np.int(best_estimation.poly[0]))

full_data_processed_poly = poly.fit_transform(full_data_processed_new)



clf.fit(full_data_processed_poly, y_all)

mean_squared_error(y_all, clf.predict(full_data_processed_poly))



feature_select = SelectKBest(f_regression, k=best_estimation.feature_num[0]).fit(x_all, y_all)

test_data_processed_new = feature_select.transform(full_data_test[columns_to_run])

test_data_processed_poly = poly.fit_transform(test_data_processed_new)



# Prediction

predicted_value = pd.DataFrame(clf.predict(test_data_processed_poly), columns=['shopping_time'])

test_id = test_trip[['trip_id']].astype(int)

test_id.index = range(len(test_id))

predicted_value_pd = pd.concat([test_id, predicted_value], axis=1)



print('final finished')



#  OUTPUT TO CSV



predicted_value_pd.to_csv('G:/onedrive/document/python_projects/instacart/instacart-picking-time-challenge-data/predictions.csv', index=False)



print 'output finish'
