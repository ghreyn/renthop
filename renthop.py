#
# RentHop Rental Listings Interest Competition
#

import numpy as np
import pandas as pd
import datetime
import cleaning
import rating

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
import xgboost as xgb

import sys

TRAIN_JSON = "data/train.json"
TEST_JSON = "data/test.json"
TRAIN_FILE ="data/train.csv"
TEST_FILE ="data/test.csv"
LOG_FILE="xgbresults.txt"

VAL_SIZE=0.10   # validation set size
NUM_FOLDS=5
SEED=2017
NUM_ROUNDS=175


interest_map = {'low':0, 'medium':1, 'high':2}


def write_csv_data(dfdata, filename):
    """ Write dataframe to CSV and log the operation
    """
    now = datetime.datetime.now()
    
    with open("data/csvfiles.txt", "a") as myfile:
        print("\n{0}".format(now), file=myfile)
        print("Writing {}".format(filename), file=myfile)
        print(list(dfdata.columns), file=myfile)
    
    print("\nData shape: {}".format(dfdata.shape))
    dfdata.to_csv(filename, index=False)
    print("Writing {}".format(filename))
    return

def clean_data():
    """ Clean the data and add features
    """

    dropfeat = ['features', 'description']

    print ("Reading training json...")
    train_df = pd.read_json(TRAIN_JSON)  
    
    print ("Reading test json...")
    test_df = pd.read_json(TEST_JSON)    

    # preprocessing that doesn't require both training and test data
    clean_train_df = cleaning.preprocess(train_df)
    clean_test_df = cleaning.preprocess(test_df, istest=True)

    # find median prices by number of bedrooms
    # add feature for the difference between price and median price    
    br_median_prices = clean_train_df.groupby(['bedrooms'])['price'].median().to_dict()
    
    clean_train_df['median_price_delta'] = clean_train_df.apply(lambda x: x.price - br_median_prices.get(x.bedrooms), axis=1)   
    clean_test_df['median_price_delta'] = clean_test_df.apply(lambda x: x.price - br_median_prices.get(x.bedrooms), axis=1)

    # add a manager rating feature
    clean_train_df, clean_test_df = rating.mgr_rating_pct(clean_train_df, clean_test_df)

    # add features for listing frequency of buildings
    bldgs_cnt = clean_train_df['building_id'].value_counts()
    
    p99_cnt = np.percentile(bldgs_cnt.values, 99)
    p95_cnt = np.percentile(bldgs_cnt.values, 95)
    p90_cnt = np.percentile(bldgs_cnt.values, 90)
    p80_cnt = np.percentile(bldgs_cnt.values, 80)
    p70_cnt = np.percentile(bldgs_cnt.values, 70)
    p60_cnt = np.percentile(bldgs_cnt.values, 60)
    p50_cnt = np.percentile(bldgs_cnt.values, 50)

    clean_train_df['top1pct_building'] = clean_train_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p99_cnt] else 0)    

    clean_train_df['top5pct_building'] = clean_train_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p95_cnt] else 0)    
    
    clean_train_df['top10pct_building'] = clean_train_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p90_cnt] else 0)    
    
    clean_train_df['top20pct_building'] = clean_train_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p80_cnt] else 0)    
  
    clean_train_df['top30pct_building'] = clean_train_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p70_cnt] else 0)     

    clean_train_df['top40pct_building'] = clean_train_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p60_cnt] else 0)

    clean_train_df['top50pct_building'] = clean_train_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p50_cnt] else 0)    
 
 
    clean_test_df['top1pct_building'] = clean_test_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p99_cnt] else 0)

    clean_test_df['top5pct_building'] = clean_test_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p95_cnt] else 0)
    
    clean_test_df['top10pct_building'] = clean_test_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p90_cnt] else 0)    
    
    clean_test_df['top20pct_building'] = clean_test_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p80_cnt] else 0)    
  
    clean_test_df['top30pct_building'] = clean_test_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p70_cnt] else 0)     

    clean_test_df['top40pct_building'] = clean_test_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p60_cnt] else 0)

    clean_test_df['top50pct_building'] = clean_test_df['building_id'].apply(lambda x: 1 if x in 
            bldgs_cnt.index.values[bldgs_cnt.values >= p50_cnt] else 0)      
    
    # label encoding for building_id and manager_id
    # requires fitting on values in both training and test data
    le = LabelEncoder()
    
    for f in ['building_id', 'manager_id', 'street_address', 'display_address']:
        le.fit(list(clean_train_df[f].values) + list(clean_test_df[f].values))
        clean_train_df[f] = le.transform(clean_train_df[f].values)
        clean_test_df[f] = le.transform(clean_test_df[f].values)

    # use clusters tas proxy for neighborhoods
    # fit on training, predict for test
    kmeans = KMeans(n_clusters=20, n_jobs=-1, random_state=0).fit(clean_train_df[['longitude','latitude']])
    clean_train_df['cluster'] = kmeans.labels_       
    
    clean_test_df['cluster'] = kmeans.predict(clean_test_df[['longitude','latitude']])

    # create tokens/counts for the most common training set features 
    # use the fitted vectorizer on the test data    
    clean_train_df['features'] = clean_train_df['features'].apply(lambda x: " ".join(["_".join(i.strip().lower().split(" ")) for i in x]))
    vectorizer = CountVectorizer(stop_words='english', max_features=200)
    
    tr_sparse_feat = vectorizer.fit_transform(clean_train_df['features'])
    cvfeatnames = np.asarray(vectorizer.get_feature_names())
    
    data = np.hstack([clean_train_df, tr_sparse_feat.toarray()])
    allcolumns = np.hstack([clean_train_df.columns, cvfeatnames])
    clean_train_df  = pd.DataFrame(data, columns=allcolumns)

    clean_test_df['features'] = clean_test_df["features"].apply(lambda x: " ".join(["_".join(i.strip().lower().split(" ")) for i in x]))    
    te_sparse_feat = vectorizer.transform(clean_test_df['features'])
    data = np.hstack([clean_test_df, te_sparse_feat.toarray()]) 
    allcolumns = np.hstack([clean_test_df.columns, cvfeatnames])
    clean_test_df  = pd.DataFrame(data, columns=allcolumns)

    # create tokens/counts for the most common training set description n-grams 
    # use the fitted vectorizer on the test data 
    vectorizer = CountVectorizer(ngram_range=(2,4), stop_words='english', max_features=25)
    tr_sparse_desc = vectorizer.fit_transform(clean_train_df['description'])

    cvdescnames = ['desc_' +i for i in vectorizer.get_feature_names()]
    data = np.hstack([clean_train_df, tr_sparse_desc.toarray()])
    allcolumns = np.hstack([clean_train_df.columns, cvdescnames])
    clean_train_df  = pd.DataFrame(data, columns=allcolumns)
    
    tr_sparse_desc = vectorizer.transform(clean_test_df['description'])
    data = np.hstack([clean_test_df, tr_sparse_desc.toarray()])
    allcolumns = np.hstack([clean_test_df.columns, cvdescnames])
    clean_test_df  = pd.DataFrame(data, columns=allcolumns)    
    
    clean_train_df.drop(dropfeat, axis=1, inplace=True)
    clean_test_df.drop(dropfeat, axis=1, inplace=True)

    # map target category values in training data to ints
    clean_train_df['interest_level'] = clean_train_df['interest_level'].apply(lambda x: interest_map[x])

    # write out the cleaned files for reuse
    write_csv_data(clean_train_df, TRAIN_FILE)
    write_csv_data(clean_test_df, TEST_FILE)   
    
    return


def run_xgb_fold(args, xtrain, ytrain, rounds, early_stop=20):
    """ Kfold cross validation
    """
    cv_scores = []
    
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    now = datetime.datetime.now()

    with open(LOG_FILE, "a") as myfile:
        print("\n{0} START".format(now), file=myfile)             
    
    for train_idx, val_idx in kf.split(xtrain, ytrain):
        print("Processing fold...")
        X_train, X_val = xtrain.iloc[train_idx], xtrain.iloc[val_idx]
        y_train, y_val = ytrain.iloc[train_idx], ytrain.iloc[val_idx]
    
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        xgval = xgb.DMatrix(X_val, label=y_val)
        
        # watch metrics on both training and validation sets
        # watchlist required for early stopping
        watchlist = [(xgtrain, 'train'), (xgval, 'val')]
        
        # train and predict with remaining fold
        model = xgb.train(args, xgtrain, rounds, watchlist, early_stopping_rounds=early_stop)
        predictions = model.predict(xgval)
        
        # keep scores of validation set, reduce precision for readability
        cv_scores.append(np.float32(log_loss(y_val,predictions)))
#        break
    
#    plot_xgb_importances(model)    

    scores_array = np.array(cv_scores)
        
    now = datetime.datetime.now()

    with open(LOG_FILE, "a") as myfile:
        print("{0} DONE".format(now), file=myfile)  
        print("Folds: {} Seed: {}".format(NUM_FOLDS, SEED), file=myfile)
        print("Rounds: {} Early stop: {} Num features: {} ".format(rounds, early_stop, len(list(xtrain.columns))), file=myfile)
        print(list(xtrain.columns), file=myfile)
        print(args, file=myfile)
        print("{} fold(s) processed".format(len(cv_scores)), file=myfile)
        print("Fold validation scores:", cv_scores, file=myfile)
        print("Mean: {:.5f} Std: {:.5f}".format(scores_array.mean(), scores_array.std()), file=myfile)
    
    print("\nFold validation scores:", cv_scores)
    print("Mean: {:.5f} Std: {:.5f}".format(scores_array.mean(), scores_array.std()))
    
    return

def run_xgb(args, xtrain, ytrain, xtest, ytest=None, rounds=1000):
    """ Train XGB model and generate predictions
    """
    
    xgtrain = xgb.DMatrix(xtrain, label=ytrain)
    xgval = xgb.DMatrix(xtest, ytest)

    print ("\nTraining and predicting...")
    model = xgb.train(args, xgtrain, rounds) 
    predictions = model.predict(xgval)   
    
    # score if target available
    if ytest is not None:
        score = np.float32(log_loss(ytest, predictions))
 
        print("Scoring holdout/validation set...")
        with open(LOG_FILE, "a") as myfile:
            print("Holdout loss: {}".format(score), file=myfile)        
    
        print("Holdout loss: {}".format(score))        
       
    return model, predictions


def write_submission_file(filename, ids, predictions):
    """
    """
    submission = pd.DataFrame(predictions)
    submission.columns = ['low', 'medium', 'high']
    submission['listing_id'] = ids
    submission.to_csv(filename, index=False)
    print("Wrote output file {0} {1}".format(filename, submission.shape))

    with open(LOG_FILE, "a") as myfile:
       print("wrote output file: {0} {1}".format(filename, submission.shape), file=myfile)     
    return



def plot_xgb_importances(model):
    """ Plot feature importances
    """
    importances = model.get_fscore()
    importance_df = pd.DataFrame({'Importance':list(importances.values()), 'Feature': list(importances.keys())})
    importance_df.sort_values(by = 'Importance', inplace=True)
    importance_df.plot(kind = 'barh', x= 'Feature', figsize=(10,25))    
    return

def main(existing_data, validate, predict, write_predictions):
    # Clean the json training and test data and write to csv
    
    if existing_data == False:
        clean_data()

    print("\nLoading training data...")
    train = pd.read_csv(TRAIN_FILE)
    print("Shape {}".format(train.shape))
        
    target = train['interest_level']

    print("\nLoading test data...")
    test = pd.read_csv(TEST_FILE)
    print("Shape {}".format(test.shape))

    # split out a hold out set
    print("\nCreating a holdout set...")
    X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(train.drop(['interest_level'], axis=1), target, test_size=VAL_SIZE, random_state=SEED)

    xgbargs = {
        'eval_metric': 'mlogloss', 
        'objective': 'multi:softprob',  # get probs for log_loss     
        'num_class': 3, # need this with above objective
#        'eta': 0.01,
        'colsample_bytree': 0.6,
        'min_child_weight': 6,
        'max_depth': 5,
        'subsample': 0.9,
        'alpha': 5.0,
#        'lambda': 1,
        'gamma': 0,
#        'silent': 1,    # default 0, silent mode when 1
        'seed': SEED
        }
   
    # Kfold cross validation
    run_xgb_fold(xgbargs, X_trainval, y_trainval, NUM_ROUNDS) 
    
    # evaluate holdout set
    if validate == True:
        model, preds = run_xgb(xgbargs, X_trainval, y_trainval, X_holdout, y_holdout, rounds=NUM_ROUNDS) 
        plot_xgb_importances(model) 

    # generate predictions for test data, use all training data
    if predict == True:
        model, preds = run_xgb(xgbargs, train.drop(['interest_level'], axis=1), target, test, rounds=NUM_ROUNDS) 
        
        if write_predictions == True:
#            write_submission_file("renthop.csv", test['listing_id'].values, preds)
            output_file = sys.argv[0].split(".")[0]+str(SEED)+".csv"
            write_submission_file(output_file, test['listing_id'].values, preds)


#    if (notify_completion):
#        send_sms.alertme(sys.argv[0])

    return

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preserve", help="use previously cleaned data", action="store_true")
    parser.add_argument("-s", "--score", help="score validation/holdout set", action="store_true")
    parser.add_argument("-t", "--test", help="generate test set predictions", action="store_true")
    parser.add_argument("-w", "--write", help="write predictions to csv", action="store_true")

    args = parser.parse_args()

    main(args.preserve, args.score, args.test, args.write)
