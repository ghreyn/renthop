#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:19:26 2017

@author: reynold
"""

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

gym_substrings = ['gym', 'fitness', 'health club']
ac_substrings = ['air', 'air conditioning', 'a/c']

featmap = [(gym_substrings, 'gym'),
           (ac_substrings, 'ac'), 
           (['hardwood'], 'hardwood'), 
           (['in unit'], ""),
           (['swimming pool'], 'pool'),
           (['in unit'], ''),
           (['valet'], 'valet services'),
           (['high ceil'], 'high c'),
           (['ss'], 'ss'),
           (['walk in'], 'walk in'),
           (['pets '], 'pets')
            ]

def has_long_feature(features):
    """Some posters include list of features/description in a single field
    """
    for f in features:
        if len(f) > 70:
            return 1
    return 0

def total_feat_chars(features):
    """
    """
    num_chars = 0
    for f in features:
        num_chars += len(f)
    return num_chars

def split_large_feature(x):
    """Handle features provided as one long string rather than a list.
    """
    if len(x) == 1:
        if len(x[0]) > 70:
            return list(x[0].replace("**", "").replace(" * ", "*").replace(" ~ ","*").split("*"))

    return x

def map_common_terms(s):
    """
    """
    # for inconsistent hyphenation
    x = s.replace("-", " ")
 
    # vectorizer will stop at punctuation
    x = x.replace(".", "")
    x = x.replace("'", "")
   
        
    x = x.replace("prewar", "pre war")    
    x = x.replace("twenty four hour", "24")
    x = x.replace("24/7", "24")
    x = x.replace("24hr", "24")
    x = x.replace("24 hr", "24")
    x = x.replace("24-hour", "24")
    x = x.replace("24hour", "24")
    x = x.replace("24 hour", "24")
    x = x.replace("on site ", "")
    x = x.replace("live in ", "")
    x = x.replace("full time ", "")
    x = x.replace("common ", "")    
    x = x.replace("building", "bldg")
    x = x.replace(" in bldg", "")
    x = x.replace("bicycle", "bike")   
    x = x.replace("superintendent", "super")   
    x = x.replace("stainless steel", "ss")
    x = x.replace("s/s", "ss")
    x = x.replace("central ", "")
    x = x.replace("/", "")
    x = x.replace(" & ", "")

    return x

def map_common_feature(feature):
    """
    """
    # use consistent terminology, abbreviations
    feature = map_common_terms(feature)
    
    # try to map to a common feature
    for tple in featmap:
        for s in tple[0]:
            if feature.find(s) != -1:
                return feature if tple[1] == '' else tple[1]

    return feature


def map_feat_to_common(flist):
    """
    """
    newlist = []
    for s in flist:
        newlist.append(map_common_feature(s))
    return newlist


def clean_street(address):
    """
    """
    x = address.strip().lower().replace(".","")

    x = x.replace("first","1st").replace("second", "2nd").replace("third", "3rd").\
        replace("fourth", "4th").replace("fifth", "5th").replace("sixth", "6th").\
        replace("seventh", "7th").replace("eighth", "8th").replace("ninth", "9th").replace("tenth", "10th")

    x = x.replace("street", "st").replace("avenue", "ave").replace("road", "rd").\
        replace("boulevard", "blvd")

    return x

def preprocess(dfdata, istest=False):
    """ Initial data cleaning
    """
    droplist = ['created', 'photos']

    # encode day of month, day of week, and hour of day that listing was created
    dfdata['created'] = pd.to_datetime(dfdata['created'])
    dfdata['created_dayofweek'] = dfdata['created'].dt.weekday
    dfdata['created_hour'] = dfdata['created'].dt.hour
    dfdata['created_day'] = dfdata['created'].dt.day 
    
    # also encode months also since data spans multiple months
    dfdata['created_month'] = dfdata['created'].dt.month        

    # create feats for number of listing photos and features, length of description
    dfdata['num_photos'] = dfdata['photos'].apply(lambda x: len(x))
    dfdata['num_features'] = dfdata['features'].apply(lambda x: len(x))
    dfdata['desc_len'] = dfdata['description'].apply(len)

    # created features   
    dfdata['num_bdba'] = dfdata['bedrooms'] + dfdata['bathrooms']

    # clean feature list
    dfdata['features'] = dfdata['features'].apply(lambda x: list([i.lower() for i in x]))
    dfdata['features'] = dfdata['features'].apply(lambda x: split_large_feature(x))
    dfdata['features'] = dfdata['features'].apply(lambda x: map_feat_to_common(x))

    # clean street and display addresses
    dfdata['street_address'] = dfdata['street_address'].apply(lambda x: clean_street(x))
    dfdata['display_address'] = dfdata['display_address'].apply(lambda x: clean_street(x))
    
    # strip tags from description
    dfdata['description'] = dfdata['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text().lower())
   
    # if longitude or latitude, not provided, set to median values
    # note: if either is zero, the other is zero in both training and test data
    dfdata['latitude'] = dfdata['latitude'].replace({0.0: np.nan})
    dfdata['longitude'] = dfdata['longitude'].replace({0.0: np.nan})

    dfdata["longitude"] = dfdata["longitude"].fillna(dfdata["longitude"].median())
    dfdata["latitude"] = dfdata["latitude"].fillna(dfdata["latitude"].median())

    dfdata['lat_5digit'] =dfdata['latitude'].apply(lambda x: round(x,5))
    dfdata['long_5digit'] =dfdata['longitude'].apply(lambda x: round(x,5))

    # adjust bathroom outliers found in EDA; assume typos
    if istest == True:
        dfdata['bathrooms'].ix[dfdata['bathrooms'] == 112] = 1.5
        dfdata['bathrooms'].ix[dfdata['bathrooms'] == 20] = 2.0
    else:
        dfdata['bathrooms'].ix[dfdata['bathrooms'] == 10] = 1.0
    
    image_date = pd.read_csv("data/listing_image_time.csv")

    # rename columns so you can join tables later on
    image_date.columns = ["listing_id", "time_stamp"]

    # reassign the only one timestamp from April, to beginning of Nov,
    # since all others from Oct/Nov
    image_date.loc[80240,"time_stamp"] = 1478129766 

    image_date["img_date"]                  = pd.to_datetime(image_date["time_stamp"], unit="s")
    image_date["img_days_passed"]           = (image_date["img_date"].max() - image_date["img_date"]).astype("timedelta64[D]").astype(int)
    image_date["img_date_month"]            = image_date["img_date"].dt.month
    image_date["img_date_week"]             = image_date["img_date"].dt.week
    image_date["img_date_day"]              = image_date["img_date"].dt.day
    image_date["img_date_dayofweek"]        = image_date["img_date"].dt.dayofweek
    image_date["img_date_dayofyear"]        = image_date["img_date"].dt.dayofyear
    image_date["img_date_hour"]             = image_date["img_date"].dt.hour
    image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)

    image_date.drop(['img_date'], axis=1, inplace=True)
    dfdata = pd.merge(dfdata, image_date, on="listing_id", how="left")    
    
    # drop feats we transformed or won't using
    dfdata.drop(droplist, axis=1, inplace=True)    

    return dfdata
