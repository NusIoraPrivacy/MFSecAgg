import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
# import sys
# sys.path.append('/home/nusbac/miniconda3/envs/my-170-env/lib/python3.12/site-packages')
# from utils.globals import *
import utils

def standard_id(item_df, user_df, rating_df):
    userIDs = user_df.UserID.unique()
    itemIDs = item_df.ItemID.unique()

    userid2encode = {}
    for i, userid in enumerate(userIDs):
        userid2encode[userid] = i

    itemid2encode = {}
    for i, itemid in enumerate(itemIDs):
        itemid2encode[itemid] = i

    # rating_df['UserID'] = rating_df['UserID'].apply(lambda x: userid2encode[x] if x in userid2encode.keys() else x)
    # rating_df['ItemID'] = rating_df['ItemID'].apply(lambda x: itemid2encode[x] if x in itemid2encode.keys() else x)
    rating_df['UserID'] = rating_df['UserID'].apply(lambda x: userid2encode[x])
    rating_df['ItemID'] = rating_df['ItemID'].apply(lambda x: itemid2encode[x])
    # rating_df = rating_df[rating_df['UserID'].isin(userid2encode.values())]
    # rating_df = rating_df[rating_df['ItemID'].isin(itemid2encode.values())]

    item_df['ItemID'] = item_df['ItemID'].apply(lambda x: itemid2encode[x])
    user_df['UserID'] = user_df['UserID'].apply(lambda x: userid2encode[x])

    return item_df, user_df, rating_df

def get_rating_list(rating_df, args, item_id_list=None):
    ratings = rating_df.values
    user_id_list = rating_df.UserID.unique()
    
    ratings_dict = {e:[] for e in user_id_list}
    counter = 0
    for record in ratings:
        rating_feat_list = []
        for r in record[1:]:
            rating_feat_list.append(int(r))
        ratings_dict[record[0]].append(tuple(rating_feat_list))
        # ratings_dict[record[0]].append([int(record[1]), float(record[2])])
        counter += 1
    return ratings_dict

def get_feature_list(feat_df):
    feat_array = feat_df.values
    feat_dict = {}
    for feat_vec in feat_array:
        feat_id = int(feat_vec[0])
        feats = [int(e) for e in feat_vec[1:]]
        feat_dict[feat_id] = feats
    return feat_dict

def train_test_split(ratings_dict, args):
    train_data = {}
    test_data = {}
    for user_id in ratings_dict:
        rating_list = ratings_dict[user_id]
        random.shuffle(rating_list)
        test_num = int(len(rating_list) * args.test_pct)
        if test_num == 0:
            train_data[user_id] = rating_list
            test_data[user_id] = []
            # print(user_id, len(rating_list))
        else:
            train_data[user_id] = rating_list[:-test_num]
            test_data[user_id] = rating_list[-test_num:]

    return train_data, test_data

def train_test_split_neg(ratings_dict, args, item_dict, user_dict, item_id_list, train=False):
    train_data = {}
    test_data = {}
    for user_id in tqdm(ratings_dict):
        rating_list = ratings_dict[user_id]
        random.shuffle(rating_list)
        test_num = int(len(rating_list) * args.test_pct)
        if test_num == 0:
            train_data[user_id] = rating_list
            test_data[user_id] = []
            # print(user_id, len(rating_list))
        else:
            train_data[user_id] = rating_list[:-test_num]
            test_data[user_id] = rating_list[-test_num:]
        # sample negative items for test data
        rated_items = [rate[0] for rate in rating_list]
        unrated_items = list(set(item_id_list) - set(rated_items))
        random.shuffle(unrated_items)
        sample_neg_items = unrated_items[:100]
        for item_id in sample_neg_items:
            rating_vec = [item_id, 0]
            rating_vec += item_dict[item_id]
            rating_vec += user_dict[user_id]
            test_data[user_id].append(tuple(rating_vec))
        # print("Before:", len(train_data[user_id]))
        train_len = len(train_data[user_id])
        train_len = int(train_len * args.neg_ratio)
        if train:
            sample_neg_items = unrated_items[100:(100+train_len)]
            for item_id in sample_neg_items:
                rating_vec = [item_id, 0]
                rating_vec += item_dict[item_id]
                rating_vec += user_dict[user_id]
                train_data[user_id].append(tuple(rating_vec))
        # print("After:", len(train_data[user_id]))
    return train_data, test_data

def fed2central(train_data):
    central_data = []
    for user_id in train_data:
        rating_list = train_data[user_id]
        for rating_vec in rating_list:
            rating_vec = [user_id] + list(rating_vec)
            central_data.append(rating_vec)
    return central_data

def sample_negative(all_items, ratings_dict, user):
    positive_items = set(ratings_dict[user])
    available_items = all_items - positive_items
    return random.choice(list(available_items))

def sample_item_central(train_data, args):
    random.shuffle(train_data)
    sample_train_data = []
    n_per_user = defaultdict(int)
    for rating_vector in train_data:
        user = rating_vector[0]
        if n_per_user[user] < args.n_sample_items:
            n_per_user[user] += 1
            sample_train_data.append(rating_vector)
    avg_n_per_u = sum(n_per_user.values()) / len(n_per_user) 
    return sample_train_data, avg_n_per_u

def train_test_split_central(ratings_df, args):
    ratings_df = ratings_df.sample(frac=1)
    n_test = int(args.test_pct * len(ratings_df))
    rating_mat = ratings_df.values
    rating_mat = rating_mat.tolist()
    train_data = rating_mat[:-n_test]
    test_data = rating_mat[-n_test:]
    return train_data, test_data

def genre_to_onehot(item_df, args):
    genre_array = item_df["Genres"].values
    # all_genre = []
    # for genre_list in genre_array:
    #     for genre in genre_list:
    #         if genre not in all_genre:
    #             all_genre.append(genre)
    # print(all_genre)
    drop_cols = [col for col in item_df.columns if col != "ItemID"]
    item_df.drop(drop_cols, axis=1, inplace=True)
    all_genres = all_genre_dict[args.dataset]
    genre_mat = np.zeros((len(item_df), len(all_genres)))
    for i, genre_list in enumerate(genre_array):
        for j, genre in enumerate(all_genres):
            if genre in genre_list:
                genre_mat[i,j] = 1
    for i, genre in enumerate(all_genres):
        item_df[genre] = genre_mat[:, i]
    return item_df

def process_user_df(user_df, args):
    if args.dataset in ["ml-1m", "ml-100k"]:
        gender_one_hot = pd.get_dummies(user_df["Gender"], prefix="gender", dtype=int)
        age_one_hot = pd.get_dummies(user_df["Age"], prefix="age", dtype=int)
        occupation_one_hot = pd.get_dummies(user_df["Occupation"], prefix="occ", dtype=int)
        drop_cols = [col for col in user_df.columns if col != "UserID"]
        user_df.drop(drop_cols, axis=1, inplace=True)
        for df in [gender_one_hot, age_one_hot, occupation_one_hot]:
            for col in df.columns:
                user_df[col] = df[col]

    elif args.dataset == "bookcrossing":
        user_df["Country"] = user_df["Location"].apply(lambda x: (x.split(",")[-1]).strip().strip('"').strip("\\"))
        user_df["Country"] = user_df["Country"].apply(lambda x: replace_loc[x] if x in replace_loc.keys() else x)
        user_df["Country"] = user_df["Country"].apply(lambda x: "na" if x not in common_countries else x)
        country_one_hot = pd.get_dummies(user_df["Country"], prefix="ctr", dtype=int)
        # age_one_hot = pd.get_dummies(user_df["Age"], prefix="ctr", dtype=int)
        drop_cols = [col for col in user_df.columns if col != "UserID"]
        user_df.drop(drop_cols, axis=1, inplace=True)
        # for df in [country_one_hot, age_one_hot]:
        #     for col in df.columns:
        #         user_df[col] = df[col]
        user_df = pd.concat([user_df, country_one_hot], axis=1)
    return user_df

def process_item_df(item_df, args):
    if args.dataset == "bookcrossing":
        pub_year_one_hot = pd.get_dummies(item_df["Year-Of-Publication"], prefix="year", dtype=int)
        drop_cols = [col for col in item_df.columns if col != "ItemID"]
        item_df.drop(drop_cols, axis=1, inplace=True)
        # for df in [pub_year_one_hot]:
        #     for col in df.columns:
        #         item_df[col] = df[col]
        item_df = pd.concat([item_df, pub_year_one_hot], axis=1)
        return item_df
    elif args.dataset == "yelp":
        item_df["Categories"] = item_df["Categories"].apply(lambda x: x.split(", ") if isinstance(x, str) else [])
        categories_values = item_df["Categories"].tolist()
        all_categories = []
        for categories in categories_values:
            for cat in categories:
                if cat not in all_categories:
                    all_categories.append(cat)
        # print(len(all_categories))
        # print(all_categories)
        category_mat = np.zeros((len(item_df), len(all_categories)))
        for i, cat_list in enumerate(categories_values):
            for j, cat in enumerate(all_categories):
                if cat in cat_list:
                    category_mat[i,j] = 1
        all_categories = ["cat_"+str(s) for s in all_categories]
        category_df = pd.DataFrame(category_mat, columns=all_categories, dtype=int)
        category_df.index = item_df.index
        state_df = pd.get_dummies(item_df["State"], prefix="state", dtype=int)
        drop_cols = [col for col in item_df.columns if col != "ItemID"]
        item_df.drop(drop_cols, axis=1, inplace=True)
        # print(item_df.shape, state_df.shape, category_df.shape)
        # item_df = pd.concat([item_df, state_df, category_df], axis=1)
        item_df = pd.concat([item_df, state_df], axis=1)
        return item_df

def load_data(args):
    data_path = f"{args.root_path}/data/{args.dataset}"
    if args.dataset == "ml-1m":
        rating_path = f"{data_path}/ratings.dat"
        rating_df = pd.read_csv(rating_path, delimiter='::', header=None, names=["UserID", "ItemID", "Rating", "Timestamp"])
        item_path = f"{data_path}/movies.dat"
        item_df = pd.read_csv(item_path, delimiter='::', header=None, names=["ItemID", "Title", "Genres"], encoding="iso-8859-1")
        # print(item_df.head())
        user_path = f"{data_path}/users.dat"
        user_df = pd.read_csv(user_path, delimiter='::', header=None, names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
        item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        item_df["Genres"] = item_df["Genres"].apply(lambda x: x.split("|"))
        item_df = genre_to_onehot(item_df, args)
        user_df = process_user_df(user_df, args)
        # rating_per_user = rating_df.groupby("UserID")["Rating"].count()
        # print(len(item_df))
        # print("max item features:", item_df.drop("ItemID", axis=1).values.sum(axis=1).max())
        # print("average item features:", item_df.drop("ItemID", axis=1).values.sum(axis=1).mean())
        # print("max user features:", user_df.drop("UserID", axis=1).values.sum(axis=1).max())
        # print("average user features:", user_df.drop("UserID", axis=1).values.sum(axis=1).mean())
        combine_df = rating_df.merge(item_df, on="ItemID", how='left')
        combine_df = combine_df.merge(user_df, on="UserID", how='left')
        combine_df.drop("Timestamp", axis=1, inplace=True)
        # print(combine_df.shape)
        return item_df, user_df, combine_df
    
    if args.dataset == "ml-100k":
        rating_path = f"{data_path}/u.data"
        rating_df = pd.read_csv(rating_path, delimiter='\t', header=None, names=["UserID", "ItemID", "Rating", "Timestamp"])
        item_path = f"{data_path}/u.item"
        item_df = pd.read_csv(item_path, delimiter='|', header=None, names=["ItemID", "Title", "release date", "video release date",
                    "IMDb URL", "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", 
                    "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], encoding="iso-8859-1")
        user_path = f"{data_path}/u.user"
        user_df = pd.read_csv(user_path, delimiter='|', header=None, names=["UserID", "Age", "Gender", "Occupation", "Zip-code"])
        rated_user_ids = set(rating_df.UserID.unique()).intersection(set(user_df.UserID.unique()))
        rated_item_ids = set(rating_df.ItemID.unique()).intersection(set(item_df.ItemID.unique()))
        rating_df = rating_df[rating_df['UserID'].isin(rated_user_ids)]
        rating_df = rating_df[rating_df['ItemID'].isin(rated_item_ids)]
        user_df = user_df[user_df["UserID"].isin(rated_user_ids)]
        item_df = item_df[item_df['ItemID'].isin(rated_item_ids)]
        item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        item_df = item_df.drop(["Title", "release date", "video release date", "IMDb URL"], axis=1)
        user_df = process_user_df(user_df, args)
        # print("max item features:", item_df.drop("ItemID", axis=1).values.sum(axis=1).max())
        # print("average item features:", item_df.drop("ItemID", axis=1).values.sum(axis=1).mean())
        # print("max user features:", user_df.drop("UserID", axis=1).values.sum(axis=1).max())
        # print("average user features:", user_df.drop("UserID", axis=1).values.sum(axis=1).mean())
        combine_df = rating_df.merge(item_df, on="ItemID", how='left')
        combine_df = combine_df.merge(user_df, on="UserID", how='left')
        combine_df.drop("Timestamp", axis=1, inplace=True)
        # print(item_df.head())
        # print(user_df.head())
        # avg_user = rating_df.groupby("UserID")["Rating"].count()
        # print(avg_user.mean())
        # print(combine_df.shape)
        return item_df, user_df, combine_df

    if args.dataset == "ml-10m":
        rating_path = f"{data_path}/ratings.dat"
        rating_df = pd.read_csv(rating_path, delimiter='::', header=None, names=["UserID", "ItemID", "Rating", "Timestamp"])
        # rating_df.drop("Timestamp", axis=1, inplace=True)
        # rating_df.columns = ["user", "item", "rating"]
        # rating_df.to_csv(f"{data_path}/ratings.csv", index=False)
        item_path = f"{data_path}/movies.dat"
        item_df = pd.read_csv(item_path, delimiter='::', header=None, names=["ItemID", "Title", "Genres"], encoding="iso-8859-1")
        item_df["Genres"] = item_df["Genres"].apply(lambda x: x.split("|"))
        item_df = genre_to_onehot(item_df, args)
        unique_user_ids = rating_df.UserID.unique()
        user_df = pd.DataFrame(data={"UserID": unique_user_ids})
        item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        rating_per_user = rating_df.groupby("UserID")["Rating"].count()
        # print("max item features:", item_df.drop("ItemID", axis=1).values.sum(axis=1).max())
        # print("average item features:", item_df.drop("ItemID", axis=1).values.sum(axis=1).mean())
        combine_df = rating_df.merge(item_df, on="ItemID", how='left')
        combine_df.drop("Timestamp", axis=1, inplace=True)
        # print(combine_df.shape)
        return item_df, user_df, combine_df
    
    if args.dataset == "ml-20m":
        rating_path = f"{data_path}/ratings.csv"
        rating_df = pd.read_csv(rating_path)
        rating_df.columns = ["UserID", "ItemID", "Rating", "Timestamp"]
        item_path = f"{data_path}/movies.csv"
        item_df = pd.read_csv(item_path, encoding="iso-8859-1")
        item_df.columns = ["ItemID", "Title", "Genres"]
        item_df["Genres"] = item_df["Genres"].apply(lambda x: x.split("|"))
        item_df = genre_to_onehot(item_df, args)
        # user_path = f"{data_path}/users.dat"
        # user_df = pd.read_csv(user_path, delimiter='::', header=None, names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
        # item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        unique_user_ids = rating_df.UserID.unique()
        user_df = pd.DataFrame(data={"UserID": unique_user_ids})
        item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        combine_df = rating_df.merge(item_df, on="ItemID", how='left')
        combine_df.drop("Timestamp", axis=1, inplace=True)
        # print(combine_df.head())
        return item_df, user_df, combine_df
    
    if args.dataset == "yelp":
        rating_path = f"{data_path}/yelp_academic_dataset_review.csv"
        rating_df = pd.read_csv(rating_path)
        rating_df.columns = ["UserID", "ItemID", "Rating"]
        # print(rating_df["Rating"].min(), rating_df["Rating"].max())
        top_users = rating_df.groupby('UserID')['Rating'].count()
        top_users = top_users.sort_values(ascending=False)[:10000].index
        rating_df = rating_df[rating_df['UserID'].isin(top_users)]
        item_path = f"{data_path}/yelp_academic_dataset_business.csv"
        item_df = pd.read_csv(item_path)
        item_df.columns = ["ItemID", "State", "is_open", "Categories"]
        rated_item_ids = set(rating_df.ItemID.unique()).intersection(set(item_df.ItemID.unique()))
        rating_df = rating_df[rating_df['ItemID'].isin(rated_item_ids)]
        item_df = item_df[item_df['ItemID'].isin(rated_item_ids)]
        unique_user_ids = rating_df.UserID.unique()
        user_df = pd.DataFrame(data={"UserID": unique_user_ids})
        item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        # rating_df.columns = ["user", "item", "rating"]
        # rating_df.to_csv(f"{data_path}/full.csv", index=False)
        item_df = process_item_df(item_df, args)
        # print("max item features:", item_df.drop("ItemID", axis=1).values.sum(axis=1).max())
        # print("average item features:", item_df.drop("ItemID", axis=1).values.sum(axis=1).mean())
        avg_user = rating_df.groupby("UserID")["Rating"].count()
        combine_df = rating_df.merge(item_df, on="ItemID", how='left')
        # print(combine_df[combine_df["cat_Trade Fairs"].isna()])
        avg_rating = combine_df["Rating"].mean()
        base_rmse = np.sqrt(((combine_df["Rating"] - avg_rating) ** 2).mean())
        print("baseline rmse:", base_rmse)
        # print(combine_df.shape)
        return item_df, user_df, combine_df

    if args.dataset == "ml-25m":
        rating_path = f"{data_path}/ratings.csv"
        rating_df = pd.read_csv(rating_path)
        rating_df.columns = ["UserID", "ItemID", "Rating", "Timestamp"]
        # print(rating_df["Rating"].min(), rating_df["Rating"].max())
        item_path = f"{data_path}/movies.csv"
        item_df = pd.read_csv(item_path, encoding="iso-8859-1")
        item_df.columns = ["ItemID", "Title", "Genres"]
        item_df["Genres"] = item_df["Genres"].apply(lambda x: x.split("|"))
        item_df = genre_to_onehot(item_df, args)
        # print("max item features:", item_df.drop("ItemID", axis=1).values.sum(axis=1).max())
        # print("average item features:", item_df.drop("ItemID", axis=1).values.sum(axis=1).mean())
        unique_user_ids = rating_df.UserID.unique()
        user_df = pd.DataFrame(data={"UserID": unique_user_ids})
        item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        combine_df = rating_df.merge(item_df, on="ItemID", how='left')
        combine_df.drop("Timestamp", axis=1, inplace=True)
        # print(combine_df.isna().sum())
        # print(combine_df.shape)
        return item_df, user_df, combine_df
    
    if args.dataset == "bookcrossing":
        rating_path = f"{data_path}/BX-Book-Ratings.csv"
        rating_df = pd.read_csv(rating_path, delimiter=';', encoding="iso-8859-1")
        rating_df.columns = ["UserID", "ItemID", "Rating"]
        # top_items = rating_df.groupby('ItemID')['Rating'].count()
        # item_thd, user_thd = rating_thds["bookcrossing"]
        # top_items = top_items[top_items >= item_thd]
        # top_items = top_items.index
        # rating_df = rating_df[rating_df['ItemID'].isin(top_items)]
        # top_users = rating_df.groupby('UserID')['Rating'].count()
        # top_users = top_users[top_users >= user_thd]
        # print(top_users.mean())
        # top_users = top_users.index
        top_users = rating_df.groupby('UserID')['Rating'].count()
        top_users = top_users.sort_values(ascending=False)[:6000].index
        rating_df = rating_df[rating_df['UserID'].isin(top_users)]
        top_items = rating_df.groupby('ItemID')['Rating'].count()
        top_items = top_items.sort_values(ascending=False)[:3000].index
        rating_df = rating_df[rating_df['ItemID'].isin(top_items)]
        avg_users = rating_df.groupby('UserID')['Rating'].count()
        print(avg_users.mean())
        # top_users = np.random.choice(top_users, 6000, replace=False)
        
        item_path = f"{data_path}/BX_Books.csv"
        item_df = pd.read_csv(item_path, delimiter=';', encoding="iso-8859-1")
        item_df.columns = ["ItemID", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L"]
        user_path = f"{data_path}/BX-Users.csv"
        user_df = pd.read_csv(user_path, delimiter=';', encoding="iso-8859-1")
        user_df.columns = ["UserID", "Location", "Age"]
        # filter records
        rated_user_ids = set(rating_df.UserID.unique()).intersection(set(user_df.UserID.unique()))
        rated_item_ids = set(rating_df.ItemID.unique()).intersection(set(item_df.ItemID.unique()))
        rating_df = rating_df[rating_df['UserID'].isin(rated_user_ids)]
        rating_df = rating_df[rating_df['ItemID'].isin(rated_item_ids)]
        user_df = user_df[user_df["UserID"].isin(rated_user_ids)]
        item_df = item_df[item_df['ItemID'].isin(rated_item_ids)]
        item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        user_df = process_user_df(user_df, args) 
        item_df = process_item_df(item_df, args)
        print(item_df.shape)
        print(user_df.shape)
        # print(filter_user_df.shape)
        # print(filter_item_df.shape)
        combine_df = rating_df.merge(item_df, on="ItemID", how='left')
        combine_df = combine_df.merge(user_df, on="UserID", how='left')
        avg_rating = combine_df["Rating"].mean()
        base_rmse = np.sqrt(((combine_df["Rating"] - avg_rating) ** 2).mean())
        print("baseline rmse:", base_rmse)
        return item_df, user_df, combine_df
