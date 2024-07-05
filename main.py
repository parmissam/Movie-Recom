import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)

path_to_udata = r"D:\python projects\project LR\ml-100k\u.data"
path_to_usersdata = r"D:\python projects\project LR\ml-100k\u.user"
path_to_uitem = r"D:\python projects\project LR\ml-100k\u.item"

# Load ratings data
ratings_data = pd.read_csv(path_to_udata, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Load users data
users_data = pd.read_csv(path_to_usersdata, sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

# Load movies data with Latin-1 encoding
movies_data = pd.read_csv(path_to_uitem, sep='|', names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], encoding='latin-1')

# Merge ratings and users data
merged_data = pd.merge(ratings_data, users_data, on='user_id')

# Merge with movies data
merged_data = pd.merge(merged_data, movies_data, left_on='item_id', right_on='movie_id')

# Create user-item interaction matrix
user_item_matrix = merged_data.pivot(index='user_id', columns='movie_id', values='rating')

user_features = merged_data[['age', 'gender', 'occupation']]  # User features
item_features = merged_data[['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
                             'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                             'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
# Optionally include other item features like release_date, genre-specific data, etc.
X_train = pd.concat([user_features, item_features], axis=1)

# Target (y_train)
y_train = merged_data['rating']
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")