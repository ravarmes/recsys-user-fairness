import pandas as pd
import random
import RecSysALS
import RecSysKNN
import RecSysNMF

class UserFairness():
        
    def __init__(self, n_users, n_movies, top_users, l, theta, k):
        self.n_users = n_users
        self.n_movies = n_movies
        self.top_users = top_users
        self.l = l
        self.theta = theta
        self.k = k

    ###################################################################################################################
    # function to read the data
    def read_movieitems(self, n_movies, n_users, top_users, data_dir):
        # get ratings
        df = pd.read_table('{}/ratings.dat'.format(data_dir),names=['UserID','MovieID','Rating','Timestamp'], sep='::', engine='python')

        # create a dataframe with movie IDs on the rows and user IDs on the columns
        ratings = df.pivot(index='MovieID', columns='UserID', values='Rating')
        
        movies = pd.read_table('{}/movies.dat'.format(data_dir), names=['MovieID', 'Title', 'Genres'], sep='::', engine='python')
                            
        user_info = pd.read_table('{}/users.dat'.format(data_dir), names=['UserID','Gender','Age','Occupation','Zip-code'], sep='::', engine='python')
        user_info = user_info.rename(index=user_info['UserID'])[['Gender','Age','Occupation','Zip-code']]
        
        # put movie titles as index on rows
        movieSeries = pd.Series(list(movies['Title']), index=movies['MovieID'])
        ratings = ratings.rename(index=movieSeries)
        
        # read movie genres
        movie_genres = pd.Series(list(movies['Genres']),index=movies['Title'])
        movie_genres = movie_genres.apply(lambda s:s.split('|'))

        # select the top n_movies with the highest number of ratings
        num_ratings = (~ratings.isnull()).sum(axis=1) # quantitative ratings for each movie: Movie 1: 4, Movie 2: 5, Movie 3: 2 ...
        rows = num_ratings.nlargest(n_movies) # quantitative ratings for each movie (n_movies) sorted: Movie 7: 6, Movie 2: 5, Movie 1: 4 ...
        ratings = ratings.loc[rows.index] # matrix[n_movies rows , original columns]; before [original rows x original columns]
        
        if top_users:
            # select the top n_users with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=0) # quantitative ratings made by each user: User 1: 5, User 2: 5, User 3: 5, ...
            cols = num_ratings.nlargest(n_users) # quantitative evaluations by each user (n_users) sorted: User 1: 5, User 2: 5, User 3: 5, ...
            ratings = ratings[cols.index] # matrix [n_movies rows , original columns]; before [n_movies rows , original columns] (just updated the index)
        else:
            # select the first n_users from the matrix
            cols = ratings.columns[0:n_users]
            ratings = ratings[cols] # matrix [n_movies rows , n_users columns]; before [n_movies rows , original columns]

        ratings = ratings.T # transposed: matrix [n_users rows x n_movies columns];

        return ratings, movie_genres, user_info

    ###################################################################################################################
    # compute_X_est: 
    def  compute_X_est(self, X, algorithm='RecSysALS'):
        
        if(algorithm == 'RecSysALS'):
            
            # factorization parameters
            rank = 1 # before 20
            lambda_ = 1 # before 20 - ridge regularizer parameter

            # initiate a recommender system of type ALS (Alternating Least Squares)
            RS = RecSysALS.als_RecSysALS(rank,lambda_)

            X_est,error = RS.fit_model(X)

        elif(algorithm == 'RecSysKNN'):
            RecSysKNN
        else:
            RecSysNMF
        return X_est


    ###################################################################################################################
    # compute_R_k_top: : top k recommendations for each of the n_users
    def  compute_R_k_top(self, X_est, omega, k):
        #print("compute_R_k_top")
        list_R_k_top = []
        for u in range(0, self.n_users):
            #print("u: ", u)
            X_est_rowU = X_est.iloc[u, :] # extracting the u row (Series) from the DataFrame X_est
            #print(X_est_rowU)
            X_omega_rowU = omega.iloc[u, :] # extracting the u row (Series) from the DataFrame omega
            number_cols = X_est_rowU.count()

            for j in range(0, number_cols):
                if (X_omega_rowU[j] == True):
                    X_est_rowU[j] = 0 # eliminating ratings to leave only recommendations

            X_est_rowU =  X_est_rowU.nlargest(k) # sorting cells in k columns
            X_est_rowU = X_est_rowU.reset_index() # resetting row indices
            list_R_k_top.append(X_est_rowU) # adding user u top k recommendations to all users recommendations list
        return list_R_k_top

    ###################################################################################################################
    # compute_R_k_random: : k recommendations for each of the n_users (using the Random algorithm)
    def compute_R_k_random(self, X_est, omega, k):
        list_R_k_random = []
        for u in range(0, self.n_users):
            X_est_rowU = X_est.iloc[u, :] # extracting the u row (Series) from the DataFrame X_est
            X_omega_rowU = omega.iloc[u, :] # extracting the u row (Series) from the DataFrame omega
            number_cols = X_est_rowU.count()

            for i in range(0, number_cols):
                if (X_omega_rowU[i] == True):
                    X_est_rowU[i] = 0

            X_est_rowU =  X_est_rowU.nlargest(self.l) # sorting cells in l columns
            X_est_rowU = X_est_rowU.reset_index() # resetting row indices

            for i in range(1, self.l-self.k+1):
                X_est_rowU = X_est_rowU.drop([random.randint(0,self.l-i-1)])
                X_est_rowU = X_est_rowU.reset_index(drop=True)
            
            list_R_k_random.append(X_est_rowU)
        return list_R_k_random

    ###################################################################################################################
    # user satisfaction for a user u: A(u)
    def compute_A(self, list_R_k_top, list_R_k_random, u):
        R_k_top_sum = 0
        R_k_random_sum = 0

        for i in range(0, self.k):
            R_k_top_sum += list_R_k_top[u-1].iat[i,1]
            R_k_random_sum += list_R_k_random[u-1].iat[i,1]

        return R_k_random_sum/R_k_top_sum

    ###################################################################################################################
    # score disparity: Ds
    def compute_Ds(self, list_R_k_top, list_R_k_random): # calculating A for all users
        sum_A = 0 # sum of all A(u) user satisfactions
        sum_dif_A = 0 # sum of all differences between user satisfactions |A(u1) - A(u2)|
        list_A_k_random = []
        for u in range(1, self.n_users+1):
            A = self.compute_A(list_R_k_top, list_R_k_random, u)
            list_A_k_random.append(A)

        for u in range(1, self.n_users+1):
            sum_A += list_A_k_random[u-1]
            for i in range(1, self.n_users+1):
                sum_dif_A += abs(list_A_k_random[u-1] - list_A_k_random[i-1])

        return (sum_dif_A) / (2 * self.n_users * sum_A)

    ###################################################################################################################
    # similarity among the recommended items to users and their top-k: sim(u)
    def compute_sim(self, list_R_k_top, list_R_k_random, u):
        intercession = 0

        for i in range(0, self.k):
            for j in range(0, self.k):
                if list_R_k_random[u-1].iat[i,1] == list_R_k_top[u-1].iat[j,1]:
                    intercession+=1

        return intercession/self.k

    ###################################################################################################################
    # recommendation disparity: Dr
    def compute_Dr(self, list_R_k_top, list_R_k_random):
        sum_sim = 0 # sum of all user similarities A(u)
        sum_dif_sim = 0 # sum of all differences between user similarities |sim(u1) - sim(u2)|
        list_sim_k_random = []
        for u in range(1, self.n_users+1):
            sim = self.compute_sim(list_R_k_top, list_R_k_random, u)
            list_sim_k_random.append(sim)

        for u in range(1, self.n_users+1):
            sum_sim += list_sim_k_random[u-1]
            for i in range(1, self.n_users+1):
                sum_dif_sim += abs(list_sim_k_random[u-1] - list_sim_k_random[i-1])

        return (sum_dif_sim) / (2 * self.n_users * sum_sim)

    ###################################################################################################################
    # aggregate diversity: Da
    def compute_Da(self, list_R_k_random):
        list_movies_diverse = []

        for u in range(1, self.n_users+1):
            for i in range(0, self.k):
                if not(list_R_k_random[u-1].iat[i,0] in list_movies_diverse):
                    list_movies_diverse.append(list_R_k_random[u-1].iat[i,0])

        return len(list_movies_diverse) / self.n_movies

#######################################################################################################################