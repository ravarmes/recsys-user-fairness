from UserFairness import UserFairness


# reading data from a base with 10 movies and 8 users
Data_path = 'Data/Movie10Items'
n_users=  8
n_movies= 10
top_users = False # True: to use users with more ratings; False: otherwise

# reading data from 3883 movies and 6040 users 
#Data_path = 'Data/MovieLens-1M'
#n_users=  300
#n_movies= 1000
#top_users = True # True: to use users with more ratings; False: otherwise

# recommendation algorithm
algorithm = 'RecSysALS'

# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

uf = UserFairness(n_users, n_movies, top_users, l, theta, k)

X, genres, user_info = uf.read_movieitems(n_movies, n_users, top_users, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_moveis columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

#print("X")
#print(X)

#print("Omega")
#print(omega)

X_est = uf.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF
#print("X_est")
#print(X_est)

list_R_k_top = uf.compute_R_k_top(X_est, omega, k)
list_R_k_random = uf.compute_R_k_random(X_est, omega, k)

print("list_R_k_top")
print(*list_R_k_top, sep = "\n-----------------\n")

print("\n")

print("list_R_k_random")
print(*list_R_k_random, sep = "\n-----------------\n")

Ds = uf.compute_Ds(list_R_k_top, list_R_k_random)
print("Score Disparity => Ds: ", Ds)

Dr = uf.compute_Dr(list_R_k_top, list_R_k_random)
print("Recommendation Disparity => Dr: ", Dr)

Da = uf.compute_Da(list_R_k_random)
print("Aggregate Diversity => Da: ", Da)