from UserFairness import UserFairness
import matplotlib.pyplot

# reading data from a base with 10 movies and 8 users
#Data_path = 'Data/Movie10Items'
#n_users=  8
#n_movies= 10
#top_users = False # True: to use users with more ratings; False: otherwise

# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users=  1000
n_movies= 1700
top_users = True # True: to use users with more ratings; False: otherwise

# recommendation algorithm
algorithm = 'RecSysAntidoteData'

# parameters for calculating fairness measures
list_l = [10, 100, 200, 500, 1000]
theta = 5
k = 5

# results list
list_ds = []
list_dr = []
list_da = []

for l in list_l:

    print("\nl = ", l)

    uf = UserFairness(n_users, n_movies, top_users, l, theta, k)

    X, genres, user_info = uf.read_movieitems(n_movies, n_users, top_users, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_moveis columns
    omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

    X_est = uf.compute_X_est(X, algorithm) # RecSysAntidoteData or RecSysKNN or RecSysNMF

    list_R_k_top = uf.compute_R_k_top(X_est, omega, k)
    list_R_k_random = uf.compute_R_k_random(X_est, omega, k)

    Ds = uf.compute_Ds(list_R_k_top, list_R_k_random)
    print("Score Disparity => Ds: ", Ds)

    Dr = uf.compute_Dr(list_R_k_top, list_R_k_random)
    print("Recommendation Disparity => Dr: ", Dr)

    Da = uf.compute_Da(list_R_k_random)
    print("Aggregate Diversity => Da: ", Da)

    list_ds.append(Ds)
    list_dr.append(Dr)
    list_da.append(Da)

matplotlib.pyplot.plot(list_da, list_ds)
matplotlib.pyplot.title('Random post-processing algorithm (k =' + str(k) + ' )')
matplotlib.pyplot.suptitle('CF algorithms: ' + algorithm)
matplotlib.pyplot.xlabel('Aggregate Diversity')
matplotlib.pyplot.ylabel('Score Disparity')
matplotlib.pyplot.show()

matplotlib.pyplot.plot(list_da, list_dr)
matplotlib.pyplot.title('Random post-processing algorithm (k =' + str(k) + ' )')
matplotlib.pyplot.suptitle('CF algorithms: ' + algorithm)
matplotlib.pyplot.xlabel('Aggregate Diversity')
matplotlib.pyplot.ylabel('Recommendation Disparity')
matplotlib.pyplot.show()