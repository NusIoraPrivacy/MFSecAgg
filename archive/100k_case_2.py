import pandas as pd
import numpy as np
import cupy as cu # https://docs.cupy.dev/en/stable/index.html
import math
import cupyx.scipy.sparse as sparse
from cupyx.scipy.sparse.linalg import spsolve
import time

# loading dataset:
def dataset_loader(
    dataset_name: str,
    file_path: str,
):
    # Book-Rating
    if dataset_name == "Book":
        data = pd.read_csv(file_path) # 'Book-Ratings.csv'
        top_users = data.groupby('User-ID')['Book-Rating'].count()
        top_users = top_users.sort_values(ascending=False)[:6000].index
        data = data[data['User-ID'].isin(top_users)]
        top_items = data.groupby('ISBN')['Book-Rating'].count()
        top_items = top_items.sort_values(ascending=False)[:3000].index
        data = data[data['ISBN'].isin(top_items)]
        rating_matrix = pd.pivot_table(data,values = ['Book-Rating'],index=['User-ID'], columns=['ISBN'],fill_value=0)

    # Movie-Rating
    elif dataset_name == "Movie":
        if file_path.endswith(".csv"):
          df = pd.read_csv(file_path)
          rating_matrix = pd.pivot_table(df,values = ['Rating'],index=['UserID'], columns=['MovieID'],fill_value=0)
        elif file_path.endswith(".data"):
          df = pd.read_csv(file_path, '\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
          rating_matrix = pd.pivot_table(df,values = ['rating'],index=['user_id'], columns=['item_id'],fill_value=0)
        else:
          df = pd.read_csv(file_path, delimiter='::',names=['UserID','MovieID','Rating','Timestamp'], engine = "python")
          rating_matrix = pd.pivot_table(df,values = ['Rating'],index=['UserID'], columns=['MovieID'],fill_value=0)
        #df = 0
        #n = 69878//2
        #m = 10677//2

    return cu.asarray(rating_matrix.values)


# set splitting
def set_splitter(
    rating_matrix,
    user_size,
    item_size,
    pct: float = 0.8
):
    num_test = item_size - int(item_size * pct)
    train_val_matrix = rating_matrix.copy()
    test_index_matrix = cu.zeros((user_size, num_test))
    #test_val_matrix = cu.zeros((user_size,item_size))

    for u in range(user_size):
        test_index = np.random.choice(np.arange(item_size),size = num_test,replace=False)
        #test_val_matrix[u,test_index] = train_val_matrix[u,test_index]
        train_val_matrix[u,test_index] = 0
        test_index_matrix[u] =  cu.asarray(test_index)

    test_index_matrix = test_index_matrix.astype(int)

    return train_val_matrix, test_index_matrix

# metric
def score(r_pre, r, metric):
    if metric == "MSE":
        return cu.mean((r_pre-r)**2)
    elif metric == "R_squared":
        ss_res = cu.sum( (r - r_pre)**2 )
        ss_tot = cu.sum( (r - cu.average(r)) **2)
        return 1 - ss_res/ss_tot
    elif metric == "MAE":
        return cu.mean(cu.abs(r-r_pre))

def get_theta_m(eps, alpha, p, k):
    # solve the theta given q=1
    const = eps*10/(p*k)*((alpha-1)/alpha**2)
    const = np.sqrt(const)
    coeff = [4*const, -4*const-1, const]
    thetas = np.roots(coeff)
    theta = min(thetas)
    theta = min(theta, 1/4)
    # solve the q again if theta = 1/4
    if theta == 1/4:
        pbm_m = const**2 * (1-2*theta)**4/theta**2
        pbm_m = int(pbm_m)
    else:
        pbm_m = 1
    return pbm_m, theta

# average precision
def cal_average_precision(rated_item_list, recommendation_list):
    # avergae precision for each user
    recom_num = len(recommendation_list)
    rated_num = len(rated_item_list)
    accurate_num = 0
    average_precision = 0
    for i in range(recom_num):
      rel = int(recommendation_list[i] in rated_item_list)
      accurate_num += rel
      current_precision = accurate_num / (i + 1)
      average_precision += current_precision * rel
    if rated_num > 0:
      average_precision /= rated_num
    return average_precision

def F1_score(r_pre, rating_matrix, train_matrix, test_index_matrix, num_users, recommend_number=10):
    mse = 0
    precision = 0
    recall = 0
    mapping = 0
    for u in range(num_users):
        test_index = test_index_matrix[u]
        current_test_actual = rating_matrix[u, test_index]
        current_test_pred = r_pre[u, test_index]
        rated_item_list = cu.where(current_test_actual >= 4)[0]
        recommendation_list = current_test_pred.argsort()[-recommend_number:][::-1]
        intersect_item_list = np.intersect1d(cu. asnumpy(rated_item_list), cu. asnumpy(recommendation_list))
        current_precision = len(intersect_item_list) / recommend_number
        try:
            current_recall = len(intersect_item_list) / len(rated_item_list)
        except ZeroDivisionError:
            current_recall = 0
        current_ap = cal_average_precision(rated_item_list, recommendation_list)

        precision += current_precision
        recall += current_recall
        mapping += current_ap

        non_zero_actual = cu.where(current_test_actual!=0)[0]
        if len(non_zero_actual)>0:
            nonzero_test_pred = current_test_pred[non_zero_actual]
            nonzero_train_actual = train_matrix[u]
            nonzero_train_actual = nonzero_train_actual[nonzero_train_actual > 0]
            non_zero_train_actual = cu.where(nonzero_train_actual!=0)[0]
            if len(non_zero_train_actual)>0:
                train_actual_avg = nonzero_train_actual.mean()
                nonzero_test_pred = nonzero_test_pred + train_actual_avg - nonzero_test_pred.mean()
                current_mse = cu.sqrt(((current_test_actual[non_zero_actual] - nonzero_test_pred) ** 2).mean())
                mse += current_mse

    rmse = mse/num_users
    precision = precision/num_users
    recall = recall/num_users
    f1 = 2 * precision * recall / (precision + recall)
    mapping /= num_users

    print(rmse)
    print(precision)
    print(recall)
    print(f1)
    print(mapping)

def sparseProject(w, d, t):
  sparseMat = cu.zeros((w*t, d))
  for i in range(t):
    hashIdx = cu.random.choice(w, d, replace=True)
    randSigns = cu.random.choice([-1, 1], d, replace=True)
    for j in range(w):
      sparseMat[j+w*i] = (hashIdx == j) * randSigns
  return sparseMat / cu.sqrt(t)

def hadamard(p):
    if p == 1:
        return np.array([[1]])
    else:
        H = hadamard(p // 2)
        return np.block([[H, H], [H, -H]])

def walsh(p):
    H = hadamard(p)
    H = cu.asarray(H) / cu.sqrt(p)
    return H

class federated_matrix_factorization_with_projection_p:
    def __init__(self, train_data, rating_matrix, test_index_matrix, k, p, p_2, e_max, e_min,  user_size, item_size,
                 lambda_, miu,
                 S, H_D,  alpha_uncertain, C_uncertain,
                 beta_1 = 0.5, beta_2 = 0.8, epsilon_adam = 0.001, lr_adam = 0.05):

        # Read and store icuut variables
        self.__rating_train = train_data
        self.__rating_matrix = rating_matrix
        self.__test_index_matrix = test_index_matrix
        self.__latent_factor = k
        self.__dim_after_reduction = p
        self.__flattern_dim = p_2
        self.__emax = e_max
        self.__emin = e_min
        self.__user_size = user_size
        self.__item_size = item_size

        #lambdas
        self.__lambda = lambda_
        self.__miu = miu

        # matrices
        self.__S = S
        self.__S_T = S.T
        self.__H_D = H_D
        self.__H_D_T = H_D.T
        self.__alpha = alpha_uncertain
        self.__C_uncertain = C_uncertain

        # adam
        self.__beta_1 = beta_1
        self.__beta_2 = beta_2
        self.__epsilon_adam = epsilon_adam
        self.__lr_adam = lr_adam

        # User latent matrix, Item latent matrix, projected V
        self.__U = cu.random.rand(self.__user_size, self.__latent_factor)
        self.__V = cu.random.rand(self.__item_size, self.__latent_factor)
        self.__B = cu.dot(self.__S,self.__V)


    def update_U(self, sample_users):
        for i in sample_users:
            V = sparse.csr_matrix(self.__V)
            current_user_vec = self.__rating_train[i]
            current_user_vec = current_user_vec.astype(cu.float32)
            Rui = sparse.csr_matrix(current_user_vec).T
            current_c_vec = self.__C_uncertain[i]
            c_matrix = sparse.diags(current_c_vec)
            VT_C_V = (V.T).dot(c_matrix).dot(V)*2
            VT_C_Rui = (V.T).dot(c_matrix).dot(Rui)

            self.__U[i] = spsolve((VT_C_V + self.__lambda * sparse.eye(self.__latent_factor)), sparse.csr_matrix.toarray(VT_C_Rui))


    def U_l1_const_bound(self, sample_users, B_1):
        for i in sample_users:
            this_norm = abs(self.__U[i]).sum()
            if this_norm >= B_1:
                self.__U[i] = self.__U[i] * B_1/this_norm


    def flatMa(self, gradient_B):
        padding = self.__flattern_dim - self.__dim_after_reduction
        padded_gradient_B = cu.pad(gradient_B, ((0, padding), (0, 0)), mode='constant')
        flat_gradient_B = cu.dot(self.__H_D, padded_gradient_B)
        return flat_gradient_B

    def RevMat(self, flat_gradient_B):
        gradient_B = cu.dot(self.__H_D_T, flat_gradient_B)
        gradient_B = gradient_B[:self.__dim_after_reduction, :]
        return gradient_B

    def loss(self):
        r_pre = cu.dot(self.__U,self.__V.T)
        loss = cu.sum(self.__C_uncertain*(r_pre- self.__rating_train)**2)+ self.__lambda* cu.linalg.norm(self.__U)+self.__miu* cu.linalg.norm(self.__V)
        return loss

    def train(self, B_1, theta, m_pbm, maxitr = 20, sample_pct = 0.1):
        # initialize ADAM matrix
        m_adam = 0
        v_adam = 0

        sample_size = int(self.__user_size * sample_pct) # sample users size
        e = (abs(self.__emax)>abs(self.__emin) and abs(self.__emax) or abs(self.__emin))
        delta_1 = (4*e*(1+self.__alpha)/sample_pct) * B_1 # delta_1

        t = 0
        for i in range(maxitr):
            t1_1 = time.time()
            # self.__V =  cu.dot(self.__S_T, self.__B)
            sample_users = cu.random.choice(self.__user_size, size = sample_size, replace = True, p = None)
            #self.__V = cu.asnumpy(self.__V)
            self.update_U(sample_users)
            self.U_l1_const_bound(sample_users, B_1)

            # update V
            sample_U = self.__U[sample_users]
            MF_error_new = self.__rating_train[sample_users] - cu.dot(sample_U,self.__V.T)
            cu.putmask(MF_error_new, MF_error_new < self.__emax, self.__emax)
            cu.putmask(MF_error_new, MF_error_new > self.__emin, self.__emin)
            MF_error_uncertain = self.__C_uncertain[sample_users] * MF_error_new

            Z_sum = cu.zeros((self.__flattern_dim, self.__latent_factor))
            c_1 = delta_1/(self.__dim_after_reduction)
            c_2 = m_pbm * sample_size
            for j in range(sample_size):
                gradient_V_j = cu.outer(MF_error_uncertain[j], sample_U[j])
                gradient_B_j = cu.dot(self.__S,gradient_V_j) # projection
                flat_gradient_B_j = self.flatMa(gradient_B_j) # flatten

                # PBM
                flat_gradient_B_j = cu.clip(flat_gradient_B_j, -c_1, c_1)
                Rescale_flat_gradient_B_j = (theta/c_1) * flat_gradient_B_j + 0.5
                Z_j = cu.random.binomial(m_pbm, Rescale_flat_gradient_B_j)
                Z_sum  = Z_sum + Z_j

            gradient_B = Z_sum / sample_pct # rescale
            gradient_B = self.RevMat(gradient_B) # unflatten

            # ADAM
            m_adam = self.__beta_1*m_adam + (1-self.__beta_1)* gradient_B
            m_adam_adjust = m_adam/(1-self.__beta_1)
            v_adam = self.__beta_2*v_adam + (1-self.__beta_2)* gradient_B * gradient_B
            v_adam_adjust = v_adam/(1-self.__beta_2)
            self.__B = self.__B - ((maxitr - i)/maxitr)*self.__lr_adam*m_adam_adjust/(cu.sqrt(v_adam_adjust) + self.__epsilon_adam)

            self.__V = cu.dot(self.__S_T,self.__B)
            t1_2 = time.time()
            t  = t + t1_2 - t1_1

            # if i == maxitr - 1:
            #   # distribute the final matrix to all users for update
            #   self.update_U(cu.arange(self.__user_size))
            #   print(f'Train and test accurary for epoch: {i+1}')
            #   #print(self.loss())
            #   self.sample_score()

            if (i+1)%10 == 0:
              print(f'Train and test accurary for epoch: {i+1}')
              #print(self.loss())
              print("Average Iteration time:")
              print(t/(i+1))
              self.sample_score()


    def sample_score(self):
        #In-sample scores
        print("In-sample scores")
        r_pre = cu.dot(self.__U,self.__V.T)
        #mse_is = score(r_pre, self.__rating_train, "MSE")
        #r_sq_is = score(r_pre, self.__rating_train, "R_squared")
        #print(mse_is)
        #print(r_sq_is)
        F1_score(r_pre, self.__rating_matrix, self.__rating_train, self.__test_index_matrix, self.__user_size, recommend_number=10)

        delta_E = self.__rating_train - r_pre
        print("e_min: ",cu.min(delta_E))
        print("e_max: ",cu.max(delta_E))

# data preprocessing
rating_matrix = dataset_loader("Movie", "/home/FL_repo/FL_SecAgg/100kdataset/u.data")
user_size = rating_matrix.shape[0]
item_size = rating_matrix.shape[1]
print(user_size,item_size)
train_val_matrix, test_index_matrix = set_splitter(rating_matrix, user_size, item_size, pct = 0.8)

k = 14

# uncertain matrix
alpha_uncertain = 14
C_uncertain = (train_val_matrix > 0) * alpha_uncertain + 1
# privacy budgets
pb_eps = 5
pb_alpha = 2
p = item_size//2
pbm_m, theta = get_theta_m(pb_eps, pb_alpha, p, k)
print("Privacy budgets:", pbm_m, theta)
# uncertain matrix
alpha_uncertain = 14
C_uncertain = (train_val_matrix > 0) * alpha_uncertain + 1

# random projection matrix
S = sparseProject(p, item_size, 1)

p_2 = 2 ** math.ceil(math.log2(p))
H = walsh(p_2) # Walsh-Hadamard matrix
diag = cu.random.choice([-1, 1], size = p_2)
D = cu.diag(diag)
H_D = cu.dot(H, D)

print("0.1_____________________________________________________________")
mf = federated_matrix_factorization_with_projection_p(
    train_data = train_val_matrix, rating_matrix = rating_matrix, test_index_matrix = test_index_matrix, k = k, p = p, p_2 = p_2,
    e_max = 1.5, e_min = -1, user_size = user_size, item_size = item_size,
    lambda_ = 0.25, miu = 0.9,
    S = S, H_D = H_D,  alpha_uncertain = alpha_uncertain, C_uncertain = C_uncertain,
    beta_1 = 0.9, beta_2 = 0.8, epsilon_adam = 0.01, lr_adam = 0.05)

mf.train(B_1 = 0.001, theta = 0.25, m_pbm = 1, maxitr = 250, sample_pct = 0.1)