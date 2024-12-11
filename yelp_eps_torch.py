import pandas as pd
import numpy as np
import time
from data_util import load_data
import torch
import torch.optim as optim
import math
import argparse


# set splitting
def set_splitter(rating_matrix, user_size, item_size, pct: float = 0.8):
    num_test = int(item_size * (1 - pct))
    train_val_matrix = rating_matrix.clone()
    test_index_matrix = torch.zeros(user_size, num_test, dtype=torch.int64)
    
    for u in range(user_size):
        test_index = torch.tensor(np.random.choice(np.arange(item_size), size=num_test, replace=False))
        train_val_matrix[u, test_index] = 0
        test_index_matrix[u] = test_index

    return train_val_matrix, test_index_matrix


# average precision
def cal_average_precision(rated_item_list, recommendation_list):
    # average precision for each user
    recom_num = len(recommendation_list)
    rated_num = len(rated_item_list)
    accurate_num = 0
    average_precision = 0

    for i in range(recom_num):
        rel = int(torch.sum(torch.eq(recommendation_list[i], rated_item_list)).item())
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
        rated_item_list = torch.where(current_test_actual >= 4)[0]
        recommendation_list = torch.argsort(current_test_pred, descending=True)[:recommend_number]
        current_precision = len(torch.where(torch.isin(rated_item_list, recommendation_list))[0]) / recommend_number
        try:
            current_recall = len(torch.where(torch.isin(recommendation_list, rated_item_list))[0]) / len(rated_item_list)
        except ZeroDivisionError:
            current_recall = 0
        current_ap = cal_average_precision(rated_item_list, recommendation_list)
        precision += current_precision
        recall += current_recall
        mapping += current_ap

        non_zero_actual = torch.where(current_test_actual != 0)[0]
        if len(non_zero_actual) > 0:
            nonzero_test_pred = current_test_pred[non_zero_actual]
            nonzero_train_actual = train_matrix[u]
            nonzero_train_actual = nonzero_train_actual[nonzero_train_actual > 0]
            non_zero_train_actual = torch.where(nonzero_train_actual != 0)[0]
            if len(non_zero_train_actual) > 0:
                train_actual_avg = torch.mean(nonzero_train_actual.float())
                nonzero_test_pred = nonzero_test_pred + train_actual_avg - torch.mean(nonzero_test_pred)
                current_mse = torch.sqrt(torch.mean((current_test_actual[non_zero_actual] - nonzero_test_pred) ** 2))
                mse += current_mse

    rmse = mse/num_users
    precision = precision/num_users
    recall = recall/num_users
    f1 = 2 * precision * recall / (precision + recall)
    mapping /= num_users

    print(f"rmse:{rmse}")
    print(f"precision: {precision}")
    print(f"recall:{recall}")
    print(f"f1:{f1}")
    print(f"mapping:{mapping}")

def sparseProject(w, d, t):
    sparseMat = torch.zeros((w*t, d))
    for i in range(t):
        hashIdx = torch.randint(w, (d,))
        randSigns = torch.randint(0, 2, (d,)) * 2 - 1
        for j in range(w):
            sparseMat[j+w*i] = (hashIdx == j).float() * randSigns
    return sparseMat / torch.sqrt(torch.tensor(t).float())

# Hadamard matrix
def hadamard(p):
    if p == 1:
        return torch.tensor([[1]])
    else:
        H = hadamard(p // 2)
        return torch.cat((torch.cat((H, H), dim=1), torch.cat((H, -H), dim=1)), dim=0)

def walsh(p):
    H = hadamard(p)
    H = H / torch.sqrt(torch.tensor(p, dtype=torch.float))
    return H

# get theta and m_pbm
def get_theta_m(eps, alpha, sample_pct, n, p, k):
    # solve the theta given q=1
    const = eps*sample_pct*n*10/(p*k)*((alpha-1)/alpha**2)
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

class federated_matrix_factorization_with_projection_p:
    def __init__(self, train_data, rating_matrix, test_index_matrix, k, p, p_2, e_max, e_min,  user_size, item_size,
                 lambda_, miu,
                 S, H_D,  alpha_uncertain, C_uncertain,
                 device,
                 beta_1 = 0.5, beta_2 = 0.8, epsilon_adam = 0.001, lr_adam = 0.05):

        # Read and store icuut variables
        self.__rating_train = train_data.to(device)
        self.__rating_matrix = rating_matrix.to(device)
        self.__test_index_matrix = test_index_matrix # .to(device)
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
        S = S.to_sparse()
        self.__S = S.to(device)
        self.__S_T = S.t().to(device)
        self.__H_D = H_D.to(device)
        self.__H_D_T = H_D.t().to(device)
        self.__alpha = alpha_uncertain
        self.__C_uncertain = C_uncertain.to(device)

        # User latent matrix, Item latent matrix, projected V
        self.__U = torch.rand(self.__user_size, self.__latent_factor, device = device)
        self.__V = torch.rand(self.__item_size, self.__latent_factor, device = device)
        # self.__B = torch.matmul(self.__S, self.__V).to(device)
        self.__B = torch.sparse.mm(self.__S, self.__V)
        
        # adam
        self.optimizer = optim.Adam([self.__B], lr=lr_adam, betas=(beta_1, beta_2), eps=epsilon_adam)
        
        # device
        self.__device = device 


    def update_U(self, sample_users):
        for i in sample_users:
            Rui = self.__rating_train[i].view(-1, 1).float() # .to(torch.float32).clone().detach().to(self.__device)
            c_matrix =  torch.diag(self.__C_uncertain[i]).to(self.__device).float()
            VT_C_V =  (torch.mm(self.__V.t(), torch.mm(c_matrix, self.__V)) * 2).to(self.__device)
            VT_C_Rui = torch.mm(self.__V.t(), torch.mm(c_matrix, Rui)).to(self.__device)
            self.__U[i] = torch.mm(torch.inverse(VT_C_V + self.__lambda * torch.eye(self.__latent_factor, device = self.__device)),VT_C_Rui).flatten().detach()
            
            del Rui, c_matrix, VT_C_V, VT_C_Rui
            torch.cuda.empty_cache()


    def U_l1_const_bound(self, sample_users, B_1):
        for i in sample_users:
            this_norm = torch.norm(self.__U[i], p = 1).to(self.__device)
            if this_norm >= B_1:
                self.__U[i] = (self.__U[i] * B_1/this_norm)
            else:
                self.__U[i] = 1

            del this_norm
            torch.cuda.empty_cache()

    def flatMa(self, gradient_B):
        padding = self.__flattern_dim - self.__dim_after_reduction
        padded_gradient_B = torch.nn.functional.pad(gradient_B, (0, 0, 0, padding), mode='constant').to(self.__device)
        flat_gradient_B = torch.matmul(self.__H_D, padded_gradient_B)
        
        del padded_gradient_B
        torch.cuda.empty_cache()
        
        return flat_gradient_B

    def RevMat(self, flat_gradient_B):
        gradient_B = torch.matmul(self.__H_D_T, flat_gradient_B)
        gradient_B = gradient_B[:self.__dim_after_reduction, :]
        return gradient_B

    def loss(self):
        r_pre = torch.matmul(self.__U, self.__V.T).to(self.__device)
        # Calculate the loss function using PyTorch operations
        loss = torch.sum(self.__C_uncertain * (r_pre - self.__rating_train)**2).to(self.__device) + \
            self.__lambda * torch.norm(self.__U).to(self.__device)**2 + \
            self.__miu * torch.norm(self.__V).to(self.__device)**2
        
        del r_pre
        torch.cuda.empty_cache()
        
        return loss

    def train(self, B_1, theta, m_pbm, maxitr = 20, sample_pct = 0.1):
        # initialize ADAM matrix
        m_adam = 0
        v_adam = 0

        sample_size = int(self.__user_size * sample_pct) # sample users size
        e = (abs(self.__emax)>abs(self.__emin) and abs(self.__emax) or abs(self.__emin))
        delta_1 = (4*e*(1+self.__alpha)/sample_pct) * B_1 # delta_1
        c_1 = delta_1/(self.__dim_after_reduction)
        c_2 = m_pbm * sample_size
        Z_sum = torch.zeros((self.__flattern_dim, self.__latent_factor),device = self.__device)
        
        t = 0
        for i in range(maxitr):
            t1_1 = time.time()
            # self.__V =  cu.dot(self.__S_T, self.__B)
            sample_users = torch.randint(0, self.__user_size, (sample_size,))
            #self.__V = cu.asnumpy(self.__V)
            self.update_U(sample_users)
            self.U_l1_const_bound(sample_users, B_1)

            # update V
            sample_U = self.__U[sample_users]
            MF_error_new = self.__rating_train[sample_users] - torch.matmul(sample_U, self.__V.T)
            MF_error_new = torch.clamp(MF_error_new, min=self.__emin, max=self.__emax).to(self.__device)
            MF_error_uncertain = (self.__C_uncertain[sample_users] * MF_error_new).to(self.__device)

            for j in range(sample_size):
                gradient_V_j = torch.outer(MF_error_uncertain[j], sample_U[j]).float().to(self.__device)
                # gradient_B_j = torch.matmul(self.__S, gradient_V_j).to(self.__device)  # projection
                gradient_B_j = torch.sparse.mm(self.__S, gradient_V_j)
                flat_gradient_B_j = self.flatMa(gradient_B_j)  # flatten

                # PBM
                flat_gradient_B_j = torch.clamp(flat_gradient_B_j, -c_1, c_1).to(self.__device)
                Rescale_flat_gradient_B_j = (theta / c_1) * flat_gradient_B_j + 0.5
                Z_j = torch.distributions.Binomial(m_pbm, Rescale_flat_gradient_B_j).sample()
                Z_sum = Z_sum + Z_j
                
                del gradient_V_j, gradient_B_j, flat_gradient_B_j, Rescale_flat_gradient_B_j, Z_j
                torch.cuda.empty_cache()

            gradient_B = c_1 / (c_2 * theta) * (Z_sum - c_2 / 2)  # SecAgg
            gradient_B = self.RevMat(gradient_B/ sample_pct)  # rescale & unflatten

            # ADAM
            self.optimizer.zero_grad()
            self.__B.grad = gradient_B.clone().detach()
            self.optimizer.step()
            
            # update V
            # self.__V = torch.matmul(self.__S_T, self.__B)
            self.__V = torch.sparse.mm(self.__S_T, self.__B)
            # memory release
            Z_sum.zero_()
            del sample_U, MF_error_new, MF_error_uncertain, gradient_B
            torch.cuda.empty_cache()
            
            t1_2 = time.time()
            t  = t + t1_2 - t1_1

            # if i == maxitr - 1:
            #     # distribute the final matrix to all users for update
            #     self.update_U(torch.arange(self.__user_size))
            #     print(f'Train and test accurary for epoch: {i+1}')
            #     #print(self.loss())
            #     self.sample_score()

            if (i+1)%10 == 0:
                if i == maxitr - 1:
                    self.update_U(torch.arange(self.__user_size))
                print(f'Train and test accurary for epoch: {i+1}')
                #print(self.loss())
                print("Average Iteration time:")
                print(t/(i+1))
                self.sample_score()


    def sample_score(self):
        #In-sample scores
        print("In-sample scores")
        r_pre = torch.matmul(self.__U, self.__V.T).to(self.__device)
        F1_score(r_pre, self.__rating_matrix, self.__rating_train, self.__test_index_matrix, self.__user_size, recommend_number=10)

        delta_E = self.__rating_train - r_pre
        print("e_min: ", torch.min(delta_E).item())
        print("e_max: ", torch.max(delta_E).item())
        
        del r_pre, delta_E
        torch.cuda.empty_cache()

# data preprocessing
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/mnt/sda/MF_secagg")
    parser.add_argument("--dataset", type=str, default="yelp")
    parser.add_argument("--p_ratio", type=int, default=10)
    parser.add_argument("--pb_eps", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--sample_pct", type=float, default=0.1)
    parser.add_argument("--lambda_", type=float, default=0.25)
    parser.add_argument("--miu", type=float, default=0.9)
    args = parser.parse_args()
    return args

args = get_args()
item_df, user_df, rating_df = load_data(args)
rating_df = pd.pivot_table(rating_df,values = ['Rating'],index=['UserID'], columns=['ItemID'],fill_value=0)
user_size = 10000
item_size = 93387//3 # Memory-Usage: ~20000MB
rating_matrix = torch.tensor(rating_df.iloc[:user_size,:item_size].values) # .iloc[:5000,:5000]
print(user_size,item_size)
item_df, user_df, rating_df = 0, 0, 0

train_val_matrix, test_index_matrix = set_splitter(rating_matrix, user_size, item_size, pct = 0.8)


k = args.k
# privacy budgets
pb_eps = args.pb_eps
pb_alpha = 2
p = item_size//args.p_ratio
pbm_m, theta = get_theta_m(pb_eps, pb_alpha, 1, user_size, p, k)
print("Privacy budgets:", pbm_m, theta)
# uncertain matrix
alpha_uncertain = 14
C_uncertain = ((train_val_matrix > 0) * alpha_uncertain + 1).float()
# random projection matrix
S = sparseProject(p, item_size, 1)
p_2 = 2 ** math.ceil(math.log2(p))
H = walsh(p_2).float() # Walsh-Hadamard matrix
diag = torch.randint(0, 2, (p_2,)) * 2 - 1
D = torch.diag(diag).float()
H_D = torch.matmul(H, D)

mf = federated_matrix_factorization_with_projection_p(
    train_data = train_val_matrix, rating_matrix = rating_matrix, test_index_matrix = test_index_matrix, k = k, p = p, p_2 = p_2,
    e_max = 1.5, e_min = -1, user_size = user_size, item_size = item_size,
    lambda_ = args.lambda_, miu = args.miu,
    S = S, H_D = H_D,  alpha_uncertain = alpha_uncertain, C_uncertain = C_uncertain,
    # device=torch.device('cuda:1'),
    device="cuda",
    beta_1 = 0.9, beta_2 = 0.8, epsilon_adam = 0.01, lr_adam = 0.05)

mf.train(B_1 = 0.001, theta = theta, m_pbm = pbm_m, maxitr = 100, sample_pct = args.sample_pct) # without calculating m_pbm --> m_pbm = 1, without calculating theta --> theta = 0.25