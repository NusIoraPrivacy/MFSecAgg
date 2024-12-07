import pandas as pd
import numpy as np
import math
import time
import torch
import torch.optim as optim
from torch import nn

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
          df = pd.read_csv(file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
          rating_matrix = pd.pivot_table(df,values = ['rating'],index=['user_id'], columns=['item_id'],fill_value=0)
        else:
          df = pd.read_csv(file_path, delimiter='::',names=['UserID','MovieID','Rating','Timestamp'], engine = "python")
          rating_matrix = pd.pivot_table(df,values = ['Rating'],index=['UserID'], columns=['MovieID'],fill_value=0)
        df = 0
        #n = 69878//2
        #m = 10677//2

    return torch.tensor(rating_matrix.values)

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

# metrics
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
        intersect_item_list = torch.unique(torch.cat((rated_item_list, recommendation_list), 0))
        current_precision = len(torch.where(torch.isin(intersect_item_list, recommendation_list))[0]) / recommend_number
        try:
            current_recall = len(torch.where(torch.isin(intersect_item_list, rated_item_list))[0]) / len(rated_item_list)
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


class federated_matrix_factorization_with_projection_p:
    def __init__(self, train_data, rating_matrix, test_index_matrix, k, p, p_2, e_max, e_min,  user_size, item_size,
                 lambda_, miu,
                 H_D,  alpha_uncertain, C_uncertain,
                 device,
                 beta_1 = 0.5, beta_2 = 0.8, epsilon_adam = 0.001, lr_adam = 0.05):

        # Read and store icuut variables
        self.__rating_train = train_data.to(device)
        self.__rating_matrix = rating_matrix.to(device)
        self.__test_index_matrix = test_index_matrix.to(device)
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
        self.__S = torch.normal(0, 1, size=(item_size, p),device = device) # requires_grad=True)
        self.__V0 = torch.randn(self.__item_size, self.__latent_factor, device = device) # requires_grad=True)
        self.__H_D = H_D.to(device)
        self.__H_D_T = H_D.t().to(device) # .transpose(0,1)
        self.__alpha = alpha_uncertain
        self.__C_uncertain = C_uncertain.to(device)

        # adam
        self.__beta_1 = beta_1
        self.__beta_2 = beta_2
        self.__epsilon_adam = epsilon_adam
        self.__lr_adam = lr_adam
        
        # User latent matrix, Item latent matrix, projected V
        self.__U = torch.randn(self.__user_size, self.__latent_factor, device = device)#.to(device) # n*k
        self.__V = torch.randn(self.__item_size, self.__latent_factor)  # m*k # , requires_grad=True
        self.__B = torch.zeros([self.__dim_after_reduction,self.__latent_factor], requires_grad=True, device = device)  # p*k
        
        # optimizer
        self.optimizer = optim.Adam([self.__B], lr=self.__lr_adam, betas=(self.__beta_1, self.__beta_2), eps=self.__epsilon_adam)

        # device
        self.__device = device
        
        
        
    def flatMa(self, gradient_B):
        padding = self.__flattern_dim - self.__dim_after_reduction
        padded_gradient_B = torch.nn.functional.pad(gradient_B, (0, 0, 0, padding), mode='constant').detach()#.to(self.__device)
        flat_gradient_B = torch.matmul(self.__H_D, padded_gradient_B).detach()#.to(self.__device)
        return flat_gradient_B

    def RevMat(self, flat_gradient_B):
        gradient_B = torch.matmul(self.__H_D_T, flat_gradient_B).detach().to(self.__device)
        gradient_B = gradient_B[:self.__dim_after_reduction, :].clone().detach()
        return gradient_B

    def loss(self, u_id):
        r_pre = torch.mm(self.__U[u_id].unsqueeze(0), (self.__V0 + torch.mm(self.__S, self.__B)).t()).detach() # .to(self.__device)
        loss = torch.dot(self.__C_uncertain[u_id].to(torch.float32), 
                         ((r_pre - self.__rating_train[u_id])**2).squeeze().to(torch.float32)) \
               + self.__lambda * (torch.norm(self.__U[u_id])**2) \
               + self.__miu * (torch.norm(self.__V0 + torch.mm(self.__S, self.__B))**2)
        
        # memory release
        del r_pre
        torch.cuda.empty_cache()
        
        return loss

    def update_U(self, u_id, V_i):
        for j in u_id:
            Rui = self.__rating_train[j].view(-1, 1).float() # .to(torch.float32).clone().detach().to(self.__device)
            c_matrix =  torch.diag(self.__C_uncertain[j]).to(self.__device).float()
            VT_C_V =  (torch.mm(V_i.t(), torch.mm(c_matrix, V_i)) * 2)
            VT_C_Rui = torch.mm(V_i.t(), torch.mm(c_matrix, Rui))
            self.__U[j] = torch.mm(torch.inverse(VT_C_V + self.__lambda * torch.eye(self.__latent_factor, device = self.__device)),VT_C_Rui).flatten()
            
            # # memory release
            # Rui = Rui.cpu()
            # c_matrix = c_matrix.cpu()
            # VT_C_V = VT_C_V.cpu()
            # VT_C_Rui = VT_C_Rui.cpu()
            # del Rui, c_matrix,VT_C_V, VT_C_Rui
            # del c_matrix, VT_C_V, VT_C_Rui
            torch.cuda.empty_cache()
            
            
    def train(self, B_1, theta, m_pbm, maxitr = 20, sample_pct = 0.1):
        # initialize ADAM matrix
        m_adam = 0
        v_adam = 0

        sample_size = int(self.__user_size * sample_pct) # sample users size
        e = (abs(self.__emax)>abs(self.__emin) and abs(self.__emax) or abs(self.__emin))
        delta_1 = (4*e*(1+self.__alpha)/sample_pct) * B_1 # delta_1

        t = 0
        Z_sum = torch.zeros((self.__flattern_dim, self.__latent_factor)).to(self.__device)
        c_1 = delta_1/(self.__dim_after_reduction)
        c_2 = m_pbm * sample_size     
          
        for i in range(maxitr):
            print(f'Epoch: {i+1}')
            t1_1 = time.time()
            sample_users = torch.randint(0, self.__user_size, (sample_size,))
            V_i = self.__V0 + torch.mm(self.__S, self.__B)
            
            # loss_list = []
            for step, j in enumerate(sample_users):
                self.optimizer.zero_grad()

                # update U_j
                self.update_U([j], V_i)
                
                # compute gradient B_j
                loss = self.loss(j)
                # loss_list.append(loss.item())
                loss.backward(retain_graph=True)
                # if i >= 1:
                #     print(self.__B.requires_grad)
                #     print(self.__B.grad)
                gradient_B_j = self.__B.grad.detach()
                
                # clip gradient B
                this_norm = torch.norm(gradient_B_j, p = 1).to(self.__device)
                if this_norm >= delta_1:
                    gradient_B_j = gradient_B_j * delta_1/this_norm
                else:
                    gradient_B_j = 1
                
                # flatten gradient B_j
                flat_gradient_B_j = self.flatMa(gradient_B_j)

                # PBM
                flat_gradient_B_j = torch.clamp(flat_gradient_B_j, -c_1, c_1).to(self.__device)
                Rescale_flat_gradient_B_j = (theta / c_1) * flat_gradient_B_j + 0.5
                Z_j = torch.bernoulli(Rescale_flat_gradient_B_j * m_pbm)
                Z_sum = Z_sum + Z_j
                
                # memory release
                del loss, this_norm, gradient_B_j, flat_gradient_B_j, Rescale_flat_gradient_B_j, Z_j
                torch.cuda.empty_cache()
            
            # print("Loss:", sum(loss_list)/len(loss_list))
            gradient_B = c_1 / (c_2 * theta) * (Z_sum - c_2 / 2) # SecAgg
            gradient_B = gradient_B / sample_pct # rescale
            gradient_B = self.RevMat(gradient_B) # unflatten

            
            # ADAM
            # m_adam = self.__beta_1*m_adam + (1-self.__beta_1)* gradient_B
            # m_adam_adjust = m_adam/(1-self.__beta_1)
            # v_adam = self.__beta_2*v_adam + (1-self.__beta_2)* gradient_B * gradient_B
            # v_adam_adjust = v_adam/(1-self.__beta_2)
            # self.__B = (self.__B - ((maxitr - i)/maxitr)*self.__lr_adam*m_adam_adjust/(torch.sqrt(v_adam_adjust) + self.__epsilon_adam)).clone().detach()
            self.optimizer.zero_grad()
            self.__B.grad = gradient_B.clone().detach()
            
            self.optimizer.step()
            # self.__B = (self.__B - ((maxitr - i)/maxitr)*self.__lr_adam*m_adam_adjust/(torch.sqrt(v_adam_adjust) + self.__epsilon_adam))
            # print(self.__B.requires_grad)
            t1_2 = time.time()

            # memory release
            Z_sum.zero_()
            del V_i, gradient_B
            torch.cuda.empty_cache()
            
            # update V
            self.__V = (self.__V0 + torch.mm(self.__S, self.__B))
            t1_2 = time.time()
            t  = t + t1_2 - t1_1

            if i == maxitr - 1:
                # distribute the final matrix to all users for update
                self.update_U(torch.arange(self.__user_size),self.__V)
                print(f'Train and test accurary for epoch: {i+1}')
                # print("Loss:", sum(loss_list)/len(loss_list))
                self.sample_score()

            # if (i+1)%10 == 0:
            #     print(f'Train and test accurary for epoch: {i+1}')
            #     # print("Loss:", sum(loss_list)/len(loss_list))
            #     print(f"Average Iteration time:{t/(i+1)}")
            #     self.sample_score()

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
time1 = time.time()
rating_matrix = dataset_loader("Movie", "/mnt/sda/MF_secagg/data/ml-100k/ml-100k/u.data")
user_size = rating_matrix.size(0)
item_size = rating_matrix.size(1)
print(user_size, item_size)
train_val_matrix, test_index_matrix = set_splitter(rating_matrix, user_size, item_size, pct=0.8)

k = 14
# privacy budgets
pb_eps = 0.01
pb_alpha = 2
p = item_size//2
pbm_m, theta = get_theta_m(pb_eps, pb_alpha, 1, user_size, p, k)
print("Privacy budgets:", pbm_m, theta)
# uncertain matrix
alpha_uncertain = 14
C_uncertain = (train_val_matrix > 0).float() * alpha_uncertain + 1

# Hadamard matrix
p_2 = 2 ** math.ceil(math.log2(p))
H = walsh(p_2).to(torch.float32)

diag = torch.randint(0, 2, (p_2,))
diag[diag == 0] = -1
D = torch.diag(diag).to(torch.float32)
H_D = torch.matmul(H, D)

mf = federated_matrix_factorization_with_projection_p(
    train_data = train_val_matrix, rating_matrix = rating_matrix, test_index_matrix = test_index_matrix, k = k, p = p, p_2 = p_2,
    e_max = 1.5, e_min = -1, user_size = user_size, item_size = item_size,
    lambda_ = 0.25, miu = 0.9,
    H_D = H_D,  alpha_uncertain = alpha_uncertain, C_uncertain = C_uncertain,
    device = torch.device('cuda:3'),
    beta_1 = 0.9, beta_2 = 0.8, epsilon_adam = 0.01, lr_adam = 0.05)

# without calculating m_pbm --> m_pbm = 1, without calculating theta --> theta = 0.25
mf.train(B_1 = 0.001, theta = theta, m_pbm = pbm_m, maxitr = 200, sample_pct = 0.1) 


time2 = time.time()
print(f"Total time:{time2-time1}")
