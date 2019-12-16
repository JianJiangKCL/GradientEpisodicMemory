
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import numpy.random as random
# testfill = torch.Tensor(3,4)
import random
# testfill.fill_(1.0)#
# testsum = sum(testfill[1,:2])
# t_unsqueeze = testfill.unsqueeze(1)
# print(t_unsqueeze.shape,t_unsqueeze)
# print(testfill)
# print(testsum)
import quadprog
# margin = 0.5
# eps=1e-3
# memories = torch.randn([20,5]) # 20 p, 5 tasks
# gradient = torch.randn([20])
# gradient = gradient.unsqueeze(1)
# memories_np = memories.cpu().t().double().numpy()  # .t() means transpose in numpy # [5,20]
# print(memories_np.shape, "memo np")
# gradient_np = gradient.cpu().contiguous().view(-1).double().numpy() # contiguous is not like sharing memory, and view(-1)
# # is to let it become a column vector
# print(gradient_np.shape, "grad np")
# print(gradient.shape, "grad np")
# t = memories_np.shape[0]  # so t is 5 in this case
#
# print(t,'t')
# P = np.dot(memories_np, memories_np.transpose()) # [5,5] tasks
# print(P.shape)
# P = 0.5 * (P + P.transpose()) + np.eye(t) * eps  # G*GT
# q = np.dot(memories_np, gradient_np) * -1 # gt*GT
# G = np.eye(t)   # constraints, v>=0 ,so G matrix in QP concept is an identity matrix
# # a np list add a real number, in default will change the real number to the same size list
# h = np.zeros(t) + margin    # margin is like a threshold, loose the constraints
#
# print(h, h.shape)
# v = quadprog.solve_qp(P, q, G, h)[0]
# #quadprog.solve.QP(Dmat, dvec, Amat, bvec, meq=0, factorized=FALSE)
#
# x = np.dot(v, memories_np) + gradient_np # g~ = GT* (v*) + g => x = (v*) * GT + g
# print(x)
# print(torch.Tensor(x).view(-1, 1).shape)
# print('gradient', gradient)
# gradient.copy_(torch.Tensor(x).view(-1, 1))
# max = 6
# # a = random.randint(0,max,[2,4])
# a = [1,2,3,4,5,6]
# a = [i for i in range(0, max)]
# a= random.sample(a,4)
# k = torch.randn([9,10])
# out = k[a]
# # a = torch.rand([2,3])
# print(a)
# print(k)
# print(out)
# a = torch.from_numpy(a)
# out = a.index_select(0,torch.tensor([0,1]))
# print(out)


'''
note 
use chmod -R 777
'''

# list = []
#
# for j in range(5):
#     tmp = [[j,i] for i in range(6)]
#     random.shuffle(tmp)
#     list += tmp
# print(list)
b = 5
d =100
x = torch.randn([b,d])
v_x = x.view(x.size(0), -1)
print(v_x.shape)