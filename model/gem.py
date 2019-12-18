# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import quadprog
import random
from .common import MLP, ResNet18

# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
        'cause cifar share a common output layer

        nc_per_task is the num of classes per task
    """
    # offset1 and offset2 's difference is the nc_per_task
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2

#todo find the process of grads
def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    # grad_dims[0:1] the num of 0th layer's paras
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])  # beginning
            en = sum(grad_dims[:cnt + 1])     # end
            grads[beg: en, tid].copy_(param.grad.data.view(-1))  # to 1 dimension
        cnt += 1

def grads_sampling(grads, sampling_rate=1):
    len_params = grads.shape[0]
    sampling_size = int(len_params * sampling_rate)
    idx_params_list = [i for i in range(len_params)]
    sampled_idx = random.sample(idx_params_list, sampling_size)
    sampled_grads = grads[sampled_idx]
    return sampled_grads, sampled_idx

# self.parameters is a list that contains each layer's params
def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        # print(param)
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1
    print(cnt)

def grads_back(sampled_idx, sampled_grads, grads):
    grads[sampled_idx] = sampled_grads
    return grads

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        a network has p params.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector; each task is trained in the network, so each of them has p gradients
        output: x, p-vector; the projected gradient.
    """
    # comment refers to the formulation 11
    memories_np = memories.cpu().t().double().numpy()  # GT = -MT
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()  # g
    t = memories_np.shape[0]    # task mums
    P = np.dot(memories_np, memories_np.transpose())    # GT*G = MT*M
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps  # 0.5(MT*M + M*MT) + I*eps, original should be MT*M;todo why
    q = np.dot(memories_np, gradient_np) * -1   # GT*g = GT*g = (gT*GT)T   original (-MT*b)T
    G = np.eye(t)   # identity matrix, In constraints, G'x<=h (G' is the original G in QP)
    # i.e. as G=-MT, so Gx>=h, i.e. v>=h
    h = np.zeros(t) + margin  # v>=h; margin is like loose factor # todo why set margin to 0.5
    v = quadprog.solve_qp(P, q, G, h)[0]  # get the optimal solution of v~
    x = np.dot(v, memories_np) + gradient_np    # g~ = v*GT +g, TODO but should be  g~= GT*v~ +g
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.is_cifar = (args.data_file == 'cifar100.pt')
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
            size = [n_inputs] + [nh] * nl + [n_outputs]
            print('szie',size)

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)
        self.sampling_rate = float(args.sampling_rate)/100.0
        self.n_memories = args.n_memories   # 256 for all the tasks in gem
        self.gpu = args.cuda
        self.violate_time = 0
        self.iteration = 0
        # allocate episodic memory
        # n_inputs are detailed features of samples
        self.memory_data = torch.FloatTensor(
            n_tasks, self.n_memories, n_inputs)
        # memory_lab is labels
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        # this is for ewc
        self.grad_dims = []
        for param in self.parameters():
            # .numel() return the size of parameters in a tensor
            # so grad_dims stores para num of each layer
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            # for mnist todo
            self.nc_per_task = n_outputs


    # t is for cifar
    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output


    def get_violation_frequnecy(self):
        return self.violate_time/float(self.iteration)
    # this is like batch iteration with constraints
    def observe(self, x, t, y):

        self.iteration += 1
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
        param_num = sum(self.grad_dims[:])
        # print('param_num', param_num)

        # Update ring buffer storing examples from current task
        # todo buffer size should be changed into index of sample in the memory
        bsz = y.data.size(0)  # buffer size is the num of samples
        endcnt = min(self.mem_cnt + bsz, self.n_memories)   # compared with pre-define memory size 256
        effbsz = endcnt - self.mem_cnt  # effective buffer size
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])  # copy the eff size samples to the memory_data
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        # below condition fulfill the idea that select the last m samples from a task, as it will become 0.
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # todo select parts of gradient from the
        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):

                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                   self.is_cifar)
                # loss of a certain previous task's performance in the memory
                ptloss = self.ce(
                    self.forward(
                        self.memory_data[past_task],
                        past_task)[:, offset1: offset2],
                    self.memory_labs[past_task] - offset1)
                # loss.backward() is to calculate the gradients
                ptloss.backward()
                # the result of store_grad would be saved in self.grads
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)

        # now compute the grad on the current minibatch
        """Sets gradients of all model parameters to zero."""
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])
            # .unsqueeze(dim) add 1 dimension at dim place, start from 0
            ''' index_select(dim,indices)
            indices should be tensor form
            from all the sub-tensor in that dim, select the indices
            indices's max value is the num of elements of that dimensionF
            '''

            # dotproduction
            # the t-th task (current task)
            # dotp = torch.mm(self.grads[:, t].unsqueeze(0),
            #                 self.grads.index_select(1, indx))
            # if (dotp < 0).sum() != 0:
            #     # the result of project2cone2 saved in the first input para, self.grads
            #     project2cone2(self.grads[:, t].unsqueeze(1),
            #                   self.grads.index_select(1, indx), self.margin)
            #     # copy gradients back
            #     overwrite_grad(self.parameters, self.grads[:, t],
            #                    self.grad_dims)

            # todo sample memory. and use the constraints in a-gem
            # sampled version of constraints

            sampled_grads, sampled_idx = grads_sampling(self.grads, sampling_rate=self.sampling_rate)
            # they are all tensors
            # print(type(self.grads))
            # print(type(sampled_grads))
            sampled_idx = torch.cuda.LongTensor(sampled_idx) if self.gpu \
                else torch.LongTensor(sampled_idx)
            # sampled_grads.shape
            # torch.Size([44805, 20])
            # grads.shape
            # torch.Size([89610, 20])

            # print('sampled_grads.shape', sampled_grads.shape)
            # print('grads.shape', self.grads.shape)
            dotp = torch.mm(sampled_grads[:, t].unsqueeze(0),
                            sampled_grads.index_select(1, indx)
                            )
            if (dotp < 0).sum() != 0:
                self.violate_time += 1
                # the result of project2cone2 saved in the first input para, self.grads
                project2cone2(sampled_grads[:, t].unsqueeze(1),
                              sampled_grads.index_select(1, indx), self.margin)
                # copy gradients back

                self.grads[sampled_idx] = sampled_grads
                # self.grads = grads_back(sampled_grads, self.grads, sampled_idx)
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)

        self.opt.step()
