import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ring_buffer import RingBuffer
from torch.func import vmap, jacrev, functional_call, hessian
from torch.utils.data import DataLoader
import numpy as np
import scipy

from tqdm import tqdm

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, \
    add_rehearsal_args, ArgumentParser

import copy
import wandb

def kernel_fn(fnet_single, params, x1, x2, compute='full'):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1.values()]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2.values()]

    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf,Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf,Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf,Maf->NMa'
    else:
        assert False

    result = torch.stack(
        [torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Sparse Functional Regularizer')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--tau', type=float, required=True,
                        default=1.0,
                        help='tau for the SFR regulariser')
    parser.add_argument('--delta', type=float, required=True,
                        default=1.0,
                        help='delta hyperparam')
    parser.add_argument("--A_batchsize", type=int, required=False,
                        default=500, help="")
    parser.add_argument("--jitter", type=float, required=False,
                        default=1e-6)
    parser.add_argument("--no_covariance", action="store_true")
    return parser


class SFR(ContinualModel):
    NAME = 'sfr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform, dataset):
        super(SFR, self).__init__(backbone, loss, args, transform)

        self.n_tasks = dataset.N_TASKS
        self.n_classes = dataset.N_CLASSES
        self.buffer_size = args.buffer_size
        self.z_per_task = self.buffer_size // self.n_tasks
        self.seen_classes = []

        self.task_cnt = 0

        self.delta = args.delta
        self.tau = args.tau
        self.jitter = args.jitter

        self.buffer = RingBuffer(self.args.buffer_size, self.device,
                                 self.n_tasks)

        self.A_matrix = torch.zeros(size=(self.n_tasks, self.n_classes, self.z_per_task, self.z_per_task),
                                    device='cpu')
        self.A_batchsize = self.args.A_batchsize

        self.Sigma_inv = torch.zeros(size=(self.n_tasks, self.n_classes, self.z_per_task, self.z_per_task),
                                     device=self.device, requires_grad=False)
        # self.Sigma_diag = torch.zeros(size=(self.n_tasks * self.z_per_task, self.n_classes),
        #                               device=self.device, dtype=torch.float32, requires_grad=False)

    def l2_reg(self):
        l2_norm = sum(p.pow(2.0).sum() for p in self.net.parameters())
        return 0.5 * l2_norm

    def sfr_regularizer(self):
        R = torch.tensor(0.).to(self.device)
        if self.task_cnt == 0: return R
        buf_inputs, _, buf_logits = self.buffer.get_all_data(transform=self.transform)
        # print(buf_inputs.shape, buf_logits.shape)
        for t in range(self.task_cnt):
            R_t = torch.tensor(0.).to(self.device)
            # compute logits
            idxs_t = torch.from_numpy(
                np.arange(t * self.z_per_task,
                          (t + 1) * self.z_per_task)
            ).to(self.device)
            inputs_t, logits_t = buf_inputs[idxs_t].to(self.device), \
                                 buf_logits[idxs_t].to(self.device)

            new_logits = self.net(inputs_t)
            for c in range(self.n_classes):
                diff = torch.flatten(new_logits[:, c] - logits_t[:, c])[:, None]
                # print(new_logits[:, c])
                # print(logits_t[:, c])
                d_iS_d = diff.T @ self.Sigma_inv[t, c] @ diff#.T

                # d_iS_d = torch.einsum('mi, mn, ni -> i',
                #                       diff,
                #                       #torch.eye(diff.shape[0]).to(self.device),
                #                       self.Sigma_inv[t, c],
                #                       diff).squeeze()

                # d_iS_d = torch.einsum('ki, n, ki -> i',
                #                       diff,
                #                       torch.diag(self.Sigma_inv[t, c]),
                #                       diff).squeeze()

                # print(self.Sigma_inv[t, c].shape, self.Sigma_inv.requires_grad)
                # print(f"diff: {diff.sum()}, {d_iS_d}")
                R_t += d_iS_d.squeeze()
                # print(R_t)
            R += R_t / self.z_per_task  # /( 10_000 * (self.task_cnt + 1))# self.z_per_task / (self.task_cnt + 1)
        return 0.5 * R

    @torch.no_grad()
    def end_task(self, dataset: ContinualDataset):

        self.net.eval()
        if self.task_cnt == (self.n_tasks - 1): return

        task_dataset = dataset.train_loader.dataset
        # Random pick inducing points from the current task
        task_num_samples = len(task_dataset)
        sample_idxs = np.random.choice(a=task_num_samples,
                                       size=self.z_per_task)
        selected_samples = []
        for idx in sample_idxs:
            sample = task_dataset[idx]
            _, target, not_aug_input = sample
            selected_samples.append([not_aug_input, target])
        # list of (input, target) ->  2 lists inputs, targets
        selected_samples = list(zip(*selected_samples))
        z_inputs = torch.stack(selected_samples[0])
        z_targets = torch.tensor(selected_samples[1])
        # get logits for the sampled inputs
        z_logits = self.net(z_inputs.to(self.device)).data

        self.buffer.add_data(examples=z_inputs,
                             labels=z_targets,
                             logits=z_logits)

        if self.args.nowand == 0:
            images = wandb.Image(z_inputs)
            wandb.log({"examples": images})

        A_loader = DataLoader(dataset=task_dataset, batch_size=self.A_batchsize,
                              shuffle=False, num_workers=4)
        start_idx = 0
        lambdas = torch.zeros((task_num_samples, self.n_classes))
        for batch in A_loader:
            _, targets, images = batch
            logits = self.net(images.to(self.device))
            # logits_train.append(logits)
            lambda_i = vmap(hessian(self.loss))(logits, targets.to(self.device))
            lambda_i = vmap(torch.diag)(lambda_i)
            end_idx = start_idx + targets.shape[0]
            lambdas[start_idx: end_idx] = lambda_i
            start_idx = end_idx

        lambdas = torch.clip(lambdas, min=1e-7)
        
        # current neural network weights
        curr_net = copy.deepcopy(self.net)
        curr_net.eval()
        current_params = {k: v.detach() for k, v in curr_net.named_parameters()}

        print()
        if self.task_cnt != (self.n_tasks - 1):
            z_inputs = z_inputs.to(self.device)
            for class_out in range(self.n_classes):
                self.Sigma_inv[self.task_cnt, class_out] = torch.eye(z_inputs.shape[0])
            if self.args.no_covariance is False:
                # print(f"Classes task {self.task_cnt}: {np.unique(A_loader.dataset.targets)}")
                self.seen_classes = np.unique(np.array(np.unique(A_loader.dataset.targets).tolist() + self.seen_classes)).tolist()
                print(self.seen_classes)
                for class_out in self.seen_classes: # range(self.n_classes): #self.seen_classes: #np.unique(A_loader.dataset.targets):

                    def fnet_single(params, x):
                        f = functional_call(curr_net, params, (x.unsqueeze(0)))[:, class_out] # TODO: Why using self.net doesn't work?
                        return f

                    Kzz = 1 / (self.delta * task_num_samples * (self.task_cnt  + 1)) * \
                        kernel_fn(fnet_single,
                                    current_params,
                                    x1=z_inputs,
                                    x2=z_inputs).squeeze()
                    A_k = torch.zeros((self.z_per_task, self.z_per_task))

                    def compute_Ai(kzi, lambda2):
                        return torch.outer(kzi, kzi) * lambda2

                    start_idx = 0
                    for i, batch in enumerate(tqdm(A_loader, ncols=50)):
                        # A_loader):
                        _, _, images = batch
                        images = images.to(self.device)
                        kzi = 1 / (self.delta * task_num_samples * (self.task_cnt  + 1)) * \
                            kernel_fn(fnet_single,
                                        current_params,
                                        z_inputs,
                                        images).squeeze()
                        end_idx = start_idx + images.shape[0]
                        # print(kzi.shape, lambdas[start_idx: end_idx, class_out].shape)
                        A_k += (vmap(compute_Ai)(kzi.T.cpu(),
                                                lambdas[start_idx: end_idx,
                                                class_out].cpu())).sum(dim=0)
                        del kzi, images
                        start_idx = end_idx
                    self.A_matrix[self.task_cnt, class_out] = A_k

                    

                    Kzz = Kzz.cpu().to(torch.float64)
                    Kzz = (Kzz + Kzz.T)/2 + torch.eye(Kzz.shape[0]) * self.jitter
                    det_Kzz = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(Kzz))))
                    print(f"det Kzz: {det_Kzz}")

                    A_k = (A_k + torch.eye(Kzz.shape[0]) * self.jitter).numpy()
                    det_A = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(A_k))))
                    print(f"det A_k: {det_A}")

                    Lz = scipy.linalg.cho_factor(Kzz)
                    iKA = scipy.linalg.cho_solve(Lz, A_k)
                    Sigma_inv_k = scipy.linalg.cho_solve(Lz, iKA.T)

                    # minS = np.min(np.diag(Sigma_inv_k))
                    # maxS = np.max(np.diag(Sigma_inv_k))
                    # Sigma_inv_k = (Sigma_inv_k - minS) / (maxS - minS)

                    self.Sigma_inv[self.task_cnt, class_out] = torch.from_numpy(Sigma_inv_k)
                    print(torch.diag(self.Sigma_inv[self.task_cnt, class_out]))

                    # torch.set_default_dtype(torch.float64)
                    # A_k = A_k.to(torch.float64).to(self.device)
                    # #Kzz = (Kzz.to(torch.float64).to(self.device) + torch.eye(Kzz.shape[0]).to(
                    # #     self.device) * self.jitter).to(self.device)
                    # Kzz = Kzz.to(self.device)
                    #
                    # iKA = torch.linalg.solve(Kzz, A_k)
                    # Sigma_inv_k = torch.linalg.solve(Kzz, iKA.T).to(
                    #     torch.float32).to(self.device)
                    # print(torch.diag(Sigma_inv_k))
                    #
                    # Lz = torch.linalg.cholesky(Kzz, upper=False)
                    # iKA = torch.linalg.solve_triangular(Lz.T, A_k, upper=True)
                    # iKA = torch.linalg.solve_triangular(Lz, iKA, upper=False)
                    # iKAiK = torch.linalg.solve_triangular(Lz.T, iKA.T, upper=True)
                    # iKAiK = torch.linalg.solve_triangular(Lz, iKAiK, upper=False)
                    # self.Sigma_inv[self.task_cnt, class_out] = iKAiK
                    # print(torch.diag(self.Sigma_inv[self.task_cnt, class_out]))
                    # torch.set_default_dtype(torch.float32)
                    # del A_k, iKA, iKAiK
                torch.cuda.empty_cache()
                del z_inputs
        print(self.Sigma_inv[:self.task_cnt+1])
        self.Sigma_inv.detach_()
        print(self.Sigma_inv)
        self.net.train()
        self.task_cnt += 1


    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()

        outputs = self.net(inputs)

        ce_loss = self.loss(outputs, labels)
        l2_reg = self.l2_reg() * self.delta

        sfr_reg = self.sfr_regularizer() * self.tau

        # print(ce_loss, sfr_reg)
        # print()
        loss = ce_loss + sfr_reg + l2_reg

        if self.args.nowand == 0:
            wandb.log({"sfr_reg": sfr_reg.item(), "l2_reg": l2_reg.item()})

        loss.backward()
        self.opt.step()
        # self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
