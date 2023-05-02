# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
from utils.ring_buffer import RingBuffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser


import torch
import numpy as np
from torchvision import transforms


class Derringbuffer(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = RingBuffer(buffer_size=self.args.buffer_size,
                                 device=self.device,
                                 n_tasks=5)# Buffer(self.args.buffer_size, self.device)

    def end_task(self, dataset):

        task_dataset = dataset.train_loader.dataset
        # Random pick inducing points from the current task
        task_num_samples = len(task_dataset)
        sample_idxs = np.random.choice(a=task_num_samples,
                                       size=self.args.buffer_size // 5)
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

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _,  buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.opt.step()
        # self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
