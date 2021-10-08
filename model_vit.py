import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from vit_pytorch import ViT


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ActorCriticViT(nn.Module):
    def __init__(self):
        super(ActorCriticViT, self).__init__()


        self.vit_model = ViT(
                image_size = 768,
                patch_size = 24,
                channels = 1,
                num_classes = 256,
                dim = 1024,
                depth = 5,
                heads = 16,
                mlp_dim = 1024,
                dropout = 0.1,
                emb_dropout = 0.1)

        self.actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 113)
        )

        self.extra_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.critic_ext = nn.Linear(256, 1)
        self.critic_int = nn.Linear(256, 1)

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.extra_layer)):
            if type(self.extra_layer[i]) == nn.Linear:
                init.orthogonal_(self.extra_layer[i].weight, 0.1)
                self.extra_layer[i].bias.data.zero_()

    def forward(self, state):
        x = self.vit_model(state)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.extra_layer(x) + x)
        value_int = self.critic_int(self.extra_layer(x) + x)
        return policy, value_ext, value_int


class RNDModelViT(nn.Module):
    def __init__(self):
        super(RNDModelViT, self).__init__()

        self.predictor = ViT(
                image_size = 768,
                patch_size = 24,
                channels = 1,
                num_classes = 256,
                dim = 1024,
                depth = 5,
                heads = 16,
                mlp_dim = 1024,
                dropout = 0.1,
                emb_dropout = 0.1)

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=1),
            Flatten(),
            nn.ReLU(),
            nn.Linear(5440,256)
        )
        for p in self.target.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature
