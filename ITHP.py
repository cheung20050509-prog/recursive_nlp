import torch
import torch.nn as nn
import global_configs


# from global_configs import *

class ITHP(nn.Module):
    def __init__(self, ITHP_args):
        super(ITHP, self).__init__()
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (global_configs.TEXT_DIM, global_configs.ACOUSTIC_DIM,
                                              global_configs.VISUAL_DIM)

        self.X0_dim = ITHP_args['X0_dim']
        self.X1_dim = ITHP_args['X1_dim']
        self.X2_dim = ITHP_args['X2_dim']
        self.inter_dim = ITHP_args['inter_dim']
        self.drop_prob = ITHP_args['drop_prob']
        self.B0_dim = int(ITHP_args['B0_dim'])
        self.B1_dim = int(ITHP_args['B1_dim'])
        self.p_beta = ITHP_args['p_beta']
        self.p_gamma = ITHP_args['p_gamma']
        self.p_lambda = ITHP_args.get('p_lambda', 1.0)

        self.max_recursion_depth = getattr(global_configs, 'MAX_RECURSION_DEPTH', 3)
        self.halting_threshold = ITHP_args.get('halting_threshold', 0.01)

        self.encoder1 = nn.Sequential(
            nn.Linear(self.X0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.B0_dim * 2),
        )

        self.MLP1 = nn.Sequential(
            nn.Linear(self.B0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.X1_dim),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(self.B0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.B1_dim * 2),
        )

        self.recursive_encoder = nn.Sequential(
            nn.Linear(self.B1_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.B1_dim * 2),
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(self.B1_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.X2_dim),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        self.recursive_norm = nn.LayerNorm(self.B1_dim)

        self.criterion = nn.MSELoss()

    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x, visual, acoustic):
        h1 = self.encoder1(x)
        mu1, logvar1 = h1.chunk(2, dim=-1)
        kl_loss_0 = self.kl_loss(mu1, logvar1)
        b0 = self.reparameterise(mu1, logvar1)
        output1 = self.MLP1(b0)
        mse_0 = self.criterion(output1, acoustic)
        IB0 = kl_loss_0 + self.p_beta * mse_0

        # 第二层的第一步保持原始 ITHP 设计，确保 IB 目标不被破坏
        h2 = self.encoder2(b0)
        mu2, logvar2 = h2.chunk(2, dim=-1)
        kl_loss_1 = self.kl_loss(mu2, logvar2)
        current_state = self.reparameterise(mu2, logvar2)
        current_mean = self.recursive_norm(mu2)
        output2 = self.MLP2(current_state)
        mse_1 = self.criterion(output2, visual)
        batch_size = current_state.size(0)
        executed_steps = torch.ones(batch_size, device=current_state.device, dtype=torch.long)
        active_mask = torch.ones(batch_size, device=current_state.device, dtype=torch.bool)

        # 在原始第二层瓶颈之上做递归细化，只影响最终融合表示，不改原始 IB 损失定义
        prev_cls_mean = current_mean[:, 0, :].detach()
        for step in range(1, self.max_recursion_depth):
            if not active_mask.any().item():
                break

            h_refine = self.recursive_encoder(current_state)
            mu_refine, logvar_refine = h_refine.chunk(2, dim=-1)
            candidate_mean = self.recursive_norm(current_mean + mu_refine)
            refine_state = self.reparameterise(mu_refine, logvar_refine)
            candidate_state = self.recursive_norm(current_state + refine_state)
            executed_steps = executed_steps + active_mask.long()

            current_cls_mean = candidate_mean[:, 0, :].detach()
            cls_delta = torch.mean(torch.abs(current_cls_mean - prev_cls_mean), dim=-1)

            active_mask_3d = active_mask[:, None, None]
            current_state = torch.where(active_mask_3d, candidate_state, current_state)
            current_mean = torch.where(active_mask_3d, candidate_mean, current_mean)

            newly_halted = active_mask & (cls_delta < self.halting_threshold)
            active_mask = active_mask & (~newly_halted)
            prev_cls_mean = torch.where(active_mask[:, None], current_mean[:, 0, :].detach(), prev_cls_mean)

        IB1 = kl_loss_1 + self.p_gamma * mse_1
        IB_total = IB0 + self.p_lambda * IB1

        return current_state, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, executed_steps
