import torch
from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
from ITHP import ITHP
import global_configs
from global_configs import DEVICE


class ITHP_DebertaModel(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (global_configs.TEXT_DIM, global_configs.ACOUSTIC_DIM,
                                              global_configs.VISUAL_DIM)
        self.config = config
        self.pooler = BertPooler(config)
        model = DebertaV2Model.from_pretrained("/root/autodl-tmp/recursive_language/deberta-v3-base")
        self.model = model.to(DEVICE)
        ITHP_args = {
            'X0_dim': TEXT_DIM,
            'X1_dim': ACOUSTIC_DIM,
            'X2_dim': VISUAL_DIM,
            'B0_dim': int(multimodal_config.B0_dim),
            'B1_dim': int(multimodal_config.B1_dim),
            'inter_dim': multimodal_config.inter_dim,
            'max_sen_len': multimodal_config.max_seq_length,
            'drop_prob': multimodal_config.drop_prob,
            'p_beta': multimodal_config.p_beta,
            'p_gamma': multimodal_config.p_gamma,
            'p_lambda': multimodal_config.p_lambda,
            'halting_threshold': multimodal_config.halting_threshold,
        }

        self.ITHP = ITHP(ITHP_args)

        self.expand = nn.Sequential(
            nn.Linear(int(multimodal_config.B1_dim), TEXT_DIM),
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.syntax_composer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(multimodal_config.dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.syntax_norm = nn.LayerNorm(config.hidden_size)
        self.syntax_fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid(),
        )
        self.syntax_fusion_norm = nn.LayerNorm(config.hidden_size)
        self.init_weights()
        self.beta_shift = multimodal_config.beta_shift

    def _build_attention_mask(self, input_ids):
        pad_token_id = self.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        return input_ids.ne(pad_token_id).long()

    def _recursive_compose(self, sequence_output, attention_mask):
        current_state = sequence_output
        current_mask = attention_mask.bool()

        while current_state.size(1) > 1:
            if current_state.size(1) % 2 != 0:
                pad_state = torch.zeros(
                    current_state.size(0),
                    1,
                    current_state.size(2),
                    device=current_state.device,
                    dtype=current_state.dtype,
                )
                pad_mask = torch.zeros(
                    current_mask.size(0),
                    1,
                    device=current_mask.device,
                    dtype=current_mask.dtype,
                )
                current_state = torch.cat([current_state, pad_state], dim=1)
                current_mask = torch.cat([current_mask, pad_mask], dim=1)

            left_state = current_state[:, 0::2, :]
            right_state = current_state[:, 1::2, :]
            left_mask = current_mask[:, 0::2]
            right_mask = current_mask[:, 1::2]

            composed_state = self.syntax_composer(torch.cat([left_state, right_state], dim=-1))
            composed_state = self.syntax_norm(composed_state + 0.5 * (left_state + right_state))

            both_valid = left_mask & right_mask
            left_only = left_mask & (~right_mask)
            right_only = right_mask & (~left_mask)
            zero_state = torch.zeros_like(composed_state)

            current_state = torch.where(
                both_valid.unsqueeze(-1),
                composed_state,
                torch.where(
                    left_only.unsqueeze(-1),
                    left_state,
                    torch.where(right_only.unsqueeze(-1), right_state, zero_state),
                ),
            )
            current_mask = left_mask | right_mask

        return current_state[:, 0, :]

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
    ):
        attention_mask = self._build_attention_mask(input_ids)
        embedding_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = embedding_output[0]

        b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps = self.ITHP(x, visual, acoustic)

        h_m = self.expand(b1)
        acoustic_vis_embedding = self.beta_shift * h_m
        sequence_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + x)
        )
        pooled_output = self.pooler(sequence_output)
        syntax_root = self._recursive_compose(sequence_output, attention_mask)
        syntax_gate = self.syntax_fusion_gate(torch.cat([pooled_output, syntax_root], dim=-1))
        pooled_output = self.syntax_fusion_norm(pooled_output + syntax_gate * syntax_root)

        return pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps


class ITHP_DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dberta = ITHP_DebertaModel(config, multimodal_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.acc7_classifier = nn.Linear(config.hidden_size, 7)
        self.output_gate = nn.Linear(config.hidden_size, 1)
        self.register_buffer("acc7_values", torch.arange(-3, 4, dtype=torch.float))

        self.init_weights()

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
    ):
        pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps = self.dberta(
            input_ids,
            visual,
            acoustic,
        )
        pooled_output = self.dropout(pooled_output)
        regression_logits = self.classifier(pooled_output)
        acc7_logits = self.acc7_classifier(pooled_output)
        acc7_probs = torch.softmax(acc7_logits, dim=-1)
        acc7_expectation = torch.sum(acc7_probs * self.acc7_values.view(1, -1), dim=-1, keepdim=True)
        blend_gate = torch.sigmoid(self.output_gate(pooled_output))
        logits = blend_gate * regression_logits + (1.0 - blend_gate) * acc7_expectation

        return logits, acc7_logits, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps
