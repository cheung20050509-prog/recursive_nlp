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
        self.syntax_merge_scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.Dropout(multimodal_config.dropout_prob),
            nn.Linear(config.hidden_size, 1),
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
        self.syntax_temperature = max(getattr(multimodal_config, 'syntax_temperature', 1.0), 1e-4)

    def _build_attention_mask(self, input_ids):
        pad_token_id = self.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        return input_ids.ne(pad_token_id).long()

    def _select_merge_weights(self, merge_logits):
        if merge_logits.size(0) == 1:
            return torch.ones_like(merge_logits)

        normalized_logits = torch.clamp(merge_logits / self.syntax_temperature, min=-10.0, max=10.0)

        if self.training:
            # Use a soft relaxation during training to reduce high-variance tree updates.
            return torch.softmax(normalized_logits, dim=0)

        merge_weights = torch.zeros_like(normalized_logits)
        merge_weights[torch.argmax(normalized_logits)] = 1.0
        return merge_weights

    def _build_merge_target_distribution(self, current_spans, gold_span_mask, dtype):
        if gold_span_mask is None or len(current_spans) <= 1:
            return None

        candidate_scores = []
        for left_span, right_span in zip(current_spans[:-1], current_spans[1:]):
            span_start = left_span[0]
            span_end = right_span[1]
            candidate_scores.append(gold_span_mask[span_start, span_end - 1])

        target_distribution = torch.stack(candidate_scores).to(dtype=dtype)
        target_mass = target_distribution.sum()
        if target_mass.item() <= 0:
            return None

        return target_distribution / target_mass

    def _compose_single_tree(self, token_states, gold_span_mask=None, collect_trace=False):
        current_state = token_states
        merge_trace = []
        current_spans = [(index, index + 1) for index in range(token_states.size(0))]
        syntax_nodes = [str(index) for index in range(token_states.size(0))] if collect_trace else None
        syntax_loss_sum = token_states.new_zeros(())
        supervised_steps = 0

        while current_state.size(0) > 1:
            left_state = current_state[:-1]
            right_state = current_state[1:]

            composed_state = self.syntax_composer(torch.cat([left_state, right_state], dim=-1))
            composed_state = self.syntax_norm(composed_state + 0.5 * (left_state + right_state))

            merge_features = torch.cat([left_state, right_state, composed_state], dim=-1)
            merge_logits = self.syntax_merge_scorer(merge_features).squeeze(-1)
            target_distribution = self._build_merge_target_distribution(
                current_spans,
                gold_span_mask,
                merge_logits.dtype,
            )
            if target_distribution is not None:
                normalized_logits = torch.clamp(merge_logits / self.syntax_temperature, min=-10.0, max=10.0)
                syntax_loss_sum = syntax_loss_sum - torch.sum(
                    target_distribution * torch.log_softmax(normalized_logits, dim=0)
                )
                supervised_steps += 1

            merge_weights = self._select_merge_weights(merge_logits)
            selected_merge = int(torch.argmax(merge_weights.detach()).item())
            merged_span = (current_spans[selected_merge][0], current_spans[selected_merge + 1][1])

            current_spans = (
                current_spans[:selected_merge]
                + [merged_span]
                + current_spans[selected_merge + 2:]
            )

            if collect_trace:
                merge_trace.append(selected_merge)
                merged_node = f"({syntax_nodes[selected_merge]} {syntax_nodes[selected_merge + 1]})"
                syntax_nodes = (
                    syntax_nodes[:selected_merge]
                    + [merged_node]
                    + syntax_nodes[selected_merge + 2:]
                )

            prefix_mass = torch.cumsum(merge_weights, dim=0) - merge_weights
            suffix_mass = 1.0 - prefix_mass - merge_weights

            current_state = (
                suffix_mass.unsqueeze(-1) * left_state
                + merge_weights.unsqueeze(-1) * composed_state
                + prefix_mass.unsqueeze(-1) * right_state
            )

        syntax_tree = syntax_nodes[0] if collect_trace else None
        return current_state[0], merge_trace, syntax_tree, syntax_loss_sum, supervised_steps

    def _recursive_compose(self, sequence_output, attention_mask, gold_span_masks=None, collect_trace=False):
        valid_mask = attention_mask.bool()
        syntax_roots = []
        merge_traces = []
        syntax_trees = []
        syntax_loss_sum = sequence_output.new_zeros(())
        syntax_supervised_steps = 0

        for sample_idx in range(sequence_output.size(0)):
            sample_tokens = sequence_output[sample_idx][valid_mask[sample_idx]]
            if sample_tokens.size(0) == 0:
                sample_tokens = sequence_output[sample_idx, :1, :]
            sample_gold_span_mask = gold_span_masks[sample_idx] if gold_span_masks is not None else None
            sample_root, merge_trace, syntax_tree, sample_syntax_loss, sample_supervised_steps = self._compose_single_tree(
                sample_tokens,
                gold_span_mask=sample_gold_span_mask,
                collect_trace=collect_trace,
            )
            syntax_roots.append(sample_root)
            syntax_loss_sum = syntax_loss_sum + sample_syntax_loss
            syntax_supervised_steps += sample_supervised_steps
            if collect_trace:
                merge_traces.append(merge_trace)
                syntax_trees.append(syntax_tree)

        if syntax_supervised_steps > 0:
            syntax_loss = syntax_loss_sum / syntax_supervised_steps
        else:
            syntax_loss = syntax_loss_sum

        return torch.stack(syntax_roots, dim=0), merge_traces, syntax_trees, syntax_loss

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
            syntax_span_masks=None,
            return_syntax_info=False,
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
        syntax_root, merge_traces, syntax_trees, syntax_loss = self._recursive_compose(
            sequence_output,
            attention_mask,
            gold_span_masks=syntax_span_masks,
            collect_trace=return_syntax_info,
        )
        syntax_gate = self.syntax_fusion_gate(torch.cat([pooled_output, syntax_root], dim=-1))
        pooled_output = self.syntax_fusion_norm(pooled_output + syntax_gate * syntax_root)

        if return_syntax_info:
            syntax_info = {
                "merge_traces": merge_traces,
                "syntax_trees": syntax_trees,
            }
            return pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss, syntax_info

        return pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss


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
            syntax_span_masks=None,
            return_syntax_info=False,
    ):
        dberta_outputs = self.dberta(
            input_ids,
            visual,
            acoustic,
            syntax_span_masks=syntax_span_masks,
            return_syntax_info=return_syntax_info,
        )

        if return_syntax_info:
            pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss, syntax_info = dberta_outputs
        else:
            pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss = dberta_outputs

        pooled_output = self.dropout(pooled_output)
        regression_logits = self.classifier(pooled_output)
        acc7_logits = self.acc7_classifier(pooled_output)
        acc7_probs = torch.softmax(acc7_logits, dim=-1)
        acc7_expectation = torch.sum(acc7_probs * self.acc7_values.view(1, -1), dim=-1, keepdim=True)
        blend_gate = torch.sigmoid(self.output_gate(pooled_output))
        logits = blend_gate * regression_logits + (1.0 - blend_gate) * acc7_expectation

        if return_syntax_info:
            return logits, acc7_logits, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss, syntax_info

        return logits, acc7_logits, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss
