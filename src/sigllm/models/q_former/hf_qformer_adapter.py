import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional

try:
    from transformers import AutoTokenizer, InstructBlipQFormerConfig, InstructBlipQFormerModel
except (ImportError, ModuleNotFoundError):  # pragma: no cover - depends on runtime env
    AutoTokenizer = None
    InstructBlipQFormerConfig = None
    InstructBlipQFormerModel = None


class HFQFormerAdapter(nn.Module):
    """
    Adapter wrapper around Hugging Face InstructBLIP Q-Former.

    The preferred I/O mirrors InstructBLIP: instruction text is tokenized by
    a Q-Former tokenizer and fed to the Q-Former text stream, while learned
    query tokens read the recommendation signal through cross-attention.

        input:
            cf_vec:        [B, d_cf]
            instruction:   str or list[str]

        output:
            q_tokens:      [B, num_queries, d_model]

    Internally:
    - instruction text -> Q-Former input_ids / attention_mask
    - learned query tokens -> `query_embeds`
    - projected CF vector -> `encoder_hidden_states`

    This follows the InstructBLIP feature-extraction pattern: instructions
    interact with learned queries through Q-Former self-attention, while the
    non-language signal is read through cross-attention.

    TEMP_DISABLED_USER_CF: callers currently pass only item/history CF vectors.
    The adapter stays generic so the old user-CF path can be restored later.
    """

    def __init__(
        self,
        d_cf: int,
        d_model: int,
        num_queries: int = 8,
        num_heads: int = 8,
        num_layers: int = 2,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        intermediate_size: Optional[int] = None,
        cross_attention_frequency: int = 1,
        initializer_range: float = 0.02,
        qformer_text_model_name: str = "bert-base-uncased",
        max_instruction_length: int = 48,
    ):
        super().__init__()

        if AutoTokenizer is None or InstructBlipQFormerConfig is None or InstructBlipQFormerModel is None:
            raise ModuleNotFoundError(
                "transformers with InstructBLIP support is required to use HFQFormerAdapter. "
                "Please install or upgrade transformers in the runtime environment."
            )

        self.d_cf = d_cf
        self.d_model = d_model
        self.num_queries = num_queries
        self.output_dim = int(output_dim) if output_dim is not None else d_model
        self.max_instruction_length = int(max_instruction_length)
        self.qformer_tokenizer = AutoTokenizer.from_pretrained(
            qformer_text_model_name,
            truncation_side="right",
        )

        self.q = Parameter(torch.randn(1, num_queries, d_model))
        self.proj_cf = nn.Linear(d_cf, d_model)
        self.out_proj = nn.Identity() if self.output_dim == d_model else nn.Linear(d_model, self.output_dim)

        config = InstructBlipQFormerConfig(
            vocab_size=len(self.qformer_tokenizer),
            hidden_size=d_model,
            encoder_hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size or (4 * d_model),
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            cross_attention_frequency=cross_attention_frequency,
            initializer_range=initializer_range,
        )
        self.qformer = InstructBlipQFormerModel(config)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Accept both adapter-native keys and keys where the inner HF Q-Former
        prefix was stripped by external loading code.
        """
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict)

        remapped_state_dict = dict(state_dict)
        expected_keys = set(super().state_dict().keys())

        has_prefixed_qformer_keys = any(
            isinstance(k, str) and k.startswith("qformer.") for k in remapped_state_dict
        )
        if not has_prefixed_qformer_keys:
            fixed_state_dict = {}
            remapped_any_key = False
            for key, value in remapped_state_dict.items():
                if isinstance(key, str) and f"qformer.{key}" in expected_keys:
                    fixed_state_dict[f"qformer.{key}"] = value
                    remapped_any_key = True
                else:
                    fixed_state_dict[key] = value
            if remapped_any_key:
                remapped_state_dict = fixed_state_dict

        return super().load_state_dict(remapped_state_dict, strict=strict)

    def _tokenize_instruction(self, instruction, device):
        tokens = self.qformer_tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_instruction_length,
        )
        return tokens.input_ids.to(device), tokens.attention_mask.to(device)

    def forward(self, cf_vec: torch.Tensor, instruction) -> torch.Tensor:
        """
        Args:
            cf_vec: recommendation embedding of shape [B, d_cf]
            instruction: instruction strings consumed by the Q-Former tokenizer

        Returns:
            Query token hidden states of shape [B, num_queries, d_model]
        """
        if cf_vec.dim() != 2:
            raise ValueError(f"Expected cf_vec to have shape [B, d_cf], got {tuple(cf_vec.shape)}")

        batch_size = cf_vec.size(0)
        query_tokens = self.q.expand(batch_size, -1, -1)
        query_count = query_tokens.size(1)

        input_ids, text_attention_mask = self._tokenize_instruction(instruction, cf_vec.device)
        if input_ids.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: cf_vec has B={batch_size} while "
                f"instruction has B={input_ids.size(0)}"
            )

        encoder_hidden_states = self.proj_cf(cf_vec).unsqueeze(1)
        encoder_attention_mask = torch.ones(
            batch_size, encoder_hidden_states.size(1), dtype=torch.long, device=cf_vec.device
        )
        query_attention_mask = torch.ones(
            batch_size, query_count, dtype=torch.long, device=cf_vec.device
        )
        qformer_attention_mask = torch.cat([query_attention_mask, text_attention_mask], dim=1)

        outputs = self.qformer(
            input_ids=input_ids,
            query_embeds=query_tokens,
            attention_mask=qformer_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        query_outputs = outputs.last_hidden_state[:, :query_count]
        return self.out_proj(query_outputs)
