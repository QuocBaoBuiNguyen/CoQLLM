import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional

try:
    from transformers import Blip2QFormerConfig, Blip2QFormerModel
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    Blip2QFormerConfig = None
    Blip2QFormerModel = None


class HFQFormerAdapter(nn.Module):
    """
    Adapter wrapper around Hugging Face BLIP-2 Q-Former.

    This class preserves the same public I/O as the current local QFormer:

        input:
            cf_vec:        [B, d_cf]
            ins_token_emb: [B, T, d_model]

        output:
            q_tokens:      [B, num_queries, d_model]

    Internally, it reshapes the current inputs into the format expected by
    `Blip2QFormerModel`:

    - learned query tokens + instruction embeddings -> `query_embeds`
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
    ):
        super().__init__()

        if Blip2QFormerConfig is None or Blip2QFormerModel is None:
            raise ModuleNotFoundError(
                "transformers is required to use HFQFormerAdapter. "
                "Please install transformers in the runtime environment."
            )

        self.d_cf = d_cf
        self.d_model = d_model
        self.num_queries = num_queries
        self.output_dim = int(output_dim) if output_dim is not None else d_model

        self.q = Parameter(torch.randn(1, num_queries, d_model))
        self.proj_cf = nn.Linear(d_cf, d_model)
        self.out_proj = nn.Identity() if self.output_dim == d_model else nn.Linear(d_model, self.output_dim)

        config = Blip2QFormerConfig(
            hidden_size=d_model,
            encoder_hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size or (4 * d_model),
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            cross_attention_frequency=cross_attention_frequency,
            use_qformer_text_input=False,
            initializer_range=initializer_range,
        )
        self.qformer = Blip2QFormerModel(config)

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

    def forward(self, cf_vec: torch.Tensor, ins_token_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cf_vec: recommendation embedding of shape [B, d_cf]
            ins_token_emb: instruction token embeddings of shape [B, T, d_model]

        Returns:
            Query token hidden states of shape [B, num_queries, d_model]
        """
        # Step 0: Validate the input contract.
        # cf_vec is one CF item/history representation per sample, while
        # ins_token_emb is the token-level instruction representation produced
        # by the frozen text encoder.
        if cf_vec.dim() != 2:
            raise ValueError(f"Expected cf_vec to have shape [B, d_cf], got {tuple(cf_vec.shape)}")
        if ins_token_emb.dim() != 3:
            raise ValueError(
                f"Expected ins_token_emb to have shape [B, T, d_model], got {tuple(ins_token_emb.shape)}"
            )
        if ins_token_emb.size(-1) != self.d_model:
            raise ValueError(
                f"Expected ins_token_emb last dim to equal d_model={self.d_model}, "
                f"got {ins_token_emb.size(-1)}"
            )
        if cf_vec.size(0) != ins_token_emb.size(0):
            raise ValueError(
                f"Batch size mismatch: cf_vec has B={cf_vec.size(0)} while "
                f"ins_token_emb has B={ins_token_emb.size(0)}"
            )

        batch_size = cf_vec.size(0)

        # Step 1: Expand the learnable query tokens for the current batch.
        # self.q has shape [1, Q, d_model]; query_tokens becomes [B, Q, d_model].
        query_tokens = self.q.expand(batch_size, -1, -1)
        query_count = query_tokens.size(1)

        # Step 2: Put instruction tokens in the Q-Former self-attention stream.
        # This follows the InstructBLIP-style design: learned queries can attend
        # to instruction tokens before/while reading the external encoder source.
        # Shape: [B, Q + T, d_model].
        query_embeds = torch.cat([query_tokens, ins_token_emb], dim=1)

        # Step 3: Project the CF vector into the Q-Former hidden size and expose
        # it as the cross-attention source. Shape: [B, 1, d_model].
        cf_token = self.proj_cf(cf_vec).unsqueeze(1)
        encoder_hidden_states = cf_token

        # Step 4: Build full attention masks. There is no padding here because
        # TextEncoder already returns padded embeddings as dense vectors and this
        # adapter receives no text attention mask, so every input token is visible.
        query_attention_mask = torch.ones(
            batch_size, query_embeds.size(1), dtype=torch.long, device=cf_vec.device
        )
        encoder_attention_mask = torch.ones(
            batch_size, encoder_hidden_states.size(1), dtype=torch.long, device=cf_vec.device
        )

        # Step 5: Run the HF BLIP-2 Q-Former.
        # - query_embeds participate in self-attention.
        # - encoder_hidden_states are read through cross-attention.
        # Output shape before slicing: [B, Q + T, d_model].
        outputs = self.qformer(
            query_embeds=query_embeds,
            attention_mask=query_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        # Step 6: Return only the learned query outputs.
        # Instruction token outputs are conditioning context, not soft tokens to
        # pass into the downstream loss/LLM projection.
        query_outputs = outputs.last_hidden_state[:, :query_count]
        return self.out_proj(query_outputs)
