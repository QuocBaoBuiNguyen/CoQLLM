
import logging
import random
from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

import os

from sigllm.common.logging_utils import NotebookLogger
from sigllm.common.registry import registry
from sigllm.models.multimodal.base.rec_base_model import Rec2Base

LOGGER = NotebookLogger.rich_logger("sigllm.rec_base_model")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@registry.register_model("mini_gpt4rec_v2")
class QRecLLM(Rec2Base):
    """
    QFormer + InstructBLIP for recommendation.
    """ 
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4rec.yaml",
    }    
    
    PLACEHOLDERS_FOR_EMBED = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]

    def __init__(
        self,
        rec_model="MF",
        rec_config=None,
        pretrained_rec=None,
        freeze_rec=True,
        rec_precision='fp16',
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        proj_drop=0,
        lora_config=None,
        proj_mid=5,
        freeze_lora=False,
        freeze_proj=False
    ):
        super().__init__()

        self.low_resource = low_resource
        self.proj_token_num = proj_token_num

        log_step("Running MiniGPT4Rec_v2 initialization")

        self.rec_model_type = rec_model
        
        # Initialize components
        self._init_rec_model(rec_model, rec_config, rec_precision, pretrained_rec, freeze_rec)
        self._init_llm_model(llama_model, low_resource, device_8bit)
        # self._init_lora(lora_config, freeze_lora)
        self._init_projection(rec_model, rec_config, proj_mid, proj_token_num, freeze_proj)
        self._init_prompts(prompt_path, prompt_template, max_txt_len, end_sym)

    def _init_rec_model(self, rec_model, rec_config, rec_precision, pretrained_rec, freeze_rec):
        log_step("Loading Rec_model")
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, rec_precision)
        
        if self.rec_encoder is not None and pretrained_rec != "not_have":
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
            log_step("Successfully loaded the pretrained model")
        
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            log_step("Freeze rec encoder")

        log_step("Loading Rec_model Done")

    def _init_llm_model(self, llama_model, low_resource, device_8bit):
        log_step(f"Loading LLAMA: {llama_model}")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(
            llama_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        log_step("Loading LLAMA Done")

    def _init_lora(self, lora_config, freeze_lora):
        self.use_lora = False
        if lora_config is not None and lora_config.use_lora:
            log_step("Setting Lora")
            self.use_lora = True
            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.dropout,
                bias="none",
                task_type="CAUSAL_LM"
            ) 
            self.llama_model_lora = get_peft_model(self.llama_model, peft_config)
            log_step("Setting Lora Done")
        
        if freeze_lora and hasattr(self, 'llama_model_lora'):
            log_step("Freeze Lora...")
            for name, param in self.llama_model_lora.named_parameters():
                param.requires_grad = False

    def _init_projection(self, rec_model, rec_config, proj_mid, proj_token_num, freeze_proj):
        if self.rec_encoder is not None and 'prompt' not in rec_model:
            log_step("Initializing projection layer", f"mid={proj_mid}")
            self.llama_proj = nn.Sequential(
                nn.Linear(self.rec_encoder.config.embedding_size, self.rec_encoder.config.embedding_size*int(proj_mid)),
                nn.ReLU(),
                nn.Linear(self.rec_encoder.config.embedding_size*int(proj_mid), self.llama_model.config.hidden_size * proj_token_num),
            )
        elif self.rec_encoder is not None and rec_model == "personlized_prompt":
            log_step("Personalized prompt learning")
            self.llama_proj = nn.Linear(rec_config.item_num + rec_config.user_num, self.llama_model.config.hidden_size * proj_token_num, bias=False)
        elif self.rec_encoder is not None and rec_model == "soft_prompt":
            log_step("Soft prompt learning")
            self.llama_proj = nn.Linear(2, self.llama_model.config.hidden_size * proj_token_num, bias=False)
        else:
            self.llama_proj = None
        
        if freeze_proj and self.llama_proj is not None:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj = self.llama_proj.eval()
            self.llama_proj.train = disabled_train
            log_step("Freeze llama_proj")

    def _init_prompts(self, prompt_path, prompt_template, max_txt_len, end_sym):
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt = False

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            log_step(f"Load {len(self.prompt_list)} training prompts")
            log_step(f"Prompt List: \n{self.prompt_list}")
            self.has_pri_decode = False
            self.prompt_list_p = None
        else:
            self.prompt_list = []
            self.prompt_list_p = None

    def set_mode(self, mode):
        '''
        mode \in ['v1','v2',None]
        '''
        self.run_mode_ = mode

    def to_be_trained(self):
        if self.use_lora:
            return True

        id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True

        return False

    def set_answer_type(self,mode):
        if mode == 'v1':
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            pos_ans_id = self.llama_tokenizer(self.pos_ans[0],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0],add_special_tokens=False).input_ids[0]
            log_step("answer token ids: pos:{}, neg ids:{}".format(pos_ans_id, neg_ans_id))
            
        else:
            raise NotImplementedError("not implement this types of answers")

    def print_prompt(self):
        log_step('Prompt Pos Example \n{} {} or {}'.format(random.choice(self.prompt_list),self.pos_ans[0],self.neg_ans[0]))
    
    def rec_to_cpu(self):
        self.rec_encoder.to("cpu")
        self.rec_encoder.float()
    
    def get_placeholder_order(prompt: str, placeholders=PLACEHOLDERS_FOR_EMBED):
        positions = []
        for ph in placeholders:
            pos = prompt.find(ph)
            if pos >= 0:
                positions.append((pos, ph))
        positions.sort(key=lambda x: x[0])
        return [ph for _, ph in positions]
    
    def build_llm_inputs_from_prompt_v2(self, prompt, samples):
        placeholder_order = self.get_placeholder_order(prompt) if prompt else None
        samples_encode, atts_samples = self.encode_rec_features_to_llm_v2(samples, placeholder_order=placeholder_order)
        sample_embeds, atts_samples = self.wrap_prompt_with_soft_tokens_v2(samples_encode, samples, atts_samples, prompt)
        return sample_embeds, atts_samples

    def encode_rec_features_to_llm_v2(self, sample, placeholder_order=None):
        """
        Encodes recommendation features (User, History, Target) into LLM embedding space.
        
        Args:
            sample (dict): Dictionary containing:
                - 'UserID': (B,)
                - 'TargetItemID': (B,)
                - 'InteractedItemIDs_pad': (B, L)
            placeholder_order (list): Order of features, e.g., ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
            
        Returns:
            sample_embeds_llama (dict):
                - 'User_emb': (B, 1, H) - Individual user representation
                - 'TargetItem_emb': (B, 1, H) - Individual target item representation
                - 'InteractedItems_embs': (B, L, H) - Historical items (includes padding)
                - 'merged_embs': (N, H) - Flattened & filtered valid tokens for LLM input
            sample_atts_llama: None (Placeholder for future attention masks)
        """
        if self.rec_encoder is None:
            return None, None

        device = sample["UserID"].device
        if self.low_resource:
            self.rec_to_cpu()
            for k in sample:
                sample[k] = sample[k].to("cpu")

        with self.maybe_autocast():
            B = sample["UserID"].shape[0]
            H = self.llama_model.config.hidden_size

            all_user_embeds, all_item_embeds = self.rec_encoder.compute()

            # --- user embedding ---
            if self.rec_model_type == "sasrec":
                raise NotImplementedError("sasrec is not implemented in this version")
            elif self.rec_model_type in ("DCN", "DIN"):
                raise NotImplementedError("DCN and DIN are not implemented in this version")
            else:
                user_embeds = self.rec_encoder.user_encoder(sample["UserID"], all_users=all_user_embeds).unsqueeze(-2)

            # --- target item embedding ---
            target_item_embeds = self.rec_encoder.item_encoder(sample["TargetItemID"], all_items=all_item_embeds).unsqueeze(-2)

            # --- project to space llama ---
            user_llama = self.llama_proj(user_embeds).reshape(B, -1, self.proj_token_num, H)
            target_llama = self.llama_proj(target_item_embeds).reshape(B, -1, self.proj_token_num, H)

            interacted_llama_flat = None
            merged_flat = None

            has_interacted = "InteractedItemIDs_pad" in sample
            need_merge = has_interacted and placeholder_order is not None and len(placeholder_order) == 3

            if need_merge:
                interacted = self.rec_encoder.item_encoder(sample["InteractedItemIDs_pad"], all_items=all_item_embeds)
                interacted_llama = self.llama_proj(interacted).reshape(B, -1, self.proj_token_num, H)
                
                # internal mapping for embeddings and masks
                ph2emb = {
                    "<UserID>": user_llama,
                    "<ItemIDList>": interacted_llama,
                    "<TargetItemID>": target_llama
                }

                # Create historical item mask (0 for padding, 1 for real items)
                item_mask = torch.ones_like(sample['InteractedItemIDs_pad'])
                item_mask = torch.where(sample['InteractedItemIDs_pad'] == self.rec_encoder.padding_index, 0, item_mask)

                ph2mask = {
                    "<UserID>": torch.ones([B, 1], device=item_mask.device, dtype=item_mask.dtype),
                    "<ItemIDList>": item_mask,
                    "<TargetItemID>": torch.ones([B, 1], device=item_mask.device, dtype=item_mask.dtype)
                }

                # 3. Concatenate tensors based on the placeholder_order
                # Concat on the sequence dimension (dim 1)
                merged_embeds = torch.cat([ph2emb[ph] for ph in placeholder_order], dim=1) 
                
                # Concat masks to identify non-padded positions
                full_mask = torch.cat([ph2mask[ph] for ph in placeholder_order], dim=1).to(device)
                
                # 4. Extract valid embeddings using the mask
                idx_nopad = torch.nonzero(full_mask) # Get (N, 2) indices
                
                # Index into 4D tensor and flatten to (Total_Valid_Tokens, H)
                # Results in shape: (Total_Items * proj_token_num, H)
                merged_flat = merged_embeds[idx_nopad[:, 0], idx_nopad[:, 1]].reshape(-1, H)

                # Prepare separate flattened versions for output dictionary
                interacted_llama_flat = interacted_llama.reshape(B, -1, H)

            # --- Final output dictionary ---
            sample_embeds_llama = {
                "User_emb": user_llama.reshape(B, -1, H),
                "TargetItem_emb": target_llama.reshape(B, -1, H),
                "InteractedItems_embs": interacted_llama_flat,
                "merged_embs": merged_flat,
            }

        sample_atts_llama = None
        # {
        #     'user': atts_user,
        #     'TargetItem': atts_targetItem,
        #     'InteractedItems': atts_interactedItem
        # }
        return sample_embeds_llama, sample_atts_llama

    def wrap_prompt_with_soft_tokens_v2(self, prompt, rec_features):
        return

    def forward(self, samples):
        if self.run_mode_ == 'v2':
            return self.forward_v2(samples)
        else:
            raise NotImplementedError("Only forward_v2 is implemented in this version")
        
    def forward_v2(self, samples):
        if self.prompt_list:
            prompt = random.choices(self.prompt_list, weights=[5,5,5,1], k=1)[0] #[1,5,3,1]  #[2,5,3,1]

    @classmethod
    def from_config(cls, cfg):
        rec_model = cfg.get('rec_model',"MF")
        embedding_size = cfg.get("rec_emb_size")
        freeze_rec = cfg.get("freeze_rec",True)
        rec_precision = cfg.get("rec_precision", 'fp16')
        rec_config = cfg.get("rec_config")
        lora_config = cfg.get("lora_config")
        llama_model = cfg.get("llama_model")
        proj_token_num = cfg.get("proj_token_num")
        proj_drop = cfg.get("proj_drop")
        proj_mid = cfg.get("proj_mid_times")
        freeze_proj = cfg.get("freeze_proj")
        freeze_lora = cfg.get("freeze_lora")
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            rec_model=rec_model,
            rec_config=rec_config,
            pretrained_rec = rec_config['pretrained_path'],
            freeze_rec=freeze_rec,
            rec_precision=rec_precision,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            proj_token_num = proj_token_num,
            proj_drop = proj_drop,
            lora_config = lora_config,
            proj_mid = proj_mid,
            freeze_lora=freeze_lora,
            freeze_proj=freeze_proj
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            log_step("Load MiniGPT4Rec Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            log_step("loading message, msg.... {}".format(msg))
            if os.path.exists(rec_config['pretrained_path']) and freeze_rec:
                model.rec_encoder.load_state_dict(torch.load(rec_config['pretrained_path'], map_location="cpu"))

        ans_type = cfg.get('ans_type')
        model.set_answer_type(mode=ans_type)
        model.print_prompt()
        return model

