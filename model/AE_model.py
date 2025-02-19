import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from functools import partial
from model.base_model import BaseModel
from model.embeddings import POIembedding, TimeEmbedding, SpaceEmbedding
from utils.util import sequence_mask
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

    
class MambaEncoder(BaseModel):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        is_decoder: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.d_model = d_model
        self.is_decoder = is_decoder

        self.embeddings = POIembedding(vocab_size, d_model)  # POI embeddings
        self.time_embeddings = TimeEmbedding(24, 32)
        self.distance_embeddings = SpaceEmbedding(32)

        self.decoder_input1 = nn.Linear(2*d_model + 3*32, d_model + 3*32)
        self.decoder_input2 = nn.Linear(2*(d_model + 3*32), d_model + 3*32)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model + 3*32,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model + 3*32, eps=norm_epsilon, **factory_kwargs
        )

        self.pooling = nn.AdaptiveAvgPool1d(1) 

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, context, condition=None, inference_params=None, **mixer_kwargs):
        if (self.is_decoder):
            #是否可以考虑额外引入一个condition信息，将POI个数编码成32维度(用来替换time），地理距离编码成2*32维度？
            tensor = self.embeddings(input_ids)
            if (condition is None):
                time_embedding = torch.zeros((input_ids.size(0), input_ids.size(1), 32), device=input_ids.device)
                space_embedding = torch.zeros((input_ids.size(0), input_ids.size(1), 2*32), device=input_ids.device)
                input = torch.cat([tensor, time_embedding, space_embedding], dim=-1)
                latent_vector = context.unsqueeze(1).repeat(1, input.size(1), 1)  # shape: [batch_size, seq_len, d_model]
                hidden_states = self.decoder_input1(torch.cat([tensor, latent_vector], dim=-1))
            else:
                poi_count, distance1, distance2 = condition
                poi_count = poi_count.unsqueeze(1).repeat(1, input_ids.size(1), 1)  # shape: [batch_size, seq_len, 32]
                distance1 = distance1.unsqueeze(1).repeat(1, input_ids.size(1), 1)  # shape: [batch_size, seq_len, 1]
                distance2 = distance2.unsqueeze(1).repeat(1, input_ids.size(1), 1)  # shape: [batch_size, seq_len, 1]
                space_embedding = self.distance_embeddings(distance1, distance2)
                input = torch.cat([tensor, poi_count, space_embedding], dim=-1)
                latent_vector = context.unsqueeze(1).repeat(1, input.size(1), 1)  # shape: [batch_size, seq_len, d_model]
                hidden_states = self.decoder_input2(torch.cat([input, latent_vector], dim=-1))
            #latent_vector = context.unsqueeze(1).repeat(1, input.size(1), 1)  # shape: [batch_size, seq_len, d_model]
            #hidden_states = input + latent_vector
        else:
            tensor = self.embeddings(input_ids)  # shape: [batch_size, length, embedding_size]
            time_t = self.time_embeddings(context[0])  # context[0] -> time indices, shape: [batch_size, length, time_embedding_size]
            space = self.distance_embeddings(context[1], context[2])    #shape: [batch_size, length, space_embedding_size]
            # Concatenate POI embeddings, time embeddings, and space features
            tensor = torch.cat([tensor, time_t], dim=-1)  # Concatenate along the last dimension
            hidden_states = torch.cat([tensor, space], dim=-1)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        # pooling or average or last token
        averaged_hidden = hidden_states.mean(dim=1)
        # pooled_hidden = self.pooling(hidden_states.transpose(1, 2)).squeeze(-1)
        # last_hidden = hidden_states[:, -1, :]
        if(self.is_decoder):
            out_hidden = hidden_states
        else:
            out_hidden = averaged_hidden
        return out_hidden
    
class MambaDecoder(BaseModel):
    def __init__(self, d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False, 
        device=None, 
        dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = MambaEncoder(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            norm_epsilon=norm_epsilon,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            rms_norm=rms_norm,
            is_decoder=True,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model + 3*32, vocab_size, bias=False)

    def forward(self, input_ids, latent_vector, condition=None, dis_score=None, mask=None):
        hidden_states = self.backbone(input_ids, latent_vector, condition)  # Now hidden_states shape: [batch_size, seq_len, d_model]

        # Step 6: Compute logits for each token in the sequence
        lm_logits = self.lm_head(hidden_states)  # shape: [batch_size, seq_len, vocab_size]
        '''
        dis_score = dis_score.unsqueeze(1).expand(-1, lm_logits.shape[1], -1)   # shape: [batch_size, seq_len, vocab_size]
        re_logits = torch.exp(lm_logits) * dis_score
        re_logits = re_logits * mask.float()
        re_logits = torch.clamp(re_logits, min=0)
        sump = re_logits.sum(1)  # [bs]
        sump = torch.where(sump <= 0, torch.tensor(1., device=self.device), sump)
        p = p / sump[:, None].repeat(1, n_sensingtasks_1)
        log_p = torch.log(torch.where(p <= 0, torch.tensor(1e-10, device=self.device), p))
        '''
        probs = F.softmax(lm_logits, dim=-1)
        predict = torch.argmax(probs, dim=-1)

        return lm_logits, predict

class MambaAEModel(BaseModel):
    def __init__(self, d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False, 
        device=None, 
        dtype=None):
        super().__init__()
        self.encoder = MambaEncoder(d_model=d_model, n_layer=n_layer, d_intermediate=d_intermediate, vocab_size=vocab_size, ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm, is_decoder=False, initializer_cfg=initializer_cfg, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32, device=device)
        self.decoder = MambaDecoder(d_model=d_model, n_layer=n_layer, d_intermediate=d_intermediate, vocab_size=vocab_size, ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm, initializer_cfg=initializer_cfg, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32, device=device)
        self.device = device

    def forward(self, X, context, target_sequence_length, max_target_sequence_length, batch_size, go_int, pad_index):
        latent_vector = self.encoder(X, context)  # Get latent vector from encoder
        input = torch.full((batch_size, 1), go_int, device=self.device)  # shape: [batch_size, 1]
        outputs = torch.zeros(max_target_sequence_length, batch_size, self.decoder.lm_head.out_features, device=self.device)
        generated_tokens = torch.full((batch_size, max_target_sequence_length), pad_index, dtype=torch.int64, device=self.device)

        # Sequentially generate each token
        for i in range(max_target_sequence_length):
            output_logits, predict = self.decoder(input, latent_vector)
            output_logits = output_logits.transpose(0,1) #shape: [length, batch_size, vocab_size]
            predict = predict.transpose(0,1)    #shape: [length, batch_size]
            outputs[i] = output_logits[-1]  # Only store the last generated logits

            # Get the predicted token and update the input for the next step
            next_token = predict[-1]
            generated_tokens[:, i] = next_token

            # Update input with the new token embedding for the next timestep
            input = torch.cat([input, next_token.unsqueeze(1)], dim=1)
        outputs = outputs.transpose(0,1)
        
        for j in range(len(target_sequence_length)):
            leng = target_sequence_length[j]
            if (leng < max_target_sequence_length):
                generated_tokens[j, leng:] = pad_index
        return outputs, generated_tokens
