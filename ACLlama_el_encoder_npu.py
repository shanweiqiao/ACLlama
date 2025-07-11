from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, CTCLoss


from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import (
    WhisperProcessor,
    WhisperModel,
)

######
from transformers.models.whisper.modeling_whisper import WhisperAttention
from transformers.activations import ACT2FN
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
######

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class ACLlamaConfig(LlamaConfig):
    model_type = "ACLlama"
    


def load_whisper(audio_tower_name):
    model = WhisperModel.from_pretrained(
            audio_tower_name,torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to('npu')
    model.config.forced_decoder_ids = None
    return model

class LookBackModule(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.encoder_attn = nn.MultiheadAttention(
            cfg.hidden_size,
            cfg.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        self.atten_layer_norm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, x, wav_feature, bf_shrink_padding_mask):

        residual = x
        x, _ = self.encoder_attn(
            query=x,
            key=wav_feature,
            value=wav_feature,
            key_padding_mask=bf_shrink_padding_mask,
            #attn_mask=padding_mask,
        )
        
        x += residual
        x = self.atten_layer_norm(x)
        
        return x

########
# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper, MBART->WHISPER
class MYEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, rms_norm_eps=1e-03):
        super().__init__()
        self.embed_dim = d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=nhead,
            dropout=dropout,
            config=None,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=rms_norm_eps)
        self.dropout = dropout
        self.activation_fn = ACT2FN["gelu"]
        self.activation_dropout = dropout
        self.fc1 = nn.Linear(self.embed_dim, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        layer_head_mask = None,
        output_attentions = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.bfloat16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
########

class ACLlamaModel(LlamaModel):
    config_class = ACLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(ACLlamaModel, self).__init__(config)

        self.audio_tower = [WhisperModel.from_pretrained(
        config.audio_tower, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to('npu')]
        self.audio_tower[0].config.forced_decoder_ids = None

        if hasattr(config, "adapter_size"):
            #self.down_sampler = Conv1dSubsampler(config.adapter_size, config.hidden_size // 2, config.hidden_size // 2, [5])
            #self.conv1 = nn.Conv1d(1280, config.hidden_size//2, kernel_size=3, stride=2, padding=1)
            #self.conv2 = nn.Conv1d(4096, 4096, kernel_size=3, stride=2, padding=1)
            self.mm_projector1 = nn.Linear(config.adapter_size*2 , config.hidden_size)
            # self.mm_projector1 = nn.Sequential(
            #     nn.Linear(config.adapter_size * 2, config.hidden_size),
            #     nn.LayerNorm(config.hidden_size, eps=1e-5),
            #     nn.Tanh()
            # )
            #self.relu = nn.ReLU()
            #self.mm_projector2 = nn.Linear(config.hidden_size , config.hidden_size)
            asr_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size*2,
                dropout=0.1,
                norm_first=True
            )
            self.lbm =  LookBackModule(config)
            self.out_norm = nn.LayerNorm(config.hidden_size, eps=1e-3)
            self.audio_feature_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.asr_transformer_encoder = nn.TransformerEncoder(asr_encoder_layer, num_layers=1)

        ########
            # self.asr_transformer_encoder = MYEncoderLayer(
            #     d_model=config.hidden_size,
            #     nhead=config.num_attention_heads,
            #     dim_feedforward=config.hidden_size*2,
            #     dropout=0.1,
            # )
            
        self.text_projector = nn.Sequential(nn.Linear(config.hidden_size , config.hidden_size*2),
                                            ACT2FN["gelu"],
                                            nn.Linear(config.hidden_size*2 , config.hidden_size))
        self.text_projector_norm = nn.LayerNorm(config.hidden_size, eps=1e-3)
        
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        del self.layers
        self.act_func = ACT2FN["gelu"]
        ########


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        #######
        input_ids_neg = None,
        attention_mask_neg = None,
        neg_input_caption_ids = None,
        neg_attention_caption_mask = None,
        input_caption_ids = None,
        #######
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaAA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        #######
        # if input_ids_neg is not None:
        #     inputs_embeds_all = self.embed_tokens(torch.cat((input_ids, input_ids_neg), dim=0))
        # else:
        #     inputs_embeds = self.embed_tokens(input_ids)
        #     inputs_embeds_all = None
        if neg_input_caption_ids is not None:
            inputs_embeds_all = self.embed_tokens(torch.cat((input_caption_ids, neg_input_caption_ids), dim=0))
        else:
            inputs_embeds_all = self.embed_tokens(input_caption_ids)
        #######

        # audio_tower = getattr(self, 'audio_tower', None)
        # if audio_tower is not None and (input_ids.shape[1] != 1 or self.training) and audios is not None:
        if (input_ids.shape[1] != 1 or self.training) and audios is not None:
            # audio_tower = audio_tower[0]  # HACK: for FSDP
            audio_list=[]
            
            audio_config = self.audio_tower[0].config
            #with torch.no_grad():
            #    audio_features = audio_tower.encoder(audios).last_hidden_state
            #for audio_feature in audio_features:
            #    audio_feature = audio_feature.unsqueeze(0)
            
            # for audio in audios:
            #     audio = audio.unsqueeze(0)
            #     audio_feature_t = self.audio_tower.encoder(audio).last_hidden_state

            #     audio_feature = audio_feature_t.view(audio_feature_t.shape[0], audio_feature_t.shape[1]//2, 2 * audio_feature_t.shape[2])
            #     audio_feature = self.mm_projector1(audio_feature)
            #     audio_feature = self.asr_transformer_encoder(audio_feature)
            #     audio_feature = self.out_norm(audio_feature)
            #     audio_list.append(audio_feature[0])

            # audio_features = torch.stack(audio_list, dim=0)

 
            ######
            with torch.no_grad():
                audio_feature = self.audio_tower[0].encoder(audios).last_hidden_state.to(audios.dtype)
            audio_feature = audio_feature.view(audio_feature.shape[0], audio_feature.shape[1] // 2, 2 * audio_feature.shape[2])

            audio_feature = self.mm_projector1(audio_feature)
            # audio_feature = F.layer_norm(audio_feature, audio_feature.shape[-1:])  # or nn.LayerNorm
            # audio_feature = self.act_func(audio_feature)

            audio_feature = self.asr_transformer_encoder(audio_feature)
            # audio_feature = self.asr_transformer_encoder(audio_feature, None, None)[0]
            audio_features = self.out_norm(audio_feature)
            # audio_features = self.act_func(audio_feature)

            audio_feature_lengths = attention_mask.int().sum(dim=1)  # shape: (batch_size,)
                        
            audio_features_4_loss = audio_features.clone().permute(0, 2, 1)
            while audio_features_4_loss.size(2) // 2 - 1 > audio_feature_lengths.max():
                audio_features_4_loss = self.avg_pooler(audio_features_4_loss)
            audio_features_4_loss = audio_features_4_loss.permute(0, 2, 1)
            
            #######
            # audio_feature_lengths_neg = attention_mask_neg.int().sum(dim=1)  # shape: (batch_size,)
            audio_feature_lengths_neg = neg_attention_caption_mask.int().sum(dim=1)  # shape: (batch_size,)
            
            # print(f"audio_features_4_loss is : {audio_features_4_loss}")
            
            inputs_embeds = self.text_projector(inputs_embeds)
            # inputs_embeds = F.layer_norm(inputs_embeds, inputs_embeds.shape[-1:])  # or nn.LayerNorm
            inputs_embeds = self.text_projector_norm(inputs_embeds)
            #######

            predict_logits = self.audio_feature_head(audio_features)
            
            new_input_embeds = []
            label_shift = []
            label_extend = -1
            new_input_ids = []
            tokens = predict_logits.argmax(dim=-1)
            shrink_mask = tokens.roll(1) != tokens
            shrink_mask[:,0] = True

            #for i in range(shrink_mask.shape[0]):
            #    m_length = min(torch.nonzero(shrink_mask[i])[-1] + 5, shrink_mask.shape[1])
            #    shrink_mask[i][:m_length]=torch.ones(m_length).to(shrink_mask.device)
                
            lengths = shrink_mask.long().sum(-1)
            shrink_2d = audio_features[shrink_mask]
            #num_patches = audio_features.shape[1]
            num_patches = audio_config.audio_patch_size
            
            l_index=0
            shrink_features = []
            for v, audio_feature, mask in zip(lengths, audio_features, ~shrink_mask):
                shrink_feature = shrink_2d[l_index:l_index+v]
                shrink_feature = self.lbm(shrink_feature, audio_feature, bf_shrink_padding_mask=mask)
                #shrink_feature = self.lbm(shrink_feature, audio_feature, bf_shrink_padding_mask=None)
                shrink_features.append(shrink_feature)
                l_index += v
                
            padded_shrink_features = torch.nn.utils.rnn.pad_sequence(
                shrink_features, 
                batch_first=True, 
                padding_value=0.0
            )
            padding_tensor = torch.zeros((padded_shrink_features.size(0), shrink_mask.size(1) - padded_shrink_features.size(1), padded_shrink_features.size(2))).to(padded_shrink_features.device, dtype=padded_shrink_features.dtype)
            padded_shrink_features = torch.cat((padded_shrink_features, padding_tensor), dim=1)

            audio_feature_lengths = attention_mask.int().sum(dim=1)  # shape: (batch_size,)
                
            # if self.training: 
            #     maxn_length = lengths.max()
            #     label_extend = maxn_length - num_patches
            #     for cur_input_ids, cur_input_embeds, shrink_feature in zip(input_ids, inputs_embeds, shrink_features):
            #         pad_ids = torch.full(size=(maxn_length,), fill_value=audio_config.llm_pad_token_id, dtype=torch.long).to(attention_mask.device)
            #         pad_embeds = self.embed_tokens(pad_ids)
            #         v = shrink_feature.shape[0]
            #         audio_start_token_pos = torch.where(cur_input_ids == audio_config.audio_patch_token)[0][:1]
            #         cur_new_input_id = torch.cat((cur_input_ids[:audio_start_token_pos], cur_input_ids[audio_start_token_pos: audio_start_token_pos+1].repeat(v), cur_input_ids[audio_start_token_pos + num_patches:], pad_ids[:maxn_length - v]), dim=0)                    
            #         cur_new_input_embeds = torch.cat((
            #         cur_input_embeds[:audio_start_token_pos],
            #         shrink_feature,
            #         cur_input_embeds[audio_start_token_pos + num_patches:],pad_embeds[:maxn_length-v]), dim=0)                    
            #         new_input_embeds.append(cur_new_input_embeds)
            #         new_input_ids.append(cur_new_input_id)
            #         label_shift.append(v - num_patches)
                    
            #     input_ids = torch.stack(new_input_ids, dim=0)
            #     attention_mask=input_ids.ne(audio_config.llm_pad_token_id)
            #     inputs_embeds = torch.stack(new_input_embeds, dim=0)
            # else:
            #     for cur_input_ids, cur_input_embeds, shrink_feature in zip(input_ids, inputs_embeds, shrink_features):
            #         v = shrink_feature.shape[0]

            #         audio_start_token_pos = torch.where(cur_input_ids == audio_config.audio_patch_token)[0][:1]
            #         cur_new_input_id = torch.cat((cur_input_ids[:audio_start_token_pos],cur_input_ids[audio_start_token_pos: audio_start_token_pos+1].repeat(v), cur_input_ids[audio_start_token_pos + num_patches:]),dim=0)
            #         cur_new_input_embeds = torch.cat((
            #         cur_input_embeds[:audio_start_token_pos],
            #         shrink_feature,
            #         cur_input_embeds[audio_start_token_pos + num_patches:]), dim=0)                    
            #         new_input_embeds.append(cur_new_input_embeds)
            #         new_input_ids.append(cur_new_input_id)
            #     input_ids = torch.stack(new_input_ids, dim=0)
            #     attention_mask=input_ids.ne(audio_config.llm_pad_token_id)
            #     inputs_embeds = torch.stack(new_input_embeds, dim=0)
            # ######
            
        return_state = {"audio_features": predict_logits}
        #########
        return_state_update = {"audio_feature_lengths": audio_feature_lengths, "audio_features_4_loss": audio_features_4_loss, "inputs_embeds": inputs_embeds, "inputs_embeds_all": inputs_embeds_all, "audio_feature_lengths_neg": audio_feature_lengths_neg, "shrink_mask": shrink_mask, "audio_features_ori_4_loss": audio_features, "shrink_features": padded_shrink_features, "shrink_lengths": lengths}
        return_state.update(return_state_update)    
        #########
        
        return return_state 

class TrainableScalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.scalar = nn.Parameter(torch.log(torch.tensor([init_value])))  # 👈 注意这里是 [init_value]，不是 init_value

    def forward(self):
        return self.scalar[0]  # 返回标量值



class ACLlamaForCausalLM(LlamaForCausalLM):
    config_class = ACLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ACLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        ########
        self.similarity_function = nn.CosineSimilarity(dim=-1)
        # self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 7)))
        self.temperature = TrainableScalar(1.0 / 1e+1)
        # self.temperature = TrainableScalar(1e+1)

        # self.temperature = 1.0
        
        self.text_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-3)
        self.audio_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-3)
        ########

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        asr_targets: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ######
        input_ids_neg: Optional[torch.LongTensor] = None,
        labels_neg: Optional[torch.LongTensor] = None,
        attention_mask_neg: Optional[torch.Tensor] = None,
        audios_neg: Optional[torch.FloatTensor] = None,
        asr_targets_neg: Optional[torch.LongTensor] = None,
        input_caption_ids: Optional[torch.LongTensor] = None,
        caption_labels: Optional[torch.LongTensor] = None,
        asr_caption_targets: Optional[torch.LongTensor] = None,
        attention_caption_mask: Optional[torch.LongTensor] = None,
        neg_input_caption_ids: Optional[torch.LongTensor] = None,
        neg_attention_caption_mask: Optional[torch.LongTensor] = None,
        ######
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        ######
        # input_ids = input_ids.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        # print(f"input_ids_neg is : {input_ids_neg}")
        # if input_ids is not None:
        #     print(f"input_ids 111 is : {input_ids.size()}")
        # if inputs_embeds is not None:
        #     print(f"inputs_embeds 111 is : {inputs_embeds.size()}")
        ######
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            audios=audios,
            #######
            input_ids_neg=input_ids_neg,
            attention_mask_neg=attention_mask_neg,
            neg_input_caption_ids=neg_input_caption_ids,
            neg_attention_caption_mask=neg_attention_caption_mask,
            input_caption_ids=input_caption_ids,
            #######
        )
        
        #######
        audio_feature_lengths = outputs.pop("audio_feature_lengths")
        audio_feature_lengths_neg = outputs.pop("audio_feature_lengths_neg")
        audio_features_4_loss = outputs.pop("audio_features_4_loss")
        inputs_embeds = outputs.pop("inputs_embeds")
        inputs_embeds_all = outputs.pop("inputs_embeds_all")
        shrink_mask = outputs.pop("shrink_mask")
        audio_features_ori_4_loss = outputs.pop("audio_features_ori_4_loss")
        shrink_features = outputs.pop("shrink_features")
        shrink_lengths = outputs.pop("shrink_lengths")
        #######
        
        # logits = self.lm_head(inputs_embeds)

        loss = None
        if labels is not None:
            if asr_targets is not None:
                mask_asr_targets = (asr_targets != IGNORE_TOKEN_ID)
                target_lengths = mask_asr_targets.sum(1)
                input_lengths = torch.full(size=(outputs["audio_features"].shape[0],), fill_value=outputs["audio_features"].shape[1], dtype=torch.long)
                asr_logits = outputs["audio_features"]

                loss_ctc = CTCLoss()

                log_probs = F.log_softmax(asr_logits, dim=-1).transpose(0, 1)
                #print(asr_targets.shape)
                #print(input_lengths, target_lengths)

                with torch.backends.cudnn.flags(enabled=False):
                    loss_asr = F.ctc_loss(
                        log_probs,
                        asr_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.model.audio_tower[0].config.audio_patch_token,
                        reduction='mean',
                        zero_infinity=True,
                    )
            else:
                loss_asr=0
            
            # 这里需要把text改成asr_target
            ######### 
            # # audio_features_4_loss: [B, T, D] → audio mean
            # mean2 = audio_features_4_loss.mean(dim=1)  # [B, D]
            # mean2 = F.normalize(mean2, dim=1)
            
            
            
            
            # torch.set_printoptions(threshold=float('inf'))
            # print(f"\n\n\nfeature :")
            # rank = torch.linalg.matrix_rank(audio_features_ori_4_loss.view(-1, audio_features_ori_4_loss.size(-1)).to(torch.float), tol=1e-5)
            # print(f"mean2 (audio) rank is : {rank}, and shape is : {audio_features_ori_4_loss.view(-1, audio_features_ori_4_loss.size(-1)).shape}")
            # rank = torch.linalg.matrix_rank(inputs_embeds.view(-1, inputs_embeds.size(-1)).to(torch.float), tol=1e-5)
            # print(f"masked_mean1 (text) rank is : {rank}, and shape is : {inputs_embeds.view(-1, inputs_embeds.size(-1)).shape}")
            
            # U, S, V = torch.linalg.svd(audio_features_ori_4_loss.view(-1, audio_features_ori_4_loss.size(-1)).to(torch.float), full_matrices=False)
            # explained_var_ratio = S**2 / torch.sum(S**2)
            # print(f"mean2 (audio) rank is : {explained_var_ratio}")
            # U, S, V = torch.linalg.svd(inputs_embeds.view(-1, inputs_embeds.size(-1)).to(torch.float), full_matrices=False)
            # explained_var_ratio = S**2 / torch.sum(S**2)
            # print(f"masked_mean1 (text) rank is : {explained_var_ratio}")
            
            """
            audio feature before ctc head
            """
            assert audio_features_ori_4_loss.shape[:2] == shrink_mask.shape[:2], f"Shape mismatch in ACLlama_el_encoder.py: {audio_features_ori_4_loss.shape[:2]} vs {shrink_mask.shape[:2]}"
            shrink_mask = shrink_mask.unsqueeze(-1).type_as(audio_features_ori_4_loss).detach()
            masked_sum2 = (audio_features_ori_4_loss * shrink_mask).sum(dim=1)  # [B, D]
            masked_mean2 = masked_sum2 / (shrink_mask.sum(dim=1) + 1e-8)  # [B, D]
            # mean2 = F.normalize(masked_mean2, dim=1)
            mean2 = self.audio_layer_norm(masked_mean2)
            
            
            
            # """
            # audio feature after lbm
            # """
            # mask2 = torch.arange(shrink_features.size(1), device=shrink_features.device)[None, :] < shrink_lengths[:, None]  # [2B, L]
            # mask2 = mask2.unsqueeze(-1).type_as(shrink_features)  # [2B, L, 1]
            # masked_sum2 = (shrink_features * mask2).sum(dim=1)  # [2B, D]
            # mean2 = masked_sum2 / (mask2.sum(dim=1) + 1e-8)  # [2B, D]
            # mean2 = F.normalize(mean2, dim=1)
            # # mean2 = []
            # # for item in shrink_features:
            # #     mean2.append(item.mean(dim=0, keepdim=True))
            # # print(mean2[0].size())
            # # mean2 = torch.cat(mean2, dim=0)
            # # print(f"mean2 is : {mean2.size()}")
            # # mean2 = F.normalize(masked_mean2, dim=1)

            # inputs_embeds: [2B, L, D]
            # text attention mask（注意，扩展到 2B）
            if inputs_embeds_all is not None:
                inputs_embeds = inputs_embeds_all

            # 手动构造 attention mask（注意长度也扩展）
            text_lengths = torch.cat([audio_feature_lengths, audio_feature_lengths_neg], dim=0)  # [2B]
            mask1 = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device)[None, :] < text_lengths[:, None]  # [2B, L]
            mask1 = mask1.unsqueeze(-1).type_as(inputs_embeds)  # [2B, L, 1]

            # masked mean pooling
            masked_sum1 = (inputs_embeds * mask1).sum(dim=1)  # [2B, D]
            masked_mean1 = masked_sum1 / (mask1.sum(dim=1) + 1e-8)  # [2B, D]
            # masked_mean1 = F.normalize(masked_mean1, dim=1)
            masked_mean1 = self.text_layer_norm(masked_mean1)

            # print(f"\n\n\nfinal is :")
            # rank = torch.linalg.matrix_rank(mean2.to(torch.float), tol=1e-5)
            # print(f"mean2 (audio) rank is : {rank}, and shape is : {mean2.shape}")
            # rank = torch.linalg.matrix_rank(masked_mean1.to(torch.float), tol=1e-5)
            # print(f"masked_mean1 (text) rank is : {rank}, and shape is : {masked_mean1.shape}")
            
            # U, S, V = torch.linalg.svd(mean2, full_matrices=False)
            # explained_var_ratio = S**2 / torch.sum(S**2)
            # print(f"mean2 (audio) rank is : {explained_var_ratio}")
            # U, S, V = torch.linalg.svd(masked_mean1, full_matrices=False)
            # explained_var_ratio = S**2 / torch.sum(S**2)
            # print(f"masked_mean1 (audio) rank is : {explained_var_ratio}")

            # logits: audio anchor → text pos+neg
            logits = torch.matmul(mean2, masked_mean1.T)  # [B, 2B]
            labels = torch.arange(mean2.size(0), device=mean2.device)  # [B]
            # loss = F.cross_entropy(logits / self.temperature, labels)
            # loss = F.cross_entropy(logits / self.temperature(), labels)
            # loss = F.cross_entropy(logits * 0.5 * torch.exp(self.temperature()), labels)
            loss = F.cross_entropy(logits * torch.exp(self.temperature()), labels)
            loss_reserve = F.cross_entropy(logits.T[:logits.size(0), :] * torch.exp(self.temperature()), labels)
            loss = (loss + loss_reserve) / 2


            
            # def visualize_and_save_contrastive_features(audio_features, text_features, save_path="contrastive_tsne.png"):

            #     features = torch.cat([audio_features, text_features], dim=0).detach().cpu().numpy()
            #     n_audio = audio_features.shape[0]
            #     n_samples = features.shape[0]
            #     perplexity = min(30, max(5, n_samples // 3))

            #     tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, init='random', random_state=42)
            #     features_2d = tsne.fit_transform(features)

            #     # 拆分为 audio/text 部分
            #     audio_tsne = features_2d[:n_audio]
            #     text_tsne = features_2d[n_audio:]

            #     # 画图
            #     plt.figure(figsize=(8, 8))
            #     plt.scatter(audio_tsne[:, 0], audio_tsne[:, 1], color='red', alpha=0.6, label='Audio')
            #     plt.scatter(text_tsne[:, 0], text_tsne[:, 1], color='blue', alpha=0.6, label='Text')

            #     plt.title(f"Contrastive Feature t-SNE (perplexity={perplexity})")
            #     plt.legend()
            #     plt.grid(True)
            #     plt.savefig(save_path, dpi=300)
            #     plt.close()

            #     print(f"[✔] 图像已保存到: {save_path}")

            # visualize_and_save_contrastive_features(mean2, masked_mean1[:mean2.size(0), :], save_path="/data/s50042884/my_code/ACLlama_A100_right/ACLlama/contrastive_features_tsne.png")
            #########

            # exit(0)

            # #########
            # # print(f"audio_features_4_loss is : {audio_features_4_loss}")
            # # 创建 mask1: [B, 512]
            # mask1 = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device)[None, :] < audio_feature_lengths[:, None]
            # mask1 = mask1.unsqueeze(-1)  # [B, 512, 1]

            # # masked mean
            # masked_sum1 = (inputs_embeds * mask1).sum(dim=1)  # [B, 3072]
            # masked_mean1 = masked_sum1 / audio_feature_lengths.unsqueeze(1)     # [B, 3072]

            # # 直接对 encoder_embedding2 做 mean
            # mean2 = audio_features_4_loss.mean(dim=1)  # 假设它无 padding
            # # print(f"masked_mean1 is : {masked_mean1}")
            # # print(f"mean2 is : {mean2}")

            # masked_mean1 = F.normalize(masked_mean1, dim=1)
            # mean2 = F.normalize(mean2, dim=1)

            # ######
            # # === Step 2: 构造 global 对比相似度矩阵 ===
            # # similarity_matrix: [B, B]
            # similarity_matrix = self.similarity_function(
            #     masked_mean1.unsqueeze(1),  # [B, 1, D]
            #     mean2.unsqueeze(0)          # [1, B, D]
            # )
            # # === Step 3: InfoNCE loss ===
            # logits = similarity_matrix / 1.0  # [B, B]
            # log_probs = nn.LogSoftmax(dim=1)(logits)
            # # print(f"log_probs is : {log_probs}")
            # loss_contrastive = -log_probs.diagonal().mean()
            # # print(f"loss_contrastive is : {loss_contrastive}")

            # # inputs_embeds_filter = inputs_embeds[:, :audio_features_4_loss.size(1), :]
            
            # # mask1 = torch.arange(inputs_embeds_filter.size(1), device=inputs_embeds_filter.device)[None, :] < audio_feature_lengths[:, None]
            # # mask1 = mask1.unsqueeze(-1)  # [B, 512, 1]
            
            # # batch_size = audio_features_4_loss.shape[1]
            # # length = audio_features_4_loss.shape[0]
            # # feature_dim = audio_features_4_loss.shape[2]
            # # similarity = self.similarity_function(inputs_embeds_filter.mean(-1), audio_features_4_loss.mean(-1)).mean(-1)
            # # anchor_dot_contrast = self.similarity_function(inputs_embeds_filter.expand((length, length, batch_size, feature_dim)).transpose(0,2).to(torch.float32),
            # # audio_features_4_loss.expand((length, length, batch_size, feature_dim)).transpose(0,2).to(torch.float32))

            # # loss_contrastive = -nn.LogSoftmax(1)(anchor_dot_contrast.to(audio_features_4_loss.dtype)).diagonal().sum()
            # #########


            # #########
            # inputs_embeds_filter = inputs_embeds[:, :audio_features_4_loss.size(1), :]
            # mask1 = torch.arange(inputs_embeds_filter.size(1), device=inputs_embeds_filter.device)[None, :] < audio_feature_lengths[:, None]

            # with torch.cuda.amp.autocast(enabled=False):  # 禁用 autocast
            #     # audio_features_4_loss[~mask1] = 0
            #     # inputs_embeds_filter[~mask1] = 0
                
            #     # audio_features = audio_features_4_loss.mean(1).to(torch.float32)
            #     # text_features = inputs_embeds_filter.mean(1).to(torch.float32)
                
            #     mask1 = mask1.unsqueeze(-1)  # shape: [B, L, 1]
            #     len_x = mask1.sum(dim=1)  # number of valid positions per sample [B, 1]

            #     # mask 并求mean 去length
            #     audio_features_4_loss = audio_features_4_loss * mask1  # masked-out positions will become 0
            #     sum_audio_features_4_loss = audio_features_4_loss.sum(dim=1)  # sum over valid positions
            #     audio_features = sum_audio_features_4_loss / (len_x + 1e-8)  # shape: [B, D]
                
            #     # print(f"audio_features is 111 : {audio_features}")
                
            #     inputs_embeds_filter = inputs_embeds_filter * mask1  # masked-out positions will become 0
            #     sum_inputs_embeds_filter = inputs_embeds_filter.sum(dim=1)  # sum over valid positions
            #     text_features = sum_inputs_embeds_filter / (len_x + 1e-8)  # shape: [B, D]

            #     # print(f"text_features is 111 : {text_features}")

            #     # normalized features
            #     audio_features = audio_features / audio_features.norm(dim=1, keepdim=True).clamp(min=1e-8)
            #     text_features = text_features / text_features.norm(dim=1, keepdim=True).clamp(min=1e-8)
            #     # audio_features_4_loss = F.normalize(audio_features_4_loss, dim=1)
            #     # text_features = F.normalize(text_features, dim=1)

            #     # print(f"audio_features is 222 : {audio_features}")
            #     # print(f"text_features is 222 : {text_features}")

            #     # cosine similarity as logits
            #     logit_scale = self.logit_scale.exp()
            #     logits_per_audio = logit_scale * audio_features @ text_features.t()
            #     logits_per_text = logits_per_audio.t()

            #     # print(f"logits_per_audio is : {logits_per_audio}")

            #     labels = torch.arange(audio_features.size(0), device=logits_per_audio.device)
            #     loss_fn = nn.CrossEntropyLoss()
            #     loss_i = loss_fn(logits_per_audio, labels)
            #     loss_t = loss_fn(logits_per_text, labels)
            #     loss_contrastive = (loss_i + loss_t)/2
                
            #     # print(f"loss_i is : {loss_i}")
            #     # print(f"loss_t is : {loss_t}")
            # ########
            
            # ########
            # alpha = 0.5
            # loss_mse = F.mse_loss(audio_embed, text_embed.detach())
            # loss = loss_contrastive + alpha * loss_mse
            # ########
            
            loss = loss + loss_asr * 0.3
            # loss = loss_contrastive

        return CausalLMOutputWithPast(
            loss=loss,
            # logits=asr_logits,
            logits=logits.unsqueeze(1),
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        
        print(kwargs.keys())
        exit(0)
        
        model_inputs.update({"audios": kwargs["audios"]} if "audios" in kwargs.keys() else {})
        
        ########
        model_inputs.update({"input_ids_neg": kwargs["input_ids_neg"]} if "input_ids_neg" in kwargs.keys() else {})
        model_inputs.update({"labels_neg": kwargs["labels_neg"]} if "labels_neg" in kwargs.keys() else {})
        model_inputs.update({"attention_mask_neg": kwargs["attention_mask_neg"]} if "attention_mask_neg" in kwargs.keys() else {})
        model_inputs.update({"audios_neg": kwargs["audios_neg"]} if "audios_neg" in kwargs.keys() else {})
        model_inputs.update({"asr_targets_neg": kwargs["asr_targets_neg"]} if "asr_targets_neg" in kwargs.keys() else {})
        
        model_inputs.update({"input_caption_ids": kwargs["input_caption_ids"]} if "input_caption_ids" in kwargs.keys() else {})
        model_inputs.update({"caption_labels": kwargs["caption_labels"]} if "caption_labels" in kwargs.keys() else {})
        model_inputs.update({"asr_caption_targets": kwargs["asr_caption_targets"]} if "asr_caption_targets" in kwargs.keys() else {})
        model_inputs.update({"attention_caption_mask": kwargs["attention_caption_mask"]} if "attention_caption_mask" in kwargs.keys() else {})
        model_inputs.update({"neg_input_caption_ids": kwargs["neg_input_caption_ids"]} if "neg_input_caption_ids" in kwargs.keys() else {})
        model_inputs.update({"neg_attention_caption_mask": kwargs["neg_attention_caption_mask"]} if "neg_attention_caption_mask" in kwargs.keys() else {})
        ########

        return model_inputs


AutoConfig.register("ACLlama", ACLlamaConfig)
AutoModelForCausalLM.register(ACLlamaConfig, ACLlamaForCausalLM)
