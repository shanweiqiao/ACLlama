# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import librosa
from transformers import Trainer, GPTQConfig
from transformers.integrations import deepspeed
from transformers import WhisperProcessor
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prettytable import PrettyTable
from accelerate.utils import DistributedType
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import BitsAndBytesConfig
#from ACLlama_align import ACLlamaForCausalLM
# from ACLlama_el import ACLlamaForCausalLM
from ACLlama_el_encoder_stage2_unfenc import ACLlamaForCausalLM
import string
from tqdm import tqdm
import orjson
from torch import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

#########
import random
from functools import partial
from typing import Dict, List, Union
import torchaudio
from datasets import load_dataset, load_from_disk
random.seed(42)
#########

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

MAX_ASR_LENGTH = 200
NEG_CONTRASTIVE_BATCH = 11
DATALOADER_NUM_WORKERS = 16

@dataclass
class ModelArguments:
    text_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.1-8B-Instruct")
    audio_model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    adapter_size: int = field(default=1280, metadata={"help":"The size of adapter input."})


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    dataloader_num_workers: int = DATALOADER_NUM_WORKERS


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['o_proj', 'k_proj', 'q_proj', 'v_proj']
        # default_factory=lambda: ['k_proj', 'q_proj', 'v_proj']
    )
    # lora_target_modules = None
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language."

) -> Dict:

    # im_start = tokenizer.im_start_id
    # im_end = tokenizer.im_end_id

    DEFAULT_AUDIO_PATCH_TOKEN = "<audio_patch>"
    audio_placeholder = DEFAULT_AUDIO_PATCH_TOKEN * CONFIG.audio_token_len
    audio_placeholder = "\n"+audio_placeholder
    audio_placeholder_ids = tokenizer(audio_placeholder).input_ids

    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids
    _user = tokenizer('user').input_ids
    _assistant = tokenizer('assistant').input_ids


    def convert_to_id(source):
        input_id,target = [], []
        system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system_message).input_ids + [eot_id]
        input_id += system
        #input_id += audio_placeholder_ids
        target += [IGNORE_TOKEN_ID] * len(input_id)
        prefix_index = 0
        prefix_index += len(input_id)
        assert len(input_id) == len(target)
        for j, item in enumerate(source):
            role = item["from"]
            value = item["value"]
            if role == 'user':
                #_input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [eot_id]
                _input_id = [start_header_id] + _user + [end_header_id] + audio_placeholder_ids + tokenizer(value).input_ids + [eot_id]
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
                audio_path = item["audio"] if "audio" in item.keys() else None
                prefix_index+=(len(_input_id))
            elif role == 'assistant':
                _input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [eot_id]
                _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(_assistant) + \
                          [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(nl_tokens) + tokenizer(value).input_ids + [eot_id]
                prefix_index+=(len([start_header_id] + _assistant + [end_header_id] + nl_tokens))
                #  TODO change the process code of ASR data
                asr_value = item["transcription"]  #value.split(":")[-1].replace(".","")
                asr_id = tokenizer(asr_value).input_ids[1:]
                #asr_targets.append(asr_id+[IGNORE_TOKEN_ID] * (MAX_ASR_LENGTH - len(asr_id)))
                asr_target=asr_id+[IGNORE_TOKEN_ID] * (MAX_ASR_LENGTH - len(asr_id))
            else:
                raise NotImplementedError
            input_id += _input_id
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        return input_id,target,audio_path,asr_target

    # Apply prompt templates
    input_ids, audio_paths, targets = [], [], []
    asr_targets = []

    for source in sources:
        input_id,target = [], []
        system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system_message).input_ids + [eot_id]
        input_id += system
        #input_id += audio_placeholder_ids
        target += [IGNORE_TOKEN_ID] * len(input_id)
        prefix_index = 0
        prefix_index += len(input_id)
        assert len(input_id) == len(target)
        for j, item in enumerate(source):
            role = item["from"]
            value = item["value"]
            if role == 'user':
                #_input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [eot_id]
                _input_id = [start_header_id] + _user + [end_header_id] + audio_placeholder_ids + tokenizer(value).input_ids + [eot_id]
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
                audio_path = item["audio"] if "audio" in item.keys() else None
                prefix_index+=(len(_input_id))
            elif role == 'assistant':
                _input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [eot_id]
                _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(_assistant) + \
                          [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(nl_tokens) + tokenizer(value).input_ids + [eot_id]
                prefix_index+=(len([start_header_id] + _assistant + [end_header_id] + nl_tokens))
                #  TODO change the process code of ASR data
                asr_value = item["transcription"] if "transcription" in item.keys() else None
                if asr_value:
                    asr_id = tokenizer(asr_value).input_ids[1:]
                    asr_targets.append(asr_id+[IGNORE_TOKEN_ID] * (MAX_ASR_LENGTH - len(asr_id)))
            else:
                raise NotImplementedError
            input_id += _input_id
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        audio_paths.append(audio_path)
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    if len(asr_targets) > 0:
        asr_targets = torch.tensor(asr_targets, dtype=torch.int)
    #print("Finish process data, total:",len(input_ids))    
    return dict(
        input_ids=input_ids,
        labels=targets,
        audio_paths=audio_paths,
        asr_targets=asr_targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

########
def preprocess_caption(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        caption_prompt = None,
        asr_prompt = None,
        system_message: str = "You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language."
) -> Dict:

    # im_start = tokenizer.im_start_id
    # im_end = tokenizer.im_end_id

    DEFAULT_AUDIO_PATCH_TOKEN = "<audio_patch>"
    audio_placeholder = DEFAULT_AUDIO_PATCH_TOKEN * CONFIG.audio_token_len
    audio_placeholder = "\n"+audio_placeholder
    audio_placeholder_ids = tokenizer(audio_placeholder).input_ids

    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids
    _user = tokenizer('user').input_ids
    _assistant = tokenizer('assistant').input_ids

    # Apply prompt templates
    input_ids, audio_arrays, targets, input_caption_ids, caption_targets = [], [], [], [], []
    asr_targets = []
    asr_caption_targets = []
    caption_sampling_rates = []
    caption_text_masks = []
    
    for source in sources:
        input_id,target = [], []
        input_caption_id,caption_target = [], []
        
        system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system_message).input_ids + [eot_id]
        input_id += system
        input_caption_id += system

        ########
        caption_text_mask = [0] * len(system)
        ########

        target += [IGNORE_TOKEN_ID] * len(input_id)
        caption_target += [IGNORE_TOKEN_ID] * len(input_caption_id)
        
        prefix_index = 0
        prefix_index += len(input_id)
        assert len(input_id) == len(target)
        
        prefix_caption_index = 0
        prefix_caption_index += len(input_caption_id)
        assert len(input_caption_id) == len(caption_target)
        
        
        asr_prompt = random.choice(asr_prompt)
        caption_prompt = random.choice(caption_prompt)
        audio_array = source["audio"]["array"] if "audio" in source.keys() else None
        caption_sampling_rates.append(source["audio"]["sampling_rate"])

        # asr question
        _input_id = [start_header_id] + _user + [end_header_id] + audio_placeholder_ids + tokenizer(asr_prompt["question"]).input_ids + [eot_id]
        _target = [IGNORE_TOKEN_ID] * len(_input_id)
        prefix_index+=(len(_input_id))

        input_id += _input_id
        target += _target
            
        # asr answer
        _input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens + tokenizer(asr_prompt["reply_prompt"].strip(" ").strip(".").strip(" ") + source["speech_caption"]).input_ids + [eot_id] # 这个里面应该有个回答的前缀
        _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(_assistant) + \
                    [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(nl_tokens) + tokenizer(asr_prompt["reply_prompt"].strip(" ").strip(".").strip(" ") + source["speech_caption"]).input_ids + [eot_id]
        prefix_index+=(len([start_header_id] + _assistant + [end_header_id] + nl_tokens))
        #  TODO change the process code of ASR data
        asr_value = source["speech_caption"] if "speech_caption" in source.keys() else None
        if asr_value:
            asr_id = tokenizer(asr_value).input_ids[1:]
            asr_targets.append(asr_id+[IGNORE_TOKEN_ID] * (MAX_ASR_LENGTH - len(asr_id)))
                    
        input_id += _input_id
        target += _target
        
        # caption question
        _input_id = [start_header_id] + _user + [end_header_id] + audio_placeholder_ids + tokenizer(caption_prompt["question"].replace("<|AUDIO|>", "").strip()).input_ids + [eot_id]
        _target = [IGNORE_TOKEN_ID] * len(_input_id)
        prefix_index+=(len(_input_id))

        input_caption_id += _input_id
        caption_target += _target
        ########
        caption_text_mask += [0] * len(_input_id)
        ########

        # caption answer
        _input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens + tokenizer(caption_prompt["reply_prompt"].strip(" ").strip(".").strip(" ") + source["audio_caption"]).input_ids + [eot_id]
        _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(_assistant) + \
                    [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(nl_tokens) + tokenizer(caption_prompt["reply_prompt"].strip(" ").strip(".").strip(" ") + source["audio_caption"]).input_ids + [eot_id]
        prefix_index+=(len([start_header_id] + _assistant + [end_header_id] + nl_tokens))
        
        ########
        # print([start_header_id] + _assistant + [end_header_id] + nl_tokens + [tokenizer.convert_tokens_to_ids(item) for item in tokenizer.tokenize(caption_prompt["reply_prompt"].strip(" ").strip(".").strip(" "))])
        # print([tokenizer.convert_tokens_to_ids(item) for item in tokenizer.tokenize(source["audio_caption"])] + [eot_id])
        caption_text_mask += [0] * (len([start_header_id] + _assistant + [end_header_id] + nl_tokens + tokenizer(caption_prompt["reply_prompt"].strip(" ").strip(".").strip(" ")).input_ids) - 1)
        ########
        
        #  TODO change the process code of ASR data
        asr_value = source["audio_caption"] if "audio_caption" in source.keys() else None
        if asr_value:
            asr_id = tokenizer(asr_value).input_ids[1:]
            asr_caption_targets.append(asr_id+[IGNORE_TOKEN_ID] * (MAX_ASR_LENGTH - len(asr_id)))
        
        input_caption_id += _input_id
        caption_target += _target
        
        ########
        caption_text_mask += [1] * (len(input_caption_id) - len(caption_text_mask) - 1) + [0]
        ########
        
        assert len(input_id) == len(target)
        assert len(input_caption_id) == len(caption_target)
        
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_caption_id += [tokenizer.pad_token_id] * (max_len - len(input_caption_id))
        caption_target += [IGNORE_TOKEN_ID] * (max_len - len(caption_target))
        
        ########
        caption_text_mask += [0] * (max_len - len(caption_text_mask))
        caption_text_masks.append(caption_text_mask[:max_len])
        ########
        
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        input_caption_ids.append(input_caption_id[:max_len])
        caption_targets.append(caption_target[:max_len])
        
        audio_arrays.append(audio_array)
        
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    input_caption_ids = torch.tensor(input_caption_ids, dtype=torch.int)
    caption_targets = torch.tensor(caption_targets, dtype=torch.int)
    
    ########
    caption_text_masks = torch.tensor(caption_text_masks, dtype=torch.bool)
    ########
    
    if len(asr_targets) > 0:
        ######
        asr_targets = asr_targets[:MAX_ASR_LENGTH]
        ######

        asr_targets = torch.tensor(asr_targets, dtype=torch.int)

    if len(asr_caption_targets) > 0:
        asr_caption_targets = torch.tensor(asr_caption_targets, dtype=torch.int)
       
    return dict(
        input_ids=input_ids,
        labels=targets,
        audio_arrays=audio_arrays,
        asr_targets=asr_targets,
        input_caption_ids=input_caption_ids,
        caption_labels=caption_targets,
        asr_caption_targets=asr_caption_targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        attention_caption_mask=input_caption_ids.ne(tokenizer.pad_token_id),
        caption_sampling_rates=caption_sampling_rates,
        caption_text_masks=caption_text_masks,
    )
########


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, system_message=None, audio_processor_path=None, caption_data=None):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        
        if caption_data is not None:
            self.raw_data_caption = caption_data
            with open("/data/s50042884/my_code/data/libri_train_qa_prompt.json", "r", encoding="utf-8") as asr_p_f:
                self.asr_prompt = json.load(asr_p_f)
            with open("/data/s50042884/my_code/data/audio_caption_prompt_dynamic_replies.json", "r", encoding="utf-8") as caption_p_f:
                self.caption_prompt = json.load(caption_p_f)
        else:
            self.raw_data_caption = None
        
        if self.raw_data_caption is not None:
            sources = [self.raw_data_caption[caption_data_id] for caption_data_id in range(self.raw_data_caption.num_rows)]
            data_dict = preprocess_caption(sources, tokenizer, max_len, asr_prompt=self.asr_prompt, caption_prompt=self.caption_prompt)
            self.audio_processor = WhisperProcessor.from_pretrained(audio_processor_path, dtype=torch.bfloat16)
            self.mask_id = tokenizer.pad_token_id

            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
            self.audio_arrays = data_dict["audio_arrays"]
            self.attention_mask = data_dict["attention_mask"]
            self.asr_targets = data_dict["asr_targets"]
            self.input_caption_ids = data_dict["input_caption_ids"]
            self.caption_labels = data_dict["caption_labels"]
            self.asr_caption_targets = data_dict["asr_caption_targets"]
            self.attention_caption_mask = data_dict["attention_caption_mask"]
            self.caption_sampling_rates = data_dict["caption_sampling_rates"]
            
        else:
            sources = [example["conversations"] for example in raw_data]
            data_dict = preprocess(sources, tokenizer, max_len)
            self.audio_processor = WhisperProcessor.from_pretrained(audio_processor_path)
            self.mask_id = tokenizer.pad_token_id

            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
            self.audio_paths = data_dict["audio_paths"]
            self.attention_mask = data_dict["attention_mask"]
            self.asr_targets = data_dict["asr_targets"]


    def __len__(self):
        if self.raw_data_caption is not None:
            return self.raw_data_caption.num_rows
        else:
            return len(self.input_ids)

    def _process_item(self, i):
        
        if self.raw_data_caption is not None:
            audio = librosa.resample(self.audio_arrays[i], orig_sr=self.caption_sampling_rates[i], target_sr=CONFIG.sampling_rate)
            audio_feat = self.audio_processor(audio, sampling_rate=CONFIG.sampling_rate, return_tensors="pt").input_features
            audio_feat = audio_feat.squeeze(0).to(CONFIG.device, dtype=torch.bfloat16)
            
        else:
            audio, _ = librosa.load(self.audio_paths[i], sr=CONFIG.sampling_rate)
            audio_feat = self.audio_processor(audio, sampling_rate=CONFIG.sampling_rate, return_tensors="pt").input_features
            audio_feat = audio_feat.squeeze(0).to(CONFIG.device, dtype=torch.bfloat16)

        if self.raw_data_caption is not None:
            return dict(
                input_ids=self.input_ids[i],
                labels=self.labels[i],
                attention_mask=self.attention_mask[i],
                audios=audio_feat,
                asr_targets=self.asr_targets[i],
                input_caption_ids=self.input_caption_ids[i],
                caption_labels=self.caption_labels[i],
                asr_caption_targets=self.asr_caption_targets[i],
                attention_caption_mask=self.attention_caption_mask[i],
                caption_text_masks=self.caption_text_masks[i],
            )
        else:
            return dict(
                input_ids=self.input_ids[i],
                labels=self.labels[i],
                attention_mask=self.attention_mask[i],
                audios=audio_feat,
                asr_targets=self.asr_targets[i],
            )

    def __getitem__(self, index):
        pos_sample = self._process_item(index)

        # 负样本采样（不包含 index 本身）
        if self.raw_data_caption is not None:
            all_indices = list(range(self.raw_data_caption.num_rows))
        else:
            all_indices = list(range(len(self.raw_data)))
        all_indices.remove(index)
        neg_indices = random.sample(all_indices, k=NEG_CONTRASTIVE_BATCH)
        neg_samples = [self._process_item(i) for i in neg_indices]

        # 分别提取字段
        neg_samples_return = {}
        # neg_samples_return["neg_input_ids"] = [item["input_ids"] for item in neg_samples]
        # neg_samples_return["neg_attention_mask"] = [item["attention_mask"] for item in neg_samples]
        neg_samples_return["neg_audios"] = [item["audios"] for item in neg_samples]
        if self.raw_data_caption is not None:
            neg_samples_return["neg_input_caption_ids"] = [item["input_caption_ids"] for item in neg_samples]
            neg_samples_return["neg_attention_caption_mask"] = [item["attention_caption_mask"] for item in neg_samples]
            neg_samples_return["caption_text_masks"] = [item["caption_text_masks"] for item in neg_samples]
        
        # 其他字段可以按需添加，如 ASR 或 labels
        return pos_sample, neg_samples_return


#######
# class AudioDataCollator:
#     def __init__(self, tokenizer, dataset=None):
#         self.tokenizer = tokenizer
#         self.dataset = dataset  # 传入 Dataset 本体以便采样负样本

#     def __call__(self, batch: Dict[str, Union[List[int], torch.Tensor]]):
        
#         batch_samples, index = zip(*batch)
#         batch_size = len(batch)
#         used_indices = set(index)

#         # 采样负样本索引，确保与当前 batch 不重复
#         dataset_size = len(self.dataset)
        
#         # 剔除已用 index，随机采样负样本 index
#         available_indices = list(set(range(dataset_size)) - used_indices)
#         neg_indices = random.sample(available_indices, k=batch_size * 7)
#         neg_batch = [self.dataset[i][0] for i in neg_indices]

#         def stack_or_list(key):
#             if isinstance(batch_samples[0][key], torch.Tensor):
#                 return torch.stack([item[key] for item in batch_samples])
#             else:
#                 return [item[key] for item in batch_samples]

#         def neg_stack_or_list(key):
#             if isinstance(neg_batch[0][key], torch.Tensor):
#                 return torch.stack([item[key] for item in batch_samples])
#             else:
#                 return [item[key] for item in batch_samples]

#         # return {
#         #     "input_ids": stack_or_list("input_ids"),
#         #     "labels": stack_or_list("labels"),
#         #     "attention_mask": stack_or_list("attention_mask"),
#         #     "audios": stack_or_list("audios"),
#         #     "asr_targets": stack_or_list("asr_targets") if batch_samples[0]["asr_targets"] is not None else None,
#         #     "input_ids_neg": neg_stack_or_list("input_ids"),
#         #     "labels_neg": neg_stack_or_list("labels"),
#         #     "attention_mask_neg": neg_stack_or_list("attention_mask"),
#         #     "audios_neg": neg_stack_or_list("audios"),
#         #     "asr_targets_neg": neg_stack_or_list("asr_targets") if batch_samples[0]["asr_targets"] is not None else None,
#         # }
#         return {
#             "input_ids": stack_or_list("input_ids"),
#             "labels": stack_or_list("labels"),
#             "attention_mask": stack_or_list("attention_mask"),
#             "audios": stack_or_list("audios"),
#             "asr_targets": stack_or_list("asr_targets") if batch_samples[0]["asr_targets"] is not None else None,
#             "input_ids_neg": neg_stack_or_list("input_ids").to(batch_samples[0]["input_ids"].device),
#             "labels_neg": None,
#             "attention_mask_neg": neg_stack_or_list("attention_mask").to(batch_samples[0]["attention_mask"].device),
#             "audios_neg": None,
#             "asr_targets_neg": None,
#         }


class AudioDataCollator:
    def __init__(self, tokenizer, dataset=None):
        self.tokenizer = tokenizer
        self.dataset = dataset  # 传入 Dataset 本体以便采样负样本

    def __call__(self, batch):
        
        batch, neg_batch = zip(*batch)

        def stack_list_or_tensor(key):
            first = batch[0][key]
            if isinstance(first, torch.Tensor):
                ########
                if key == "asr_targets":
                    return torch.stack([item[key][:MAX_ASR_LENGTH] for item in batch])
                else:
                    return torch.stack([item[key] for item in batch])
                ########
            else:
                return [item[key] for item in batch]

        def stack_neg_list_or_tensor(key):
            first = neg_batch[0][key]
            if isinstance(first, torch.Tensor):
                return torch.cat([item[key] for item in neg_batch], dim=0)
            else:
                return [item[key] for item in neg_batch]

        return {
            "input_ids": stack_list_or_tensor("input_ids"),
            "labels": stack_list_or_tensor("labels"),
            "attention_mask": stack_list_or_tensor("attention_mask"),
            "audios": stack_list_or_tensor("audios"),
            "asr_targets": stack_list_or_tensor("asr_targets") if batch[0]["asr_targets"] is not None else None,
            "input_caption_ids": stack_list_or_tensor("input_caption_ids") if batch[0]["input_caption_ids"] is not None else None,
            "caption_labels": stack_list_or_tensor("caption_labels") if batch[0]["caption_labels"] is not None else None,
            "asr_caption_targets": stack_list_or_tensor("asr_caption_targets") if batch[0]["asr_caption_targets"] is not None else None,
            "attention_caption_mask": stack_list_or_tensor("attention_caption_mask") if batch[0]["attention_caption_mask"] is not None else None,
            "caption_text_masks": stack_list_or_tensor("caption_text_masks"),
            # "input_ids_neg": stack_neg_list_or_tensor("neg_input_ids"),
            # "attention_mask_neg": stack_neg_list_or_tensor("neg_attention_mask"),
            "audios_neg": stack_neg_list_or_tensor("neg_audios"),
            "neg_input_caption_ids": stack_neg_list_or_tensor("neg_input_caption_ids") if neg_batch[0]["neg_input_caption_ids"] is not None else None,
            "neg_attention_caption_mask": stack_neg_list_or_tensor("neg_attention_caption_mask") if neg_batch[0]["neg_attention_caption_mask"] is not None else None,
            "neg_caption_text_masks": stack_neg_list_or_tensor("caption_text_masks") if neg_batch[0]["caption_text_masks"] is not None else None,
        }
#######

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, training_args=None, audio_processor_path=None, caption_data=None):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        #####
        if caption_data is not None:
            self.raw_data_caption = caption_data
            with open("/data/s50042884/my_code/data/libri_train_qa_prompt.json", "r", encoding="utf-8") as asr_p_f:
                self.asr_prompt = json.load(asr_p_f)
            with open("/data/s50042884/my_code/data/audio_caption_prompt_dynamic_replies.json", "r", encoding="utf-8") as caption_p_f:
                self.caption_prompt = json.load(caption_p_f)
        else:
            self.raw_data_caption = None
            self.asr_prompt = None
            self.caption_prompt = None            
        #####
        
        self.cached_data_dict = {}
        self.audio_processor = WhisperProcessor.from_pretrained(audio_processor_path)
        
        self.training_args = training_args
      
    def __len__(self):
        # return len(self.raw_data)
        #####
        if self.raw_data_caption:
            return self.raw_data_caption.num_rows
        else:
            return len(self.raw_data)
        #####

    def _process_item(self, index):

        if self.raw_data_caption is not None:
            ret = preprocess_caption([self.raw_data_caption[index]], self.tokenizer, self.max_len, asr_prompt=self.asr_prompt, caption_prompt=self.caption_prompt)
            for audio_sample_id in range(len(ret["audio_arrays"])):
                audio = librosa.resample(ret["audio_arrays"][audio_sample_id], orig_sr=ret["caption_sampling_rates"][audio_sample_id], target_sr=CONFIG.sampling_rate)
        else:
            example = self.raw_data[index]
            ret = preprocess([example["conversations"]], self.tokenizer, self.max_len)
            audio_path = ret["audio_paths"][0]
            audio, _ = librosa.load(audio_path, sr=CONFIG.sampling_rate)

        audio_feat = self.audio_processor(audio, sampling_rate=CONFIG.sampling_rate, return_tensors="pt").input_features
        audio_feat = audio_feat.squeeze(0).to(torch.bfloat16)

        if self.raw_data_caption is not None:
                ret = dict(
                    input_ids=ret["input_ids"][0],
                    labels=ret["labels"][0],
                    attention_mask=ret["attention_mask"][0],
                    audios=audio_feat,
                    asr_targets=ret["asr_targets"][0],
                    input_caption_ids=ret["input_caption_ids"][0],
                    caption_labels=ret["caption_labels"][0],
                    asr_caption_targets=ret["asr_caption_targets"][0],
                    attention_caption_mask=ret["attention_caption_mask"][0],
                    caption_text_masks=ret["caption_text_masks"][0],
                )
        else:
            if len(ret["asr_targets"])>0:
                ret = dict(
                    input_ids=ret["input_ids"][0],
                    labels=ret["labels"][0],
                    attention_mask=ret["attention_mask"][0],
                    audios=audio_feat,
                    asr_targets=ret["asr_targets"][0],
                )
            else:
                ret = dict(
                    input_ids=ret["input_ids"][0],
                    labels=ret["labels"][0],
                    attention_mask=ret["attention_mask"][0],
                    audios=audio_feat,
                )
            
        return ret

    def __getitem__(self, index):
        pos_sample = self._process_item(index)

        # 负样本采样（不包含 index 本身）
        if self.raw_data_caption is not None:
            all_indices = list(range(self.raw_data_caption.num_rows))
        else:
            all_indices = list(range(len(self.raw_data)))
        all_indices.remove(index)
        neg_indices = random.sample(all_indices, k=NEG_CONTRASTIVE_BATCH)
        neg_samples = [self._process_item(i) for i in neg_indices]

        # 分别提取字段
        neg_samples_return = {}
        # neg_samples_return["neg_input_ids"] = torch.stack([item["input_ids"] for item in neg_samples])
        # neg_samples_return["neg_attention_mask"] = torch.stack([item["attention_mask"] for item in neg_samples])
        neg_samples_return["neg_audios"] = torch.stack([item["audios"] for item in neg_samples])
        if self.raw_data_caption is not None:
            neg_samples_return["neg_input_caption_ids"] = torch.stack([item["input_caption_ids"] for item in neg_samples])
            neg_samples_return["neg_attention_caption_mask"] = torch.stack([item["attention_caption_mask"] for item in neg_samples])
            neg_samples_return["caption_text_masks"] = torch.stack([item["caption_text_masks"] for item in neg_samples])
        
        # 其他字段可以按需添加，如 ASR 或 labels
        return pos_sample, neg_samples_return


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, training_args, max_len, audio_processor_path
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    with open(data_args.data_path,"rb") as f:
        #train_json = json.load(open(data_args.data_path, "r"))
        train_json=orjson.loads(f.read())

    # train_dataset = dataset_cls(train_json, training_args=training_args, tokenizer=tokenizer, max_len=max_len, audio_processor_path=audio_processor_path)
    
    ######## load caption data
    data = load_from_disk("/data/s50042884/huggingface_model/FusionAudio-1-9-filter-en")
    train_dataset = dataset_cls(train_json, training_args=training_args, tokenizer=tokenizer, max_len=max_len, audio_processor_path=audio_processor_path, caption_data=data)
    ########

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, training_args=training_args, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def get_quantization_config(model_args):
    if model_args.load_in_4bit:
        compute_dtype = torch.bfloat16
        # if model_args.torch_dtype not in {"auto", None}:
        #     compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config

class BasicSetting:
    def __init__(self):
        self.device = "cuda"
        self.sampling_rate = 16000
        self.audio_token_len = 1
        self.stop = "</s>"

CONFIG = BasicSetting()


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # parameter.requires_grad=False
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train():
    global local_rank

    # import glob
    # from safetensors.torch import load_file
    # combined_weights = torch.load("/data/s50042884/my_code/ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss_audio_caption_300epoch_large_batch_audio_encoder_text_proj/checkpoint-2/base_model.bin")
    # print(f"combined_weights is : {combined_weights.keys()}")
    
    # shard_files = sorted(glob.glob(os.path.join("/data/s50042884/my_code/ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss_audio_caption_300epoch_large_batch_audio_encoder_text_proj/checkpoint-2", "model-*.safetensors")))
    # need_combined_weights = {}
    # for shard in shard_files:
    #     shard_state = load_file(shard)
    #     need_combined_weights.update(shard_state)
    # print(f"need_combined_weights is : {need_combined_weights.keys()}")
    # exit(0)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.text_model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    
    # Load model and tokenizer
    quantization_config = get_quantization_config(lora_args)

    print("quantization_config：", quantization_config)

    model = ACLlamaForCausalLM.from_pretrained(
        model_args.text_model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=CONFIG.device,
        quantization_config=quantization_config if lora_args.q_lora else None,
        low_cpu_mem_usage=True
    )


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.text_model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args, max_len=training_args.model_max_length, audio_processor_path=model_args.audio_model_name_or_path
    )
    # audio_config = model.get_model().audio_tower[0].config
    audio_config = model.get_model().audio_tower.config
    audio_config.audio_patch_token = tokenizer.get_vocab()["<audio_patch>"]
    audio_config.llm_pad_token_id = tokenizer.pad_token_id
    audio_config.audio_patch_size = CONFIG.audio_token_len

    if training_args.use_lora:
        #modules_to_save = None #["embed_tokens", "lm_head"]
        #modules_to_save = ["mm_projector1","mm_projector2","asr_encoder_layer"]
        modules_to_save = ["mm_projector1","asr_transformer_encoder","out_norm","lbm", "temperature", "text_projector"]
        # modules_to_save = ["audio_tower","mm_projector1","asr_transformer_encoder","out_norm","lbm"]
        # modules_to_save = ["mm_projector1","out_norm","lbm"]

        def find_all_linear_names(args, model):
            import bitsandbytes as bnb
            cls = bnb.nn.Linear4bit if args.load_in_4bit == 4 else (
                bnb.nn.Linear8bitLt if args.load_in_8bit == 8 else torch.nn.Linear)
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if 'lm_head' in lora_module_names:  # needed for 16-bit
                lora_module_names.remove('lm_head')
            return list(lora_module_names)

        if lora_args.lora_target_modules is None:
            lora_args.lora_target_modules = find_all_linear_names(lora_args, model)
            
        print(f"model is :{model}")
            
        my_target_modules = []
        for id, (name, param) in enumerate(model.named_modules()):
            # "q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"
            lora_list = lora_args.lora_target_modules
            language_model_lora_tag = any(item in name for item in lora_list)
            if language_model_lora_tag:
                if 'audio_tower' not in name:
                    my_target_modules.append(name)
            
        print(f"my_target_modules is : {my_target_modules}")
        # exit(0)
            
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=my_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        #model.get_model().get_input_embeddings().weight[128256].requires_grad = True
        # update adpater and mebed
        #for name, parameter in model.named_parameters():
            #if "mm_projector" in name: #or "audio_feature_head" in name: #or "embed_tokens" in name or "lm_head" in name:
            #    parameter.requires_grad=True
       
        for name, parameter in model.named_parameters():
            # if "audio_tower.decoder" in name:
            #    parameter.requires_grad = False
            # if "audio_tower.encoder" in name:
            #     if "conv1" not in name and "embed_positions" not in name:
            #         parameter.requires_grad = True
            #     else:
            #         parameter.requires_grad = False
            if "audio_tower" in name:
                if "conv1" not in name and "embed_positions" not in name:
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False
        # Print peft trainable params
        model.print_trainable_parameters()

    #######-load stage2 model
    pretrained_model_path = "/data/s50042884/my_code/ACLlama_zhang/ACLlama_output/ACLlama_encoder_stage2_base_chatllm_stage1/checkpoint-6000"
    need_combined_weights = torch.load(pretrained_model_path + "/base_model.bin", map_location=f"cuda")
    model.load_state_dict(need_combined_weights, strict=False)
    #######

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Save the model config
    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

            base_model_path = os.path.join(checkpoint_folder, "base_model.bin")
            torch.save(model.state_dict(), base_model_path)

            #files = os.listdir(checkpoint_folder)
            #for file_name in files:
            #    if ".safetensors" in file_name:
            #        file_path = os.path.join(checkpoint_folder, file_name)
            #        os.remove(file_path)

            return control

    config.save_pretrained(training_args.output_dir)
    print(model) 
    call_back_list = [SavePeftModelCallback]
    #training_args.restore_callback_states_from_checkpoint=True
    # show updated parameters
    print(count_parameters(model))
    model = model.to(torch.bfloat16)
    
    #######
    audio_data_collator = AudioDataCollator(tokenizer, dataset=data_module["train_dataset"])
    
    class ControlledLossLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for key in ["lora_loss", "loss_asr", "contrastive_loss", "contrastive_temperature"]:
                    if key in logs:
                        print(f"[LOG] Step {state.global_step} | {key}: {logs[key]:.4f}")

    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            loss = outputs["loss"]

            extra_logs = {}
            for key in ["lora_loss", "loss_asr", "contrastive_loss", "contrastive_temperature"]:
                val = outputs.get(key)
                if val is not None:
                    extra_logs[key] = val.detach().cpu().float().item()

            # ✅ 只有在当前 step 是 logging_steps 的倍数时才 log
            if self.state.global_step % 25 == 0:
                self.log(extra_logs)
            
            return (loss, outputs) if return_outputs else loss


    # call_back_list.append(ControlledLossLogger())  # ✅ 加入实例

    trainer = CustomTrainer(
        #model=model, tokenizer=tokenizer, args=training_args, callbacks=call_back_list, **data_module
        # model=model, tokenizer=tokenizer, args=training_args, **data_module
        #####
        model=model, tokenizer=tokenizer, args=training_args, callbacks=call_back_list, data_collator=audio_data_collator, **data_module
        #####
    )
    #######
    
    # # Start trainner
    # trainer = Trainer(
    #     #model=model, tokenizer=tokenizer, args=training_args, callbacks=call_back_list, **data_module
    #     # model=model, tokenizer=tokenizer, args=training_args, **data_module
    #     #####
    #     model=model, tokenizer=tokenizer, args=training_args, callbacks=call_back_list, data_collator=audio_data_collator, **data_module
    #     #####
    # )

    with torch.autocast("cuda"):
        # trainer.train(resume_from_checkpoint="/data/s50042884/my_code/ACLlama_output/temp_file/checkpoint-300/")
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    train()
