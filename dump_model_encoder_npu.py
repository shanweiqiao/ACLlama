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
import librosa
from transformers import Trainer, GPTQConfig, deepspeed
from transformers import WhisperProcessor
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prettytable import PrettyTable
from accelerate.utils import DistributedType
from transformers import BitsAndBytesConfig
# from ACLlama import ACLlamaForCausalLM
# from ACLlama_el import ACLlamaForCausalLM
from ACLlama_el_encoder_npu import ACLlamaForCausalLM

########
import torch.nn as nn
import numpy as np
########

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


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
    lazy_preprocess: bool = False


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


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # ['gate_proj', 'o_proj', 'k_proj', 'q_proj', 'up_proj', 'down_proj', 'v_proj']
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['o_proj', 'k_proj', 'q_proj', 'v_proj']
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
        system_message: str = "You are a pirate chatbot who always responds in pirate speak!"
) -> Dict:

    # im_start = tokenizer.im_start_id
    # im_end = tokenizer.im_end_id

    DEFAULT_AUDIO_PATCH_TOKEN = "<audio_patch>"
    tokenizer.add_tokens(DEFAULT_AUDIO_PATCH_TOKEN, special_tokens=True)
    audio_placeholder = DEFAULT_AUDIO_PATCH_TOKEN * CONFIG.audio_token_len
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
    input_ids, audio_paths, targets = [], [], []
    #for i, source in enumerate(sources):
    source = sources[0]

    input_id,target = [], []
    system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system_message).input_ids + [eot_id]
    input_id += system
    input_id += audio_placeholder_ids
    target += [IGNORE_TOKEN_ID] * len(input_id)
    assert len(input_id) == len(target)
    for j, item in enumerate(source):
        role = item["from"]
        value = item["value"]
        if role == 'user':
            _input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [eot_id]
            _target = [IGNORE_TOKEN_ID] * len(_input_id)
            audio_path = item["audio"] if "audio" in item.keys() else None
        elif role == 'assistant':
            _input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [eot_id]
            _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(_assistant) + \
                      [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(nl_tokens) + tokenizer(value).input_ids + [eot_id]
        else:
            raise NotImplementedError
        input_id += _input_id
        target += _target
    #print(input_id)
    #print(target)
    #print(tokenizer.decode(input_id))
    #print(len(input_id), len(target))
    #print(audio_path)
    assert len(input_id) == len(target)
    input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
    target += [IGNORE_TOKEN_ID] * (max_len - len(target))
    input_ids.append(input_id[:max_len])
    targets.append(target[:max_len])
    audio_paths.append(audio_path)
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        audio_paths=audio_paths,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, system_message=None, audio_processor_path=None):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)
        self.audio_processor = WhisperProcessor.from_pretrained(audio_processor_path, torch_dtype=torch.bfloat16)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.audio_paths = data_dict["audio_paths"]
        self.attention_mask = data_dict["attention_mask"]


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        audio, _ = librosa.load(self.audio_paths[i], sr=CONFIG.sampling_rate)
        audio_feat = self.audio_processor(audio, sampling_rate=CONFIG.sampling_rate, return_tensors="pt").input_features
        #audio_feat = audio_feat.unsqueeze(0).unsqueeze(0).to(CONFIG.device, dtype=torch.bfloat16)
        audio_feat = audio_feat.unsqueeze(0).to(CONFIG.device, dtype=torch.bfloat16)

        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            audios=audio_feat
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_len, audio_processor_path
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len, audio_processor_path=audio_processor_path)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
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
        self.llasm_context_len = 2048
        self.sampling_rate = 16000
        self.audio_token_len = 375
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

    is_chat_model = 'instruct' in model_args.text_model_name_or_path.lower()
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
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
    config.use_cache = False
    
    # Addtional settings
    config.audio_tower = model_args.audio_model_name_or_path
    config.adapter_size = model_args.adapter_size
    config.audio_token_len=CONFIG.audio_token_len

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

    #torch.nn.init.xavier_uniform_(model.get_model().conv1.weight)
    #torch.nn.init.xavier_uniform_(model.get_model().conv2.weight)
    torch.nn.init.xavier_uniform_(model.get_model().mm_projector1.weight)
    # torch.nn.init.xavier_uniform_(model.get_model().mm_projector2.weight)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    def text_init_weights(module, std=0.02):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            
    model.get_model().asr_transformer_encoder.apply(init_weights)
    model.get_model().text_projector.apply(init_weights)    
    model.get_model().text_projector_norm.apply(init_weights)    
    model.audio_layer_norm.apply(init_weights)
    model.text_layer_norm.apply(init_weights)
    model.temperature.scalar = torch.nn.init.constant_(model.temperature.scalar, np.log(1 / 1e-1))
    
    
    
    #update embeddings
    embeddings=model.get_input_embeddings()
    new_embeddings = torch.nn.Embedding(embeddings.num_embeddings+1, embeddings.embedding_dim, embeddings.padding_idx).to(CONFIG.device,dtype=torch.bfloat16)
    new_embeddings.weight.data[:embeddings.num_embeddings] = embeddings.weight.data
    model.set_input_embeddings(new_embeddings)

    import copy
    new_head = torch.nn.Linear(config.hidden_size, config.vocab_size+1, bias=False)
    new_head.weight.data[ : config.vocab_size] = model.lm_head.weight.data
    model.lm_head = new_head
    model.audio_feature_head = copy.deepcopy(new_head)
    model.model.audio_feature_head = copy.deepcopy(new_head)

    config.vocab_size+=1
    model.config.vocab_size+=1

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
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length, audio_processor_path=model_args.audio_model_name_or_path
    )
    audio_config = model.get_model().audio_tower[0].config
    #audio_config.audio_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_PATCH_TOKEN])[0]
    #audio_config.audio_start_token, audio_config.audio_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN])
    audio_config.audio_patch_token = tokenizer.get_vocab()["<audio_patch>"]
    
    model = model.to(torch.bfloat16)
    print(f"model is : {model}")

    model.save_pretrained(training_args.output_dir)
    model.config.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    #if training_args.use_lora:
    #    if is_chat_model:
    #        modules_to_save = None
    #    else:
    #        modules_to_save = ["embed_tokens", "lm_head"]
    #    modules_to_save=["embed_tokens", "lm_head"]

    #    def find_all_linear_names(args, model):
    #        import bitsandbytes as bnb
    #        cls = bnb.nn.Linear4bit if args.load_in_4bit == 4 else (
    #            bnb.nn.Linear8bitLt if args.load_in_8bit == 8 else torch.nn.Linear)
    #        lora_module_names = set()
    #        for name, module in model.named_modules():
    #            if isinstance(module, cls):
    #                names = name.split('.')
    #                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    #        if 'lm_head' in lora_module_names:  # needed for 16-bit
    #            lora_module_names.remove('lm_head')
    #        return list(lora_module_names)

    #    if lora_args.lora_target_modules is None:
    #        lora_args.lora_target_modules = find_all_linear_names(lora_args, model)

    #    print(lora_args.lora_target_modules)
    #    print(modules_to_save)

    #    lora_config = LoraConfig(
    #        r=lora_args.lora_r,
    #        lora_alpha=lora_args.lora_alpha,
    #        target_modules=lora_args.lora_target_modules,
    #        lora_dropout=lora_args.lora_dropout,
    #        bias=lora_args.lora_bias,
    #        task_type="CAUSAL_LM",
    #        modules_to_save=modules_to_save  # This argument serves for adding new tokens.
    #    )
    #    if lora_args.q_lora:
    #        model = prepare_model_for_kbit_training(
    #            model, use_gradient_checkpointing=training_args.gradient_checkpointing
    #        )

    #    model = get_peft_model(model, lora_config)

    #    for name, parameter in model.named_parameters():
    #        if "mm_projector" in name or "embed_tokens" in name:
    #            parameter.requires_grad=True
    #   
    #    # Print peft trainable params
    #    model.print_trainable_parameters()


    #if training_args.gradient_checkpointing:
    #    model.enable_input_require_grads()

    ## Save the model config
    #config.save_pretrained(training_args.output_dir)
    #print(model) 
    ## show updated parameters
    ##print(count_parameters(model))
    ## Start trainner
    #trainer = Trainer(
    #    model=model, tokenizer=tokenizer, args=training_args, **data_module
    #)

    #with torch.autocast("cuda"):
    #    trainer.train()
    #trainer.save_state()

    #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
