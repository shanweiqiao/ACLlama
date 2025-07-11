#! /bin/bash

device=(0,1,2,3,4,5,6,7)
gpu_num=8
# device=(0)
# gpu_num=1


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_COMPILE=0
export DISABLE_TORCH_COMPILE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# output_tag="../ACLlama_output/ACLlama_encoder_contrastive_base_stage1_chatllm_npu"
# output_tag="../ACLlama_output/test_analize"
output_tag="../ACLlama_output/test"

if [[ ! -e ${output_tag} ]]; then
    mkdir -p ${output_tag}
fi
code_save_path=$output_tag"/code_save/"
if [[ ! -e ${code_save_path} ]]; then
    mkdir -p ${save_dir}
fi

export ASCEND_VISIBLE_DEVICES=${device}
export ASCEND_DEVICE_ID=${device}
cmd="torchrun
    --nproc_per_node 8
    --nnodes 1
    --node_rank 0
    --master_addr localhost
    --master_port 6601
    finetune_acllama_encoder_npu.py
    --audio_model_name_or_path "/data/s50042884/huggingface_model/whisper-large-v3"
    --text_model_name_or_path "/data/s50042884/my_code/ACLlama_A100_right/ACLlama_output/ACLlama_encoder_chatllm_contrastive_npu_stag1"
    --data_path "/data/s50042884/my_code/data/audio_caps_formatted_clean.json"
    --output_dir ${output_tag}
    --num_train_epochs 40
    --fp16 True
    --per_device_train_batch_size 8
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 32
    --evaluation_strategy "no"
    --save_strategy "steps"
    --save_steps 100
    --save_total_limit 100
    --learning_rate 1e-4
    --weight_decay 0.1
    --adam_beta2 0.95
    --warmup_ratio 0.01
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --report_to "none"
    --model_max_length 512
    --gradient_checkpointing True
    --deepspeed "./config/ds_config_zero2.json""

save_cmd="${output_tag}/train.log"
echo $cmd
eval $cmd 2>&1 | tee $save_cmd


    # --per_device_train_batch_size 32 \
    # --per_device_eval_batch_size 1 \
    # --gradient_accumulation_steps 8 \
    # --deepspeed "./config/ds_config_zero2.json" \
    # --fp16 True \
    # --bf16 True \
    # --use_lora

# python3 finetune_acllama.py \
#     --audio_model_name_or_path "/data/s50042884/huggingface_model/whisper-large-v3" \
#     --text_model_name_or_path "/data/s50042884/my_code/ACLlama_output/ACLlama_lora" \
#     --data_path "/data/s50042884/huggingface_model/libri_train_update.json" \
#     --fp16 True \
#     --output_dir "../ACLlama_output/ACLlama_lora" \
#     --num_train_epochs 40 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 64 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 1 \
#     --learning_rate 1e-5 \
#     --weight_decay 0.1 \
#     --adam_beta2 0.95 \
#     --warmup_ratio 0.01 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --report_to "none" \
#     --model_max_length 512 \
#     --gradient_checkpointing True \
#     --deepspeed "./config/ds_config_zero2.json" \
#     --use_lora