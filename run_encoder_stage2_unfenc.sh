#! /bin/bash

# sleep 4h

device=(0,1,2,3,4,5,6,7)
gpu_num=8
# device=(0)
# gpu_num=1


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export TORCH_COMPILE=0
# export DISABLE_TORCH_COMPILE=1
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# output_tag="../ACLlama_output/ACLlama_lora_finetune"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_contrastive_loss"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_contrastive_loss_v1"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss_audio_caption_300epoch"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss_audio_caption_300epoch_large_batch_audio_encoder"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss_audio_caption_300epoch_large_batch_audio_encoder_text_proj"
# output_tag="../ACLlama_output/ACLlama_encoder_finetune_contrastive_loss_audio_caption_large_batch_after_stage2_fix_text_emb"
# output_tag="../ACLlama_output/ACLlama_encoder_finetune_contrastive_loss_audio_caption_large_batch_after_stage2"
# output_tag="../ACLlama_output/ACLlama_encoder_finetune_contrastive_captiondata_large_batch_after_stage1"
# output_tag="../ACLlama_output/ACLlama_encoder_finetune_contrastive_captiondata_all_newcontrastiveloss_smalllr_after_stage1"
# output_tag="../ACLlama_output/test"
# output_tag="../ACLlama_output/ACLlama_encoder_finetune_contrastive_captiondata_all_newcontrastiveloss_smalllr_after_stage1_lbmproj_textrealmask"
# output_tag="../ACLlama_output/temp_file"
output_tag="../ACLlama_output/contrastive_captiondata_all_smalllr_after_stage2_allloss_lbmproj_textrealmask_unfreezeenc"

if [[ ! -e ${output_tag} ]]; then
    mkdir -p ${output_tag}
fi
code_save_path=$output_tag"/code_save/"
if [[ ! -e ${code_save_path} ]]; then
    mkdir -p ${save_dir}
fi

export CUDA_VISIBLE_DEVICES=${device[@]}
cmd="torchrun
    --nproc_per_node 8
    --nnodes 1
    --node_rank 0
    --master_addr localhost
    --master_port 6601
    finetune_acllama_encoder_stage2_unfenc.py
    --audio_model_name_or_path "/data/s50042884/huggingface_model/whisper-large-v3"
    --text_model_name_or_path "/data/s50042884/my_code/ACLlama_output/ACLlama_encoder_chatllm_contrastive_lbmproj_testproj_unfenc"
    --data_path "/data/s50042884/my_code/data/audio_caps_formatted_clean.json"
    --output_dir ${output_tag}
    --num_train_epochs 8
    --per_device_train_batch_size 16
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 16
    --evaluation_strategy "no"
    --save_strategy "steps"
    --save_steps 50
    --save_total_limit 100
    --learning_rate 5e-5
    --weight_decay 0.1
    --adam_beta2 0.95
    --warmup_ratio 0.01
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --report_to "none"
    --model_max_length 512
    --gradient_checkpointing True
    --deepspeed "./config/ds_config_zero2.json"
    --use_lora"

    # --per_device_train_batch_size 32 \
    # --per_device_eval_batch_size 1 \
    # --gradient_accumulation_steps 8 \
    # --deepspeed "./config/ds_config_zero2.json" \
    # --fp16 True \
    # --bf16 True \
    # --num_train_epochs 40 \
    # --use_lora
    # --text_model_name_or_path "/data/s50042884/my_code/ACLlama_zhang/ACLlama_output/ACLlama_model_ori_zhang_chatllm"
    # --text_model_name_or_path "/data/s50042884/my_code/ACLlama_output/ACLlama_encoder_chatllm_contrastive"


script_path=$(realpath "$0")
script_dir=$(dirname "$(realpath "$0")")
cp ${script_path} ${save_dir}/
cp ./finetune_acllama.py ${code_save_path}
cp ./ACLlama_el.py ${code_save_path}
cp ./dump_model.py ${code_save_path}
cp ./my_dump_model.sh ${code_save_path}

timestamp=$(date +"%Y%m%d_%H%M%S")
save_cmd="${output_tag}/train_${timestamp}.log"
echo $cmd
eval $cmd 2>&1 | tee $save_cmd

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