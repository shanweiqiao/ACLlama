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

model_path="/data/s50042884/my_code/ACLlama_A100_right/ACLlama_output/ACLlama_encoder_contrastive_base_stage1_chatllm_npu/"
export CUDA_VISIBLE_DEVICES=${device[@]}
python llama3.1_peft_lora_predict_npu.py \
    --eval_data "/data/s50042884/huggingface_model/libri_test.json" \
    --audio_tower "/data/s50042884/huggingface_model/whisper-large-v3" \
    --base_model_path "/data/s50042884/my_code/ACLlama_A100_right/ACLlama_output/ACLlama_encoder_stage2_base_chatllm_stage1/checkpoint-6000" \
    --peft_model_id "/data/s50042884/my_code/ACLlama_A100_right/ACLlama_output/ACLlama_encoder_stage2_base_chatllm_stage1/checkpoint-6000" \
    --clean_out_path ${model_path}"test_clean.txt" \
    --other_out_path ${model_path}"test_other.txt" \
    --num_threads ${gpu_num}


    # --eval_data "/data/s50042884/huggingface_model/libri_test_clean.json" \
    # --eval_data "/data/s50042884/huggingface_model/libri_test_other.json" \
