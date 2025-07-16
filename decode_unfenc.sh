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

# model_path="/data/s50042884/my_code/audio_pretrain/ACLlama_zhang_73/ACLlama_output/ACLlama_encoder_stage2_base_chatllm_stage1/"
# model_path="/data/s50042884/my_code/audio_pretrain/ACLlama_zhang_73/ACLlama_output/ACLlama_encoder_stage2_base_contrastive_asr_loss_from_stage1_chatllm/"
model_path="/data/s50042884/my_code/ACLlama_output/contrastive_captiondata_all_smalllr_after_stage2_allloss_lbmproj_textrealmask_unfreezeenc/"

export CUDA_VISIBLE_DEVICES=${device[@]}
python llama3.1_peft_lora_predict_unfenc.py \
    --eval_data "/data/s50042884/huggingface_model/libri_test.json" \
    --audio_tower "/data/s50042884/huggingface_model/whisper-large-v3" \
    --base_model_path "/data/s50042884/my_code/ACLlama_output/ACLlama_encoder_chatllm_contrastive_lbmproj_testproj_unfenc" \
    --peft_model_id ${model_path}"checkpoint-1200" \
    --clean_out_path ${model_path}"test_clean.txt" \
    --other_out_path ${model_path}"test_other.txt" \
    --num_threads ${gpu_num}

    # --clean_out_path "/data/s50042884/my_code/audio_pretrain/ACLlama_zhang/ACLlama_output/ACLlama_encoder_stage2_from_contrastive_asr_loss_base_stage1/test_clean.txt" \
    # --other_out_path "/data/s50042884/my_code/ACLlama_zhang/ACLlama_output/ACLlama_encoder_stage2_from_contrastive_asr_loss_base_stage2/test_other.txt" \
    # --eval_data "/data/s50042884/huggingface_model/libri_test_clean.json" \
    # --eval_data "/data/s50042884/huggingface_model/libri_test_other.json" \
    # --clean_out_path "/data/s50042884/my_code/ACLlama_zhang/ACLlama_output/ACLlama_encoder_stage2_from_contrastive_asr_loss_base_stage2/test_Spoken_Language_Identification.txt" \
    # --other_out_path "/data/s50042884/my_code/ACLlama_zhang/ACLlama_output/ACLlama_encoder_stage2_from_contrastive_asr_loss_base_stage2/test_temp.txt" \

