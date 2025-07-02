#! /bin/bash

cd /home/ma-user/work/algorithm/encllm_guiyang_v2/ACLlama

sed 's|/data/s50042884/all_data/audio-enc-moe-data/asr/librispeech_asr/LibriSpeech|/home/ma-user/work/dataset/s2tt_ellm_asr_v1/s50042884/s50042884/asr/librispeech_asr/LibriSpeech|g' /home/ma-user/work/dataset/s2tt_ellm_asr_model_v2/data/libri_train_update.json > /home/ma-user/work/dataset/s2tt_ellm_asr_model_v2/data/libri_train_update_new.json

sed -i "s|/data/s50042884/huggingface_model/whisper-large-v3|/home/ma-user/work/dataset/s2tt_ellm_asr_model_v4/whisper-large-v3|g" my_dump_model.sh
sed -i "s|/data/s50042884/huggingface_model/Llama-3.2-3B|/home/ma-user/work/dataset/s2tt_ellm_asr_model_v3/Llama-3.2-3B|g" my_dump_model.sh
sed -i "s|/data/s50042884/huggingface_model/libri_train_update.json|/home/ma-user/work/dataset/s2tt_ellm_asr_model_v2/data/libri_train_update_new.json|g" my_dump_model.sh
sed -i "s|../ACLlama_output/ACLlama_load_pretrained_encoder|../ACLlama_output/ACLlama_load_pretrained_encoder|g" my_dump_model.sh
sed -i "s|CUDA_VISIBLE_DEVICES=6|ASCEND_VISIBLE_DEVICES=0|g" my_dump_model.sh
sed -i "s|cuda|npu|g" dump_model.py
sed -i "s|from ACLlama_el import ACLlamaForCausalLM|from ACLlama_el_npu import ACLlamaForCausalLM|g" dump_model.py
sed -i "s|cuda|npu|g" ACLlama_el_npu.py
sed -i "s|audio_config = model.get_model().audio_tower.config|audio_config = model.get_model().audio_tower[0].config|g" dump_model.py

sed -i "s|/data/s50042884/huggingface_model/whisper-large-v3|/home/ma-user/work/dataset/s2tt_ellm_asr_model_v4/whisper-large-v3|g" run_npu.sh
sed -i "s|/data/s50042884/my_code/ACLlama_output/ACLlama_load_pretrained_encoder|/home/ma-user/work/algorithm/encllm_guiyang_v2/ACLlama_output/ACLlama_load_pretrained_encoder|g" run_npu.sh
sed -i "s|/data/s50042884/huggingface_model/libri_train_update.json|/home/ma-user/work/dataset/s2tt_ellm_asr_model_v2/data/libri_train_update_new.json|g" run_npu.sh
sed -i "s|--nproc_per_node 8|--nproc_per_node 4|g" run_npu.sh
sed -i "s|from ACLlama_el import ACLlamaForCausalLM|from ACLlama_el_npu import ACLlamaForCausalLM|g" finetune_acllama_npu.py
sed -i "s|per_device_train_batch_size 2|per_device_train_batch_size 16|g" run_npu.sh
sed -i "s|gradient_accumulation_steps 128|gradient_accumulation_steps 16|g" run_npu.sh

sed -i "s|cuda|npu|g" finetune_acllama_npu.py