import argparse
import os
import time

def replace_da_file(file_name, old_path, new_path):
    with open(file_name, 'r+', encoding='utf-8') as file:
        lines = file.readlines()
        file.seek(0)
        for line in lines:
            modified_line = line.replace(old_path, new_path)
            file.write(modified_line)
        file.truncate() 

def env_build(args, my_args):

    
    conda deactivate
    conda create -n audiopre --clone PyTorch-2.1.0
    conda activate audiopre
    cd /home/ma-user/work/algorithm/encllm_guiyang_v2/ACLlama
    pip3 install -r requirements_npu.txt
    pip3 install deepspeed
    
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
    
    print(f"pwd")
    os.system(f"pwd")
    
    print(f"ls -R")
    os.system("ls -R")
    
    print(f"cd {args.transformers_url}  && pip3 install -e ./")
    os.system(f"cd {args.transformers_url}  && pip3 install -e ./")
    
    print(f"cd {args.peft_url}  && pip3 install -e ./")
    os.system(f"cd {args.peft_url}  && pip3 install -e ./")
    
    print(f"cd {args.env_url}  && pip3 install -r requirements.txt")
    os.system(f"cd {args.env_url}  && pip3 install -r requirements.txt")
    
    return


def deal_data_yaml(args, my_args):
    
    TGT_LANG = my_args["TGT_LANG"]
    
    print("now is here!")
    config_share = args.data_url + "st/config_share.yaml"
    print(f"cat {config_share}")
    os.system(f"cat {config_share}")
    
    # config file
    file_name = args.data_url + "st/config_share.yaml"
    old_path = '/home/ma-user/work/s50042884/TAB/data/mustc.en-{}/'.format(TGT_LANG)
    new_path = '{}'.format(args.data_url)
    replace_da_file(file_name, old_path, new_path)
            
    print(f"after cat {config_share}")
    os.system(f"cat {config_share}")
          
    # data file
    file_name = args.data_url + "st/dev.tsv"
    old_path = '/home/ma-user/work/s50042884/TAB/data/mustc.en-{}/'.format(TGT_LANG)
    new_path = '{}'.format(args.data_url)
    replace_da_file(file_name, old_path, new_path)
    
    file_name = args.data_url + "st/tst-COMMON.tsv"
    old_path = '/home/ma-user/work/s50042884/TAB/data/mustc.en-{}/'.format(TGT_LANG)
    new_path = '{}'.format(args.data_url)
    replace_da_file(file_name, old_path, new_path)
    
    file_name = args.data_url + "st/train.tsv"
    old_path = '/home/ma-user/work/s50042884/TAB/data/mustc.en-{}/'.format(TGT_LANG)
    new_path = '{}'.format(args.data_url)
    replace_da_file(file_name, old_path, new_path)
    
    return


def deal_model_yaml(args, my_args):
    
    TGT_LANG = my_args["TGT_LANG"]
        
    # asr ckpt
    file_name = args.fairseq_url + "egs/mustc/st/conf/su2t_base_nottune.yaml"
    old_path = '/data/s50042884/my_code/TAB/checkpoints/mustc_en_${}_asr.pt'.format(TGT_LANG)
    new_path = '{}mustc_en_${}_asr.pt'.format(args.pretrain_model_url, TGT_LANG)
    replace_da_file(file_name, old_path, new_path)
          
    # mt ckpt
    file_name = args.fairseq_url + "egs/mustc/st/conf/su2t_base_nottune.yaml"
    old_path = '/data/s50042884/my_code/TAB/checkpoints/mustc_en_${}_mt.pt'.format(TGT_LANG)
    new_path = '{}mustc_en_${}_mt.pt'.format(args.pretrain_model_url, TGT_LANG)
    replace_da_file(file_name, old_path, new_path)
    
    # # training epoch
    # file_name = args.fairseq_url + "egs/mustc/st/conf/basis.yaml"
    # old_path = '/data/s50042884/my_code/TAB/checkpoints/mustc_en_${}_mt.pt'.format(TGT_LANG)
    # new_path = '{}mustc_en_${}_mt.pt'.format(args.pretrain_model_url, TGT_LANG)
    # replace_da_file(file_name, old_path, new_path)
    
    return


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sh_url",
        default=None,
        type=str,
        help="start bash script",
    )
    parser.add_argument(
        "--data_url",
        default=None,
        type=str,
        help="training data path",
    )
    parser.add_argument(
        "--pretrain_model_url",
        default=None,
        type=str,
        help="pretrain model path",
    )
    parser.add_argument(
        "--output_url",
        default=None,
        type=str,
        help="path for the return file",
    )
    
    
    parser.add_argument(
        "--env_url",
        default=None,
        type=str,
        help="environment package path",
    )

    parser.add_argument(
        "--src_url",
        default=None,
        type=str,
        help="",
    )

    parser.add_argument(
        "--transformers_url",
        default=None,
        type=str,
        help="transformers file path",
    )
    parser.add_argument(
        "--peft_url",
        default=None,
        type=str,
        help="peft file path",
    )


    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_cmdline()

    SH_FILE = args.sh_url
    DATA_PATH = args.data_url
    PT_MODEL_PATH = args.pretrain_model_url
    SAVE_PATH = args.output_url

    TRANSFORMERS_PATH = args.transformers_url
    PEFT_PATH = args.peft_url
    
    print(f"start script args is : {args}")
    
    TGT_LANG = "fr"
    my_args = {}
    my_args["TGT_LANG"] = TGT_LANG
    
    
    
    
    
    
    
    print(f"Make output dir based on SAVE_PATH :{SAVE_PATH}")
    os.makedirs(SAVE_PATH, exist_ok=True)

    print(f"install env like `pip install --editable` when use fairseq toolkit")
    env_build(args, my_args)








    #### my args and path
    deal_data_yaml(args, my_args)
    deal_model_yaml(args, my_args)

    print("env check:")
    print(f"pip list")
    os.system("pip list")
    print(f"conda list")
    os.system("conda list")
    print(f"nvidia-smi")
    os.system(f"nvidia-smi")
    print(f"df -h .")
    os.system(f"df -h .")

    print("start infer...") # -1是为了去掉"/"
    ENV_PATH = args.env_url[:-1] if args.env_url[-1]=="/" else args.env_url
    print(SH_FILE, DATA_PATH[:-1], TRANSFORMERS_PATH[:-1], PEFT_PATH[:-1], SAVE_PATH[:-1], ENV_PATH)
    
    print(f"SH_FILE is : {SH_FILE}")
    print(f"DATA_PATH is : {DATA_PATH}")
    print(f"SRC_PATH is : {SRC_PATH}")
    print(f"SAVE_PATH is : {SAVE_PATH}")
    print(f"ENV_PATH is : {ENV_PATH}")
    print(f"PT_MODEL_PATH is : {PT_MODEL_PATH}")
    print(f"TGT_LANG is : {TGT_LANG}")

    start_time = time.time()
    
    print(f"/bin/bash {SH_FILE} {DATA_PATH} {PT_MODEL_PATH}")
    os.system(f"/bin/bash {SH_FILE}")
    
    end_time = time.time()

    print("\n\n\nstop training...")
    print(f"All training time is : {end_time - start_time}\n\n\n")


if __name__ == '__main__':
    main()
