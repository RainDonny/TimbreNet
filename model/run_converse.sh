# ===================== Conversion for M4Singer =====================
# Note:
# --model_file:
#   <root_path>/model/ckpts/<target_dataset (training dataset)>/<expriement_name>/<epoch>.pt
# --target_singer_f0_file: 
#   <root_path>/preprocess/<source_dataset>/F0/test_f0.pkl
# --inference_dir:
#   <root_path>/model/ckpts/<source_dataset>/Transformer_eval_conversion

python -u converse.py \
--source_dataset 'M4Singer' --dataset_type 'test' \
--model_file '/content/Singing-Voice-Conversion/model/ckpts/Opencpop/Transformer_lr_0.0001/53.pt' \
--target_singer_f0_file '/content/Singing-Voice-Conversion/preprocess/M4Singer/F0/test_f0.pkl' \
--inference_dir '/content/Singing-Voice-Conversion/model/ckpts/M4Singer/Transformer_eval_conversion'
