# ===================== Inference for M4Singer =====================
# Note:
# --resume:
#   <root_path>/model/ckpts/<training_dataset>/<expriement_name>/<epoch>.pt

python -u main.py --debug False --evaluate True \
--dataset 'M4Singer' --converse True --model 'Transformer' \
--resume '/content/Singing-Voice-Conversion/model/ckpts/Opencpop/Transformer_lr_0.0001/53.pt'
