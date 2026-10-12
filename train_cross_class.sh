# Set the path to save checkpoints
OUTPUT_DIR='output_dir/vit_base_16_obj_to_texture'
# Mvtec Anomaly Detection dataset
DATA_PATH='/Mvtec-ImageNet/train'
# Download the tokenizer weight from OpenAI's DALL-E
TOKENIZER_PATH='weights/tokenizer'


OMP_NUM_THREADS=1 python train_cross_class.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --mask_ratio 0.3 \
        --model vit_base_patch16_224_8k_vocab --tokenizer_weight_path ${TOKENIZER_PATH} \
        --batch_size 1 --lr 5e-4 --warmup_epochs 10 --epochs 300 \
        --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std \
        --from_obj_to_texture  # cross class from objects to textures, ortherwise from textures to objects
