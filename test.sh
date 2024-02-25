SEED=123456
CKPT_PATH=checkpoints/???????

python3 main/main_mamba_chem.py test --seed_everything ${SEED} \
                    --ckpt_path ${CKPT_PATH} \
                    --data.batch_size 320 --data.pretrained_tokenizer_path "chem_tokenizer.json" --model.top_k 10 \
                    --trainer.callbacks+=ModelCheckpoint --trainer.callbacks.dirpath='checkpoints' \
                    --trainer.accelerator gpu --trainer.devices [0]
