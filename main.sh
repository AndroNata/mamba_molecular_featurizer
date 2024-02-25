SEED=123456
DATA_DIR=data/benchmark_uspto
MODEL_NAME=chem_symbol_repair

python3 main/main_multilabel_classification.py fit --seed_everything ${SEED} --data.pretrained_tokenizer_path "chem_tokenizer.json" \
                    --trainer.num_sanity_val_steps 2 --trainer.detect_anomaly true --data.shuffle_train true --data.batch_size 80 --model.top_k 10 \
                    --d_model 256 --n_layer 10 --vocab_size 421 \
                    --trainer.callbacks+=ModelCheckpoint --trainer.callbacks.dirpath='checkpoints' --trainer.callbacks.monitor="val/mAP" \
                    --trainer.callbacks.filename=${MODEL_NAME}'-ep={epoch}-mAP={val/mAP:.3f}' --trainer.callbacks.mode='max' --trainer.callbacks.save_last=true \
                    --trainer.callbacks.auto_insert_metric_name=false  --trainer.callbacks.every_n_epochs=1 --trainer.callbacks.save_top_k=2 \
                    --trainer.val_check_interval 0.5 --trainer.log_every_n_steps 1 --trainer.max_epochs 60 \
                    --trainer.accelerator gpu --trainer.devices [0] \
                    --trainer.logger+=WandbLogger --trainer.logger.save_dir . --trainer.logger.project='reagspace'
