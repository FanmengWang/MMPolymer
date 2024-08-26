data_list=("Egc" "Egb" "Eea" "Ei" "Xc" "EPS" "Nc" "Eat")
for data in "${data_list[@]}"
        do
                data_path="./dataset/finetune_data"  # replace to your data path
                save_dir="./ckpt/${data}"  # replace to your save path
                n_gpu=1
                MASTER_PORT=9999
                dict_name='dict.txt'
                weight_path="./ckpt/pretrain.pt"
                task_name="${data}"  # data folder name
                task_num=1
                lr=1e-4
                batch_size=32
                epoch=600
                dropout=0.1
                warmup=0.06
                local_batch_size=32
                only_polar=0
                conf_size=11
                seed=0
                metric="valid_agg_r2"

                export NCCL_ASYNC_ERROR_HANDLING=1
                export OMP_NUM_THREADS=1
                update_freq=`expr $batch_size / $local_batch_size`
                python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path \
                        --task-name $task_name --user-dir ./MMPolymer --train-subset train --valid-subset valid \
                        --conf-size $conf_size \
                        --num-workers 8 --ddp-backend=c10d \
                        --dict-name $dict_name \
                        --task MMPolymer_finetune --loss MMPolymer_finetune --arch MMPolymer_base  \
                        --classification-head-name $task_name --num-classes $task_num \
                        --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
                        --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout\
                        --update-freq $update_freq --seed $seed \
                        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
                        --finetune-from-model $weight_path \
                        --log-interval 100 --log-format simple \
                        --validate-interval 1 --keep-last-epochs 2 \
                        --best-checkpoint-metric $metric --patience 30 \
                        --save-dir $save_dir --only-polar $only_polar --tmp-save-dir $save_dir/tmp \
                        --maximize-best-checkpoint-metric 
        done
