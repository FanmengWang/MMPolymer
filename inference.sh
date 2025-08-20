data_list=("Egc" "Egb" "Eea" "Ei" "Xc" "EPS" "Nc" "Eat")
for data in "${data_list[@]}"
    do
        data_path="./dataset/finetune_data"  # replace to your data path
        results_path="./infer_polymer_results/infer_data_${data}_results"  # replace to your results path
        weight_path="./ckpt/${data}/checkpoint_best.pt"  # replace to your ckpt path
        batch_size=32
        task_name="${data}" # data folder name 
        task_num=1
        dict_name='dict.txt'
        conf_size=11
        only_polar=0

        export CUDA_VISIBLE_DEVICES=0
        python ./MMPolymer/infer.py --user-dir ./MMPolymer $data_path --task-name $task_name --valid-subset test \
            --results-path $results_path \
            --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
            --task MMPolymer_finetune --loss MMPolymer_finetune --arch MMPolymer_base \
            --classification-head-name $task_name --num-classes $task_num \
            --dict-name $dict_name --conf-size $conf_size \
            --only-polar $only_polar  \
            --path $weight_path  \
            --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
            --log-interval 50 --log-format simple 
    done

    
