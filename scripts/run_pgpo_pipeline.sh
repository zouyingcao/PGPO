cd ..

model_name=Llama-2-7b-hf
task=alfworld

node_num=8  # number of GPUs for SFT training
worker_num=8 # number of workers for exploration
sample_node_num=8 # number of GPUs for launching models in exploration
sample_num_workers=8 # number of workers for launching models in exploration

model_path=./llm_ckpt/ # path to the original LLM
save_dir=./checkpoints_${task}/    # checkpoint save path
save_path=./experiments/${model_name}-${task}-sft-with-plan-with-paraphrase/  # output save path
logs_path=${save_path}logs

mkdir -p ${save_path}
mkdir -p ${logs_path}/

export WANDB_MODE=offline

# Part 1: SFT stage (if you run run_sft.sh before, you can )
sft_data_path="data/${task}_sft_with_plan.json"
batch_size=48
micro_batch_size=4
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

sft_model_name=${model_name}-${task}-sft-with-plan

python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20002 fastchat/train/train.py \
    --model_name_or_path ${model_path}${model_name} \
    --data_path ${sft_data_path} \
    --bf16 True \
    --output_dir ${save_dir}${sft_model_name} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess False

# if failed, exit
if [ $? -ne 0 ]; then
    echo "SFT training failed"
    exit 1
fi

# Evaluate SFT agent
fs_worker_port=21012
python -u -m fastchat.serve.model_worker --model-path ${save_dir}${sft_model_name} --host 127.0.0.1 --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &

fs_worker_pid=$!
sleep 200

# evaluate on the test set
python -m eval_agent.main --agent_config fastchat --model_name ${sft_model_name} --exp_config ${task} --split test --override

# if failed, exit
if [ $? -ne 0 ]; then
    echo "base agent evaluation failed"
    kill -9 $fs_worker_pid
    exit 1
fi

# kill the model worker
kill -9 $fs_worker_pid

cur_model_name=${sft_model_name}
monte_carlo_explore_model_name=${cur_model_name}-monte-carlo-explore
for i in {1..3}; do
    # Part 2: Base agent explore stage
    # launch the fastchat model worker
    explore_model_name=${cur_model_name}-explore-with-paraphrase

    for ((j=0;j<${sample_num_workers};j=j+1)); do
        if [ -d "${save_dir}${explore_model_name}-${j}" ]; then
            echo "Link to model exists"
        else
            ln -s ${save_dir}${cur_model_name} ${save_dir}${explore_model_name}-${j}
        fi
    done
    if [ -f "${logs_path}/worker_pid.txt" ]; then
        rm ${logs_path}/worker_pid.txt
    fi

    fs_worker_port=21012
    worker_idx=0
    for ((j=0;j<${sample_num_workers};j=j+1)); do
        echo "Launch the model worker on port ${fs_worker_port}"
        CUDA_VISIBLE_DEVICES=$((${worker_idx} % ${sample_node_num})) python -u -m fastchat.serve.model_worker \
            --model-path ${save_dir}${explore_model_name}-${j} \
            --host 127.0.0.1 \
            --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/worker_pid.txt
        fs_worker_port=$(($fs_worker_port+1))
        worker_idx=$(($worker_idx+1))
        sleep 15
    done
    
    sleep 60

    # start explore on the same sft data
    echo "Base agent starts exploring"
    if [ -f "${logs_path}/eval_pid.txt" ]; then
        rm ${logs_path}/eval_pid.txt
    fi

    step_traj_save_path=${save_path}${explore_model_name}
    if [ -d ${step_traj_save_path} ]; then
        rm -r ${step_traj_save_path}
    fi
    mkdir -p ${step_traj_save_path}

    for ((j = 0; j < $worker_num; j++ )); do
        python3 exploration.py --with_plan --exp_config ${task} --model_name ${explore_model_name}-$((j%sample_node_num)) --part_num $worker_num --part_idx ${j} --save_path ${step_traj_save_path}  >> ${logs_path}/gen_response_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/eval_pid.txt
    done
    # wait all workers done for sampling trajs
    wait $(cat ${logs_path}/eval_pid.txt)
    rm ${logs_path}/eval_pid.txt
    echo "Base agent has finished exploring"

    # if failed, exit
    if [ $? -ne 0 ]; then
        echo "base agent exploration failed"
        kill -9 $(cat ${logs_path}/worker_pid.txt)
        rm ${logs_path}/worker_pid.txt
        exit 1
    fi

    # kill the model worker
    echo "Kill the model workers"
    kill -9 $(cat ${logs_path}/worker_pid.txt)
    rm ${logs_path}/worker_pid.txt

    # Part 3: Estimate planning-oriented rewards
    for ((j=0;j<${sample_num_workers};j=j+1)); do
        if [ -d "${save_dir}${monte_carlo_explore_model_name}-${j}" ]; then
            echo "Link to model exists"
        else
            ln -s ${save_dir}${sft_model_name} ${save_dir}${monte_carlo_explore_model_name}-${j}
        fi
    done
    if [ -f "${logs_path}/worker_pid.txt" ]; then
        rm ${logs_path}/worker_pid.txt
    fi

    fs_worker_port=21012
    worker_idx=0
    for ((j=0;j<${sample_num_workers};j=j+1)); do
        echo "Launch the model worker on port ${fs_worker_port}"
        CUDA_VISIBLE_DEVICES=$((${worker_idx} % ${sample_num_workers})) python -u -m fastchat.serve.model_worker \
            --model-path ${save_dir}${monte_carlo_explore_model_name}-${j} \
            --host 127.0.0.1 \
            --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/worker_pid.txt
        fs_worker_port=$(($fs_worker_port+1))
        worker_idx=$(($worker_idx+1))
        sleep 15
    done
    sleep 60

    echo "Base agent starts monte carlo sampling"
    if [ -f "${logs_path}/eval_pid.txt" ]; then
        rm ${logs_path}/eval_pid.txt
    fi

    sample_num=5
    per_iteration_num=5
    sample_workers=16
    sample_iterations=$((sample_num/per_iteration_num))

    for ((j=0;j<${sample_iterations};j=j+1));do
        for ((k=0;k<${per_iteration_num};k=k+1)); do
            # monte carlo sampling trajectories
            monte_carlo_sample_save_path=${save_path}monte_carlo_sample_iteration_${i}/sampled_traj_$((j*per_iteration_num+k))
            for ((l=0;l<$sample_workers; l++)); do
                output_path=${monte_carlo_sample_save_path}/
                if [ -d ${output_path} ]; then
                    rm -r ${output_path}
                fi
                mkdir -p ${output_path}
                python monte_carlo_sample.py --agent_config fastchat_explore --model_name ${monte_carlo_explore_model_name}-$((l%sample_num_workers)) --exp_config ${task} --part_num ${sample_workers} --part_idx ${l} --save_path ${output_path} --data_path ${step_traj_save_path} >> ${logs_path}/gen_response_worker-$((j*per_iteration_num+k))-${l}.log 2>&1 &
                echo $! >> ${logs_path}/eval_pid.txt
            done
            wait $(cat ${logs_path}/eval_pid.txt)
            rm ${logs_path}/eval_pid.txt
        done
        # wait $(cat ${logs_path}/eval_pid.txt)
        # rm ${logs_path}/eval_pid.txt
        echo "Base agent has finished exploring ${j} iteration"
    done


    # kill the model worker
    echo "Kill the model workers"
    kill -9 $(cat ${logs_path}/worker_pid.txt)
    rm ${logs_path}/worker_pid.txt


    # Part 4: Build contrastive pairs
    echo "Build preference data"
    pm_data_path=${save_path}data_pm/${task}_pm_${i}.json
    explored_traj_path=${save_path}explored_traj/explored_traj_${i}.json
    if [ ! -d ${save_path}data_pm ]; then
        mkdir -p ${save_path}data_pm
    fi
    if [ ! -d ${save_path}explored_traj ]; then
        mkdir -p ${save_path}explored_traj
    fi

    python score.py --task $task --output_path ${explored_traj_path} --traj_path ${step_traj_save_path} --sample_path ${save_path}monte_carlo_sample_iteration_${i}
    global_traj_threshold=0.01 #0.5
    step_traj_threshold=0.01 #0.5
    python construct_preference_data.py --with_plan --task $task --output_path ${pm_data_path} --traj_path ${explored_traj_path} --global_traj --local_traj --traj_threshold ${global_traj_threshold} --step_threshold ${step_traj_threshold}

    # Part 5: Conduct mixture trajectory optimization to learn from incorrect actions
    batch_size=32
    micro_batch_size=2
    node_num=8
    accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))
    if [ ${i} -eq 1 ]; then
        beta=0.1
        lr=1e-6 
    else
        beta=0.5
        lr=5e-7
    fi

    dpo_model_name=${sft_model_name}-dpo-with-paraphrase-lr${lr}-beta${beta}-iter-${i}

    python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20002 fastchat/train/train_dpo.py \
        --model_name_or_path ${save_dir}${cur_model_name} \
        --ref_model_name_or_path ${save_dir}${cur_model_name} \
        --data_path ${pm_data_path} \
        --bf16 True \
        --output_dir ${save_dir}${dpo_model_name} \
        --num_train_epochs 3 \
        --per_device_train_batch_size ${micro_batch_size} \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps ${accumulation_step} \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --save_total_limit 5 \
        --beta ${beta} \
        --learning_rate ${lr} \
        --weight_decay 0.0 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 5 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --model_max_length 4096 \
        --max_prompt_length 512 \
        --max_target_length 3072 \
        --gradient_checkpointing True \
        --lazy_preprocess False

    # Part 6: Evaluate the agent, iter-${i}/
    fs_worker_port=21012
    python -u -m fastchat.serve.model_worker --model-path ${save_dir}${dpo_model_name} --host 127.0.0.1 --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &

    fs_worker_pid=$!
    sleep 360

    # evaluate on the test set
    python -m eval_agent.main --agent_config fastchat --model_name ${dpo_model_name} --exp_config ${task} --split test --override

    # if failed, exit
    if [ $? -ne 0 ]; then
        echo "base agent evaluation failed"
        kill -9 $fs_worker_pid
        exit 1
    fi

    # kill the model worker
    kill -9 $fs_worker_pid

    cur_model_name=${dpo_model_name}
done
