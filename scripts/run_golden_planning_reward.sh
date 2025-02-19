
cd ..

model_name=Llama-2-7b-hf
task=alfworld

worker_num=8
sample_num_workers=8 # number of workers for launching models in exploration

model_path=./llm_ckpt/ # path to the original LLM
save_dir=./checkpoints_${task}/    # checkpoint save path
save_path=./experiments/${model_name}-${task}-sft-with-plan-golden-planning-reward/  # output save path
logs_path=${save_path}logs

mkdir -p ${save_path}
mkdir -p ${logs_path}/

sft_data_path="data/${task}_sft_with_plan.json"
sft_model_name=${model_name}-${task}-sft-with-plan

translated_traj_save_path=./data/data_with_plan/${model_name}/${task}
if [ -d ${translated_traj_save_path} ]; then
    rm -r ${translated_traj_save_path}
fi
mkdir -p ${translated_traj_save_path}
if [ -f "${logs_path}/eval_pid.txt" ]; then
    rm ${logs_path}/eval_pid.txt
fi

for (( j = 0; j < $worker_num; j++ )); do
    python3 translate2pair.py --with_plan --exp_config ${task} --part_num $worker_num --part_idx ${j} --save_path ${translated_traj_save_path}  >> ${logs_path}/gen_response_worker-${j}.log 2>&1 &
    echo $! >> ${logs_path}/eval_pid.txt
done
# Wait all workers done for sampling trajs
wait $(cat ${logs_path}/eval_pid.txt)
rm ${logs_path}/eval_pid.txt
echo "The translate2pair finished"

monte_carlo_explore_model_name=${sft_model_name}-monte-carlo-explore
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
        # sample trajectories
        monte_carlo_sample_save_path=${save_path}monte_carlo_sample/sampled_traj_$((j*per_iteration_num+k))
        for ((l=0;l<$sample_workers; l++)); do
            output_path=${monte_carlo_sample_save_path}/
            if [ -d ${output_path} ]; then
                rm -r ${output_path}
            fi
            mkdir -p ${output_path}
            python monte_carlo_sample.py --agent_config fastchat_explore --model_name ${monte_carlo_explore_model_name}-$((l%sample_num_workers)) --exp_config ${task}  --part_num ${sample_workers} --part_idx ${l} --save_path ${output_path} --data_path ${translated_traj_save_path} >> ${logs_path}/gen_response_worker-$((j*per_iteration_num+k))-${l}.log 2>&1 &
            echo $! >> ${logs_path}/eval_pid.txt
        done
        wait $(cat ${logs_path}/eval_pid.txt)
        rm ${logs_path}/eval_pid.txt
    done
    echo "Base agent has finished exploring ${j} iteration"
done


# kill the model worker
echo "Kill the model workers"
kill -9 $(cat ${logs_path}/worker_pid.txt)
rm ${logs_path}/worker_pid.txt


# caluculate planning rewards
echo "Caluculate plan-following rewards"
explored_traj_path=data/data_with_plan/${model_name}
python score.py --task $task --output_path ${explored_traj_path}/explored_traj_${task}.json --traj_path ${translated_traj_save_path} --sample_path ${save_path}monte_carlo_sample

# constructed the used format
python construct_golden.py --task $task --output_path ${explored_traj_path}
