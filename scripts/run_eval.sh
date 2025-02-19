# Evaluate agent on split set
cd ..

task=alfworld
split=dev
fs_worker_port=21012
model_path=./checkpoints_alfworld/Llama-2-7b-hf-alfworld-sft-with-plan
model_name=$(basename "$model_path")

logs_path=./experiments/eval/logs/${model_name}/${task}-${split}
mkdir -p ${logs_path}/

CUDA_VISIBLE_DEVICES=0 python -u -m fastchat.serve.model_worker --model-path ${model_path} --host 127.0.0.1 --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &

fs_worker_pid=$!
sleep 60

# evaluate on the split set
CUDA_VISIBLE_DEVICES=0 python -m eval_agent.main --agent_config fastchat --model_name ${model_name} --exp_config ${task} --split ${split} --output_path outputs/${model_name}/${task}-${split} --override

# if failed, exit
if [ $? -ne 0 ]; then
    echo "base agent evaluation failed"
    kill -9 $fs_worker_pid
    exit 1
fi

# kill the model worker
kill -9 $fs_worker_pid
