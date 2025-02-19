cd ..

model_name=Llama-2-7b-hf
task=alfworld

node_num=8  # number of GPUs for SFT training

model_path=./llm_ckpt/ # path to the original LLM
save_dir=./checkpoints_${task}/    # checkpoint save path
save_path=./experiments/${model_name}-${task}-sft-with-pcode-plan/  # output save path
logs_path=${save_path}logs

if [ -d ${save_path} ]; then
    rm -r ${save_path}
fi
mkdir -p ${save_path}
mkdir -p ${logs_path}/

export WANDB_MODE=offline

# Part 1: SFT stage
sft_data_path="data/${task}_sft_with_plan.json"
batch_size=32
micro_batch_size=4
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

sft_model_name=${model_name}-${task}-sft-with-plan

python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20003 fastchat/train/train.py \
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

# Part 2: Evaluation
fs_worker_port=21012
logs_path=./experiments/eval/logs/${sft_model_name}/${task}
mkdir -p ${logs_path}/

CUDA_VISIBLE_DEVICES=0 python -u -m fastchat.serve.model_worker --model-path ${save_dir}${sft_model_name} --host 127.0.0.1 --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &

fs_worker_pid=$!
sleep 60

# evaluate on the test set if exists
CUDA_VISIBLE_DEVICES=0 python -m eval_agent.main --agent_config fastchat --model_name ${sft_model_name} --exp_config ${task} --split test --output_path outputs/${sft_model_name}/${task}-test --override
# evaluate on the dev test
CUDA_VISIBLE_DEVICES=0 python -m eval_agent.main --agent_config fastchat --model_name ${sft_model_name} --exp_config ${task} --split dev --output_path outputs/${sft_model_name}/${task}-dev --override

# if failed, exit
if [ $? -ne 0 ]; then
    echo "base agent evaluation wo vllm failed"
    kill -9 $fs_worker_pid
    exit 1
fi

# kill the model worker
kill -9 $fs_worker_pid