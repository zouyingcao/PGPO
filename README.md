# PGPO: Enhancing Agent Reasoning via Pseudocode-style Planning Guided Preference Optimization 
## Structure of This Project

- **`data`**: we include our used ReAct-style expert trajectory data in ```data/ReAct_style_data```. Following the p-code plan generation pipeline, we collect our new expert trajectory data with the incorporation of p-code plans in ```<TASK_NAME>_with_plan.json```. Some prompt examples are provided in ```data/p-code_plan_prompts```. Moreover, we conduct comparitive experiments between p-code plans and nl plans. Generated nl plans are in ```data/sft_data_with_nl_plan``` using prompts in ```data/nl_plan_prompts```.

- **`envs`**: the interaction environment of WebShop and ScienceWorld. We transform the original [WebShop](https://github.com/princeton-nlp/WebShop) repo into a package.

- **`eval_agent`**: the evaluation framework of agent tasks, which is inspired by [MINT](https://github.com/xingyaoww/mint-bench).

- **`eval_results`**: output files about the evaluation results of our PGPO agent.

- **`fastchat`**: training scripts for SFT and DPO. Our experiments are based on Llama-2, Llama-3, Mistral-v0.1/v0.2, Qwen-2.5. We have modified the original [FastChat](https://github.com/lm-sys/FastChat) to support Llama-3, Qwen-2.5 and Mistral-v0.3.

> [!NOTE]  
> This repo is under construction.

## News
- [25/03/15] Our paper was accepted as ACL 2025 Findings.

- [25/02/20] We tried to support the [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) and [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) using FastChat.

- [25/02/18] We tried to support the [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) using FastChat.

- [25/02/17] The PGPO code is released, including the dataset, benchmark, and training implementations.

> [!IMPORTANT]
> We welcome discussions about supporting more models using FastChat.

## Installation

```bash
git clone https://github.com/zouyingcao/PGPO.git
cd PGPO
bash env_setup.sh
```
Our environment is based on previous inspiring works [ETO](https://github.com/Yifan-Song793/ETO) and [IPR](https://github.com/WeiminXiong/IPR). However, we met one error during the installation:
```bash
TypeError: __init__() got an unexpected keyword argument 'dispatch_batches'
```
Recommended solution: You can change the ```create_accelerator_and_postprocess``` function in ```PGPO/lib/python3.9/site-packages/transformers/trainer.py``` by adding the following code before ```self.accelerator = Accelerator(...)```:
```bash
if is_accelerate_available("1.0.0"):
    from accelerate.utils import DataLoaderConfiguration
    dataloader_config = DataLoaderConfiguration(
        split_batches=accelerator_kwargs.pop("split_batches"),
        dispatch_batches=accelerator_kwargs.pop("dispatch_batches"),
        even_batches=accelerator_kwargs.pop("even_batches"),
        use_seedable_sampler=accelerator_kwargs.pop("use_seedable_sampler"),
    )
    accelerator_kwargs["dataloader_config"] = dataloader_config
```
> [!CAUTION]
> Libraries like PyTorch and Flash-attention are based on CUDA 11.8. If you see nonsense in the model outputs, try to reinstall these libraries suitable for your system's CUDA version.

> [!TIP]
> You can ref to: [PyTorch](https://pytorch.org/get-started/previous-versions/), [Flash-attention](https://github.com/Dao-AILab/flash-attention/releases/)

## Plan Generation Pipeline
The ```plan_generation_pipeline.ipynb``` implements the P-code Plan Generation Pipeline for reference. 

![image](https://github.com/user-attachments/assets/691e74c3-76df-4492-88df-acee6970bec8)


## P-code Plan-guided Preference Optimization Pipeline

First, launch the controller of FastChat
```bash
cd scripts
sh run_fastchat.sh
```
Then, build the base agent using SFT on expert trajectories. 
```bash
sh run_sft.sh
```

Next, you can use the SFT-based agent as scorer agent to calculate planning-oriented rewards for expert trajectories.
```bash
sh run_golden_planning_reward.sh
```
Finally, for the iterative agent learning, you can ref to:
```bash
run_pgpo_pipeline.sh
```
Moreover, the bash script ```run_eval.sh``` is used for agent task evaluation (```Optional tasks: alfworld, webshop, textcraft, sciworld```).

![image](https://github.com/user-attachments/assets/e77bdf3d-f13c-42e5-893d-f59899fd67b8)





## Acknowledgement

This repo benefits from [ETO](https://github.com/huggingface/peft), [IPR](https://github.com/huggingface/trl), [AgentGym](https://github.com/WooooDyy/AgentGym) and [FastChat](https://github.com/lm-sys/FastChat). Thanks for their inspiring works.
