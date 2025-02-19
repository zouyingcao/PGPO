import os

os.environ["OPENAI_API_KEY"] = "sb-xxxx"

import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
from copy import deepcopy

from webshop.web_agent_site.envs import WebAgentTextEnv
from glob import glob

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State
import time

prompt_for_environment ={
    "webshop": [{
            "from": "human",
            "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
        },
        {
            "from": "gpt",
            "value": "OK"
        }],
    "alfworld": [{
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\\nAction: your next action\".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        }
    ],
    "sciworld":[
        {
            "from": "human",
            "value": "You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.\n\nThe available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"
        },
        {
            "from": "gpt",
            "value": "OK"
        }
    ],
    "textcraft":[
        {
            "from": "human",
            "value": "You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format \"craft [target object] using [input ingredients]\".\nEvery round I will give you an observation, you have to respond an action based on the state and instruction. You can \"get\" an object (ingredients) from the inventory or the environment, look-up the game inventory by \"inventory\", or \"craft\" (target) using any of the crafting commands. You can use ONLY these crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like \"planks\", you can use special types of the same ingredient e.g. \"dark oak planks\" in the command instead.\nYour response should use the following format:\n\nThought:  ... \nAction:  ... "
        },
        {
            "from": "gpt",
            "value": "OK. I'll follow your instructions and try my best to solve the task."
        }
    ]
}

ignore_task_id = {
    "alfworld":[],
    "webshop":[365, 6066, 6700, 8617, 9067],
    "textcraft": [73, 78, 82, 83, 89, 286, 293, 294, 295, 296, 491, 492, 501, 502, 513],
    "sciworld":[],
}


def template_change(conversation):
    messages = []
    for item in conversation:
        message = {}
        if item['from'] == "gpt":
            message['role'] = "assistant"
            message['content'] = item['value']
        else:
            message['role'] = "user"
            message['content'] = item['value']
        messages.append(message)
    return messages


def construct_llm_data(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)

    env_config = exp_config["env_config"]
    task_name = args.exp_config
    
    if 'env_class' in env_config:
        if env_config['env_class'] == 'WebShopEnv':
            env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)
        elif env_config['env_class'] == 'SciWorldEnv':
            from scienceworld import ScienceWorldEnv
            from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
            sciworld_monkey_patch()
            env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']), envStepLimit=200)
        elif env_config['env_class'] == 'TextCraftEnv':
            from eval_agent.envs.textcraft_env import TextCraft
            env_config['env'] = TextCraft(minecraft_dir="eval_agent/envs")
          
    # initialize all the tasks    
    task_config: Dict[str, Any] = exp_config["task"]
    
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    
    all_tasks, n_tasks = task_class.load_tasks("train", args.part_num, args.part_idx)
    
    if args.with_plan:
        sft_data = json.load(open(f"data/{task_name}_sft_with_plan.json"))
    else:
        sft_data = json.load(open(f"data/{task_name}_sft.json"))
    temp = {}
    for item in sft_data:
        if task_name == "textcraft":
            temp[int(item['item_id'].split('_')[-1].strip())] = item
        else:
            temp[item['id']] = item
    sft_data = temp
    
    final_data = []
    
    print("Start generating pair_data")

    tobe_ignore = ignore_task_id[task_name]
    if task_name == "alfworld":
        new_sft_data = {}
        for value in sft_data.values():
            new_sft_data[value['game_file']] = value
        # time.sleep(60)
        # data/json_2.1.1/train/pick_and_place_simple-xx-406/trial_xx_136741

    pbar = tqdm(total=n_tasks)
    for task in all_tasks:
        if task.task_id in tobe_ignore:
            pbar.update(1)
            continue
        env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
        
        if task_name == "alfworld":
            # eval_agent/data/alfworld/json_2.1.1/train/pick_cool_then_place_in_recep-xx-6/trial_xx_012007/xx.pddl
            game_file = task.game_file.split('json_2.1.1/')[1]
            game_file = "/".join(["data"]+["json_2.1.1"]+game_file.split('/')[:-1])
            if game_file in new_sft_data:
                cur_sft_data = new_sft_data[game_file]
            else:
                pbar.update(1)
                continue
        else:
            cur_sft_data = sft_data[task.task_id]

        conversations = cur_sft_data['conversations']
        iteration_nums = 2 # total_iteration_nums is len(conversations)//2-1, while PGPO only consider first two turns.
        
        for j in range(iteration_nums):
            if args.global_only and j != 0:
                continue
            start_time = time.time()
            
            new_conversations = conversations[2 :2*j+3]
            observation, state = env.reset()
            env.state.history = template_change(prompt_for_environment[task_name] + [new_conversations[0]])
            for i in range(1, len(new_conversations), 2):
                observation, state = env.step(new_conversations[i]['value'])
            
            # cur_step_llm_output: str = agent(env.state.history)
            cur_step_llm_output = conversations[2*j+3]['value'] # gpt output
            observation, state = env.step(cur_step_llm_output)
            cur_step_llm_reward = state.success if task_name=="alfworld" else state.reward
            cur_step_conversation = deepcopy(state.history)

            new_item = {
                "id": task.task_id,
                "iteration": j,
                'agent_step_output': cur_step_llm_output,
                'agent_step_reward': cur_step_llm_reward,
                "agent_step_conversations": cur_step_conversation,
                "agent_conversations": template_change(conversations),
                'agent_final_reward': cur_sft_data['reward'] if task_name == "webshop" or task_name == "sciworld" else 1, # since webshop and sciworld provide dense rewards
                'time': "human",
            }
            
            if task_name == "alfworld":
                new_item['game_file'] = cur_sft_data['game_file']
            
            final_data.append(new_item)
        pbar.update(1)
    pbar.close()
    
    json.dump(final_data, open(f"{args.save_path}/{task_name}_traj_{args.part_idx}.json", "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./eval_agent/configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="alfworld",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    parser.add_argument(
        "--global_only",
        action="store_true",
    )
    parser.add_argument(
        "--with_plan",
        action="store_true",
    )

    args = parser.parse_args()
    construct_llm_data(args)