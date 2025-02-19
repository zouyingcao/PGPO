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
        }
    ],
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
        elif item['role'] == "human":
            message['role'] = "user"
            message['content'] = item['value']
        messages.append(message)
    return messages

def find_original_game_file(core_part, all_game_files):
    for game_file in all_game_files:
        if core_part in game_file:
            return game_file
    return None

def construct_sample_data(args: argparse.Namespace):
    to_be_sample_data = []
    for file in glob(f"{args.data_path}/*.json"):
        data = json.load(open(file))
        for item in data:
            agent_step_output = item['agent_step_output']
            if "Action: click[Buy Now]" in agent_step_output or "Action: click[buy now]" in agent_step_output:
                continue
            new_item = {}
            new_item['id'] = item['id']
            new_item['iteration'] = item['iteration']
            if 'game_file' in item:
                new_item['game_file'] = item['game_file']
            new_item['agent_step_output'] = agent_step_output
            new_item['agent_step_reward'] = item['agent_step_reward']
            new_item['agent_step_conversations'] = item['agent_step_conversations']
            to_be_sample_data.append(new_item)
    return to_be_sample_data

def sample_monte_carlo_response(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    env_config = exp_config["env_config"]
    task_name = args.exp_config
    
    if 'env_class' in env_config and env_config['env_class'] == 'WebShopEnv':
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)
    elif 'env_class' in env_config and env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']), envStepLimit=200)
    elif 'env_class' in env_config and env_config['env_class'] == 'TextCraftEnv':
        from eval_agent.envs.textcraft_env import TextCraft
        env_config['env'] = TextCraft(minecraft_dir="eval_agent/envs")
    
    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    ignore_task_ids = ignore_task_id[task_name]
    
    # initialize all the tasks    
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks("train", args.part_num, args.part_idx)

    to_be_sample_data = construct_sample_data(args)
    
    final_data = []
    print("Start generating pair_data")
    
    if task_name == "alfworld":
        all_game_files = json.load(open("data/alfworld_game_files.json"))
    
    part_len = len(to_be_sample_data) // args.part_num + 1
    to_be_sample_data = to_be_sample_data[args.part_idx * part_len: min((args.part_idx + 1) * part_len, len(to_be_sample_data))]
    n_tasks = len(to_be_sample_data)
    pbar = tqdm(total=n_tasks)
    default_task = all_tasks.__next__()
    for i in range(n_tasks):
        cur_sample_data = to_be_sample_data[i]
        id = cur_sample_data['id']
        
        if id in ignore_task_ids:
            continue
        
        default_task.task_id = id
        default_task.session_id = id
        
        env: envs.BaseEnv = getattr(envs, env_config["env_class"])(default_task, **env_config)
        
        conversations = cur_sample_data['agent_step_conversations'] # [:2] + cur_sample_data['agent_step_conversations'][10:]
        iteration = cur_sample_data['iteration']
        
        start_time = time.time()

        new_conversations = conversations[2:]

        if task_name == "alfworld":
            game_file = cur_sample_data['game_file']
            core_part = "/".join(game_file.split("/")[3:])
            original_game_file = find_original_game_file(core_part, all_game_files)
            if original_game_file is None:
                continue
            observation, state = env.reset([original_game_file])
        else:
            observation, state = env.reset()
        
        env.state.history = template_change(prompt_for_environment[task_name]) + [new_conversations[0]]
        for i in range(1, len(new_conversations), 2):
            observation, state = env.step(new_conversations[i]['content'])
        
        cur_step_llm_output: str = agent(env.state.history)
        observation, state = env.step(cur_step_llm_output)
        cur_step_llm_reward = state.reward
        cur_step_conversation = deepcopy(state.history)
        
        while not state.finished:
            try:
                llm_output: str = agent(state.history)
            except Exception as e:
                state.success = False
                state.finished = True
                state.terminate_reason = "exceeding maximum input length"
                break
            observation, state = env.step(llm_output)
            if state.finished:
                break
        
        end_time = time.time()
        
        new_item = {
            "id": id,
            "iteration": iteration,
            'agent_step_output': cur_sample_data['agent_step_output'],
            'agent_step_reward': cur_sample_data['agent_step_reward'],
            "agent_step_conversations": cur_sample_data['agent_step_conversations'],
            "agent_conversations": state.history,
            'agent_final_reward': state.success if task_name == "alfworld" else state.reward,
            'time': end_time - start_time,
        }
        
        if task_name == "alfworld":
            new_item['game_file'] = cur_sample_data['game_file']
        
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
        "--agent_path",
        type=str,
        default="./eval_agent/configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="fastchat",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="Llama-2-7b-hf-alfworld-sft-with-plan-explore-0",
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=""
    )
    
    args = parser.parse_args()
    sample_monte_carlo_response(args)