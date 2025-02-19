import os
import re
import json
import requests
import argparse
import glob
import numpy as np
from dataclasses import dataclass

tasks_with_subgoal_reward = ["sciworld", "webshop"]

@dataclass
class IdeaLabLLMOpenAI:
    # api_url: str = "https://idealab.xxx"
    api_url: str = "https://aliyuque.xxxx"
    api_key: str = "xxxx"
    model_name: str = "gpt-4o-0513"
    max_tokens: int = 4096
    temperature: float = 0

    @property
    def _llm_type(self) -> str:
        return f"IdeaLabLLM {self.model_name}"

    def _call(self,
              prompt: str,
            #   stop: Optional[List[str]] = None,
            #   run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs):
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "maxTokens": self.max_tokens,
            "temperature": self.temperature
        }

        headers = {"X-AK": f"{self.api_key}", "accept": "*/*", "Content-Type": "application/json"}

        data = {
            "platformInput": {
                "model": self.model_name
            }, "messages": [
                {
                    "role": "user",
                    "content": prompt
                }], "toolChoice": "auto"
        }

        response = requests.post(self.api_url, json=data, headers=headers, timeout=1500).json()

        if response['success']:
            # print(response)
            response_data = response['data']['choices'][0]["message"]["content"]
            return response_data
        else:
            raise Exception(f"Failed to call API: {response['success']} {response}")

def prepare_plan(task_name, traj_path):
    llm = IdeaLabLLMOpenAI(api_url = "https://idealab.xxx",
                        api_key = "xxxx",
                        model_name = 'gpt-4o-0513',
                        max_tokens = 4096,
                        temperature = 0)
    game_file_to_id = json.load(open(f"data/game_file_to_id.json"))
    plan_prompt_file = f'data/plan_prompts/{task_name}_plan_filled_prompts.json'
    with open(plan_prompt_file, 'r', encoding='utf-8') as f:
        pdict = json.load(f)
    
    if task_name == "alfworld":
        plan_levels = [p for p in pdict.keys() if p != 'room']
        plan_prompt = "Here are some examples.\n" + pdict['room'] + '\n\n'
        for eg in plan_levels:
            plan_prompt += pdict[eg] + '\n\n'
        plan_prompt += "Here is the task.\n<task>.\nBased on the previously shown plans, come up with an abstract plan to perform this task in a couple of steps (NOT more than 3-4). Constraints:\n\
            - The robot can hold/take/put only one object at a time to a location.\n\
            - Ensure each step can be understood independently and mentions the name of object.\n\
            - Keep the step as abstract or generic as possible without mentioning location.\n\
            - DO NOT make assumptions about finding an object in a particular location or receptacle (if possible).\n\n\
            Your output must be strictly follow this format:\nStep 1: ... \nStep 2: ... \nStep 3: ... \n..."
        
    results = []
    with open(traj_path, 'r', encoding='utf-8') as f:
        trajs = json.load(f)
        for line in trajs:
            # if task_name == "alfworld":
            #     game_file = line['game_file']
            #     id = game_file_to_id[game_file]
            # else:
            #     id = line['id']
            id = line['id']
            task = line['conversations'][2]["value"]
            response = llm._call(plan_prompt.replace('<task>', task))
            plan = "Step 1" + response.split("Step 1:")[1].strip('\n')
            result = {'id': id, 'plan': plan}
            results.append(result)
    with open(f"data/{task_name}_plan.json", "w", encoding='utf-8') as new_file:
        json.dump(results, new_file, ensure_ascii=False, indent=4)

def process_step(step_content, task_name):
    subgoal = None
    if task_name=="alfworld":
        # for Put Two task
        step_content = step_content.replace("first ", "").replace("second ", "")

        if "Find and take" in step_content:
            subgoal = "You pick up the " + step_content.split("Find and take")[1].strip()
        elif "Clean" in step_content:
            subgoal = "You clean the " + step_content.split("with")[0].split("using")[0].split("Clean")[1].strip()
        elif "Heat" in step_content:
            subgoal = "You heat the " + step_content.split("with")[0].split("using")[0].split("Heat")[1].strip()
        elif "Cool" in step_content:
            subgoal = "You cool the " + step_content.split("with")[0].split("using")[0].split("Cool")[1].strip()
        elif "Use" in step_content:
            subgoal = "You turn on the" + step_content.split("to")[0].strip().split("Use")[1].strip()
        elif "Open" in step_content:
            subgoal = "You open the " + step_content.split("Open")[1].strip()
        elif "Go to" in step_content or "Move to " in step_content:
            subgoal = "On the " + step_content.split("Move to")[-1].split("Go to")[-1].strip()
        elif "Put" in step_content:
            subgoal = "You put the " + step_content.split("Put")[1].strip()
    
    return "Observation: " + subgoal

def achieve_subgoal(feedback, plan_step):
    if plan_step in feedback: # Take, Clean, Heat, Cool, Examine
        return True
    elif "Put" in plan_step: # Place
        pattern = r'put\s+(.*?)\s*(?:in|on|in/on)\s+(.*?)$'
        plan_item, plan_place = re.search(pattern, plan_step).groups()
        observe_item, observe_place = re.search(pattern, feedback).groups()
        if plan_item in observe_item and plan_item in observe_place:
            return True
        else:
            return False
    else:
        return False

def achieve_sucess(task_name, reward):
    if task_name == "alfworld": 
        return reward
    elif task_name == "webshop": 
        if reward == 1:
            return 1
        else:
            return 0
    elif task_name == "scienceworld":
        if reward >= 1:
            return 1
        else:
            return 0
    else:
        print(f'Task {task_name} need to be added!')

ignore_task_id = {
    #[365, 6066, 6700, 8617, 9067], 
    "alfworld":[],
    "webshop":[],
    "textcraft":[73, 78, 82, 83, 89, 286, 293, 294, 295, 296, 491, 492, 501, 502, 513],
    "sciworld":[],
}

def cal_step_reward(args):
    # sampled_traj
    traj_path = args.sample_path
    # plan
    # if args.task not in tasks_with_subgoal_reward:
    #     plan_dict = {}
    #     for item in json.load(open(args.plan_path, encoding='utf-8')):
    #         plan = item['plan'].strip().strip('\n')
    #         steps = plan.split('\n')
    #         step_list = []
    #         for step in steps:
    #             match = re.match(r'Step (\d+):? (.*)', step)
    #             if match:
    #                 # step_number = int(match.group(1))
    #                 step_content = match.group(2)
    #                 step_list.append(process_step(step_content,args.task))
    #         plan_dict[item['id']]=step_list

    results = {}
    results_original_reward = {}
    # results_subgoal_reward = {}
    sample_num = len(glob.glob(traj_path + "/*"))
    game_file_to_id = json.load(open(f"data/game_file_to_id.json"))
    tobe_ignore = ignore_task_id[args.task]
    for i in range(sample_num):
        paths = glob.glob(f"{traj_path}/sampled_traj_{i}/*.json")
        cur_results = []
        for path in paths:
            cur_results.extend(json.load(open(path, encoding='utf-8')))
        for item in cur_results:
            if args.task == "alfworld":
                game_file = item['game_file']
                id = game_file_to_id[game_file]
            else:
                id = item['id']
            iteration = item['iteration']
            id_iteration = f"{id}_{iteration}"
            if id_iteration not in results:
                results[id_iteration] = [item['agent_final_reward']]
            else:
                results[id_iteration].append(item['agent_final_reward']) # achieve_sucess(args.task,x)
            if id_iteration not in results_original_reward:
                results_original_reward[id_iteration] = item['agent_step_reward']
            else:
                assert results_original_reward[id_iteration] == item['agent_step_reward']
            
            # judge_traj = item['agent_step_conversations']
            # sample_traj = item['agent_conversations']
            
            # if args.task in tasks_with_subgoal_reward:
            #     subgoal_reward = [item['agent_final_reward']]
            # else:
            #     plan = plan_dict[id] # for alfworld
            #     if args.task == "alfworld" and item['agent_final_reward'] == 1: # success
            #         subgoal_reward = [1./len(plan)]*len(plan)
            #     else:
            #         subgoal_reward = [0]*len(plan)
            #         for conv in sample_traj[len(judge_traj)-1:]:
            #             if conv['role'] == "user": # feedback
            #                 for subgoal_id, subgoal in enumerate(plan):
            #                     if achieve_subgoal(conv["content"],subgoal):
            #                         subgoal_reward[subgoal_id] = 1./len(plan)
            #                         break
            # if id_iteration not in results_subgoal_reward:
            #     results_subgoal_reward[id_iteration] = [sum(subgoal_reward)]
            # else:
            #     results_subgoal_reward[id_iteration].append(sum(subgoal_reward))

    final_results = {}
    for key, value in results.items():
        final_results[key] = {
            # can be improved
            "monte_carlo_reward": np.mean(value) , # + np.mean(results_subgoal_reward[key])
            "env_reward": results_original_reward[key]
        }
        
    output_data = []
    for file in glob.glob(f"{args.traj_path}/*.json"):
        data = json.load(open(file, encoding='utf-8'))
        for item in data:
            # id = item['id']
            if args.task == "alfworld":
                game_file = item['game_file']
                id = game_file_to_id[game_file]
            else:
                id = item['id']
            if id in tobe_ignore:
                continue
            iteration = item['iteration']
            id_iteration = f"{id}_{iteration}"
            
            if id_iteration in final_results:
                item['monte_carlo_step_reward'] = final_results[id_iteration]['monte_carlo_reward']
            else:
                item['monte_carlo_step_reward'] = item['agent_step_reward']
            output_data.append(item)
                
    return output_data

def cal_step_reward_for_double(args):
    # sampled_traj
    traj_path = args.sample_path

    results = {}
    results_original_reward = {}
    results_plan_reward = {}
    sample_num = len(glob.glob(traj_path + "/*"))
    game_file_to_id = json.load(open(f"data/game_file_to_id.json"))
    tobe_ignore = ignore_task_id[args.task]
    for i in range(sample_num):
        paths = glob.glob(f"{traj_path}/sampled_traj_{i}/*.json")
        cur_results = []
        for path in paths:
            cur_results.extend(json.load(open(path, encoding='utf-8')))
        for item in cur_results:
            if args.task == "alfworld":
                game_file = item['game_file']
                id = game_file_to_id[game_file]
            else:
                id = item['id']
            iteration = item['iteration']
            id_iteration = f"{id}_{iteration}"
            if id_iteration not in results:
                results[id_iteration] = [item['agent_final_reward']]
            else:
                results[id_iteration].append(item['agent_final_reward']) # achieve_sucess(args.task,x)
            if id_iteration not in results_original_reward:
                results_original_reward[id_iteration] = item['agent_step_reward']
            else:
                if iteration == 0:
                    results_plan_reward[id_iteration] = item['agent_step_reward']
                else:
                    assert results_original_reward[id_iteration] == item['agent_step_reward']
            

    final_results = {}
    for key, value in results.items():
        final_results[key] = {
            # can be improved
            "monte_carlo_reward": np.mean(value[:sample_num]) , # + np.mean(results_subgoal_reward[key])
            "env_reward": results_original_reward[key]
        }
    final_results_plan = {}
    for key, value in results_plan_reward.items():
        if len(results[key][sample_num:])!=sample_num:
            print(key)
        final_results_plan[key] = {
            # can be improved
            "monte_carlo_reward": np.mean(results[key][sample_num:]) , # + np.mean(results_subgoal_reward[key])
            "plan_reward": value,
        }
        
    output_data = []
    for file in glob.glob(f"{args.traj_path}/*.json"):
        data = json.load(open(file, encoding='utf-8'))
        for item in data:
            # id = item['id']
            if args.task == "alfworld":
                game_file = item['game_file']
                id = game_file_to_id[game_file]
            else:
                id = item['id']
            if id in tobe_ignore:
                continue
            iteration = item['iteration']
            id_iteration = f"{id}_{iteration}"
            if "paraphrase" in file:
                if id_iteration in final_results:
                    item['monte_carlo_step_reward'] = final_results_plan[id_iteration]['monte_carlo_reward']
                else:
                    item['monte_carlo_step_reward'] = item['agent_step_reward']
            else:
                if id_iteration in final_results:
                    item['monte_carlo_step_reward'] = final_results[id_iteration]['monte_carlo_reward']
                else:
                    item['monte_carlo_step_reward'] = item['agent_step_reward']
            output_data.append(item)
                
    return output_data

def main():
    parser = argparse.ArgumentParser("Construct trajectory preference dataset")
    parser.add_argument(
        "--task",
        type=str,
        default="alfworld",
        help="task name",
    )
    parser.add_argument(
        "--golden_traj_path",
        type=str,
        default="data/alfworld_sft.json",
        help="path of the golden trajectory",
    )
    parser.add_argument(
        "--plan_path",
        type=str,
        default="data/alfworld_plan.json",
        help="path of the task plan with subgoals",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='experiments/Llama-2-7b-hf-sft-with-plan/explored_traj/explored_traj_1.json',
        help="output path of the trajectory preference dataset",
    )
    parser.add_argument(
        "--traj_path",
        type=str,
        default="experiments/Llama-2-7b-hf-sft-with-plan/Llama-2-7b-hf-sft-with-plan-explore",
        help="task name",
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default="experiments/Llama-2-7b-hf-sft-with-plan/monte_carlo_sample_iteration_1"
    )
    parser.add_argument(
        "--with_double",
        action="store_true",
    )

    args = parser.parse_args()

    if args.plan_path == None and args.task not in tasks_with_subgoal_reward:
        prepare_plan(args.task, args.golden_traj_path)
        args.plan_path = f"data/{args.task}_plan.json"
    
    if args.with_double:
        explored_traj = cal_step_reward_for_double(args)
    else:
        explored_traj = cal_step_reward(args)
    json.dump(explored_traj, open(args.output_path, "w", encoding='utf-8'), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
