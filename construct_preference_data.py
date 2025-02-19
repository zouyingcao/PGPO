import os
import json
import argparse
import glob
import numpy as np


def template_change(conversation):
    '''
    {
        "role": "user", -> "human"
        "content": "..."
    },
    {
        "role": "assistant", -> "gpt"
        "content": "..."
    }
    '''
    messages = []
    for item in conversation:
        message = {}
        if item['role'] == "assistant":
            message['from'] = "gpt"
            message['value'] = item['content'].strip()
        elif item['role'] == "user":
            message['from'] = "human"
            message['value'] = item['content'].strip()
        messages.append(message)
    return messages

def is_empty_conversations(conversation):
    for item in conversation:
        if item['value'].strip() == "":
            return True
    return False

def build_preference(args):
    win = 0
    tie = 0
    lose = 0
    global_traj = 0
    local_step_traj = 0
    local_entire_traj = 0

    model_name = args.traj_path.split("/")[-3].split(f'-{args.task}')[0].strip()
    if args.with_plan:
        golden_raw = json.load(open(f"data/data_with_plan/{model_name}/sft_data_{args.task}.json", encoding='utf-8'))
    else:
        golden_raw = json.load(open(f"data/sft_data_{args.task}.json", encoding='utf-8'))
    game_file_to_id = json.load(open(f"data/game_file_to_id.json"))
    pm_data = []
    
    explored_traj = json.load(open(args.traj_path, encoding='utf-8'))
    
    step_threshold = args.step_threshold
    global_traj_threshold = args.traj_threshold
    
    if args.global_traj: # towards plan-driven reward
        for item in explored_traj:
            if args.task == "alfworld":
                game_file = item['game_file']
                id = game_file_to_id[game_file]
            else:
                id = item['id']
            
            iteration = item['iteration']
            if iteration != 0: # iteration==0 accoresponding to the whole global traj
                continue
            agent_final_reward = item['agent_final_reward'] # outcome reward
            gpt_reward = golden_raw[f"{id}_0"]['gpt_reward'] # same for all iterations
            gpt_conversations = golden_raw[f"{id}_0"]['gpt_conversations'] # the whole traj without the last feedback
            agent_conversations = template_change(item['agent_conversations']) # the whole traj with the last feedback
            if is_empty_conversations(agent_conversations):
                continue
            if agent_final_reward > gpt_reward + global_traj_threshold:
                win += 1
                global_traj += 1
                pm_data.append({
                    "id": int(f"{id}{iteration}"),
                    "prompt": gpt_conversations[:2+1], # sys_prompt+reply+task_prompt, 2 is the length of the system prompt part
                    "chosen": agent_conversations[2+1: -1], # -1 means without the last feedback
                    "rejected": gpt_conversations[2+1:], # begins with search action response
                })
            elif gpt_reward > agent_final_reward + global_traj_threshold:
                lose += 1
                global_traj += 1
                pm_data.append({
                    "id": int(f"{id}{iteration}"),
                    "prompt": gpt_conversations[:2+1],
                    "chosen": gpt_conversations[2+1:],
                    "rejected": agent_conversations[2+1: -1],
                })
            else:
                # gpt_step_num = len(gpt_conversations) - 2 # gpt conversations are zero-shot, so without icl examples
                # agent_step_num = len(agent_conversations) - 1 - 2 # the length '2'(system prompt)
                # # with the same outcome reward, traj with fewer reasoning steps wins!
                # if gpt_step_num > agent_step_num:
                #     win += 1
                #     global_traj += 1
                #     pm_data.append({
                #         "id": int(f"{id}{iteration}"),
                #         "prompt": gpt_conversations[:2+1],
                #         "chosen": agent_conversations[2+1: -1], # [...,-1]: ignore the last feedback
                #         "rejected": gpt_conversations[2+1:],
                #     })
                # elif gpt_step_num < agent_step_num:
                #     lose += 1
                #     global_traj += 1
                #     pm_data.append({
                #         "id": int(f"{id}{iteration}"),
                #         "prompt": gpt_conversations[:2+1],
                #         "chosen": gpt_conversations[2+1:],
                #         "rejected": agent_conversations[2+1: -1],
                #     })
                # else:
                    tie += 1
    
    if args.local_traj: # towards plan-following rewards
        for item in explored_traj:
            if args.task == "alfworld":
                game_file = item['game_file']
                id = game_file_to_id[game_file]
            else:
                id = item['id']
                
            iteration = item['iteration']
            if iteration != 1: # PGPO takes the first two steps as representative
                continue
            
            gpt_step_conversations = golden_raw[f"{id}_{iteration}"]['gpt_step_conversations'] # conversation before the step (contain the step action to be judged)
            gpt_step_reward = golden_raw[f"{id}_{iteration}"]['monte_carlo_step_reward']
            gpt_final_reward = golden_raw[f"{id}_{iteration}"]['gpt_reward']
            agent_step_reward = item['monte_carlo_step_reward']
            agent_final_reward = item['agent_final_reward']
            
            gpt_step_before_length = len(gpt_step_conversations)
            agent_step_before_length = len(gpt_step_conversations)
            
            agent_conversations = template_change(item['agent_conversations'])
            agent_step_conversations = item['agent_step_conversations']
            gpt_conversations = golden_raw[f"{id}_{iteration}"]['gpt_conversations']

            gpt_step_num = len(gpt_conversations) - gpt_step_before_length
            agent_step_num = len(agent_conversations) - 1 - agent_step_before_length
                
            if is_empty_conversations(agent_conversations):
                continue

            if agent_final_reward >= gpt_final_reward + global_traj_threshold:
                if agent_step_reward >= gpt_step_reward + step_threshold:
                    win += 1
                    local_entire_traj += 1
                    pm_data.append({
                        "id": int(f"{id}{iteration}"),
                        "prompt": gpt_conversations[:gpt_step_before_length-2],
                        "chosen": agent_conversations[agent_step_before_length-2: -1],
                        "rejected": gpt_conversations[gpt_step_before_length-2:],
                    })
            elif gpt_final_reward >= agent_final_reward + global_traj_threshold:
                if gpt_step_reward >= agent_step_reward + step_threshold:
                    lose += 1
                    local_entire_traj += 1
                    pm_data.append({
                        "id": int(f"{id}{iteration}"),
                        "prompt": gpt_conversations[:gpt_step_before_length-2],
                        "chosen": gpt_conversations[gpt_step_before_length-2:],
                        "rejected": agent_conversations[agent_step_before_length-2: -1],
                    })
            # with the same step reward, traj with fewer reasoning steps wins!
            # elif gpt_final_reward == agent_final_reward and gpt_step_reward == agent_step_reward:
            #     if gpt_step_num > agent_step_num:
            #         win += 1
            #         local_entire_traj += 1
            #         pm_data.append({
            #             "id": int(f"{id}{iteration}"),
            #             "prompt": gpt_conversations[:gpt_step_before_length-2],
            #             "chosen": agent_conversations[agent_step_before_length-2: -1],
            #             "rejected": gpt_conversations[gpt_step_before_length-2:],
            #         })
            #     elif gpt_step_num < agent_step_num:
            #         lose += 1
            #         local_entire_traj += 1
            #         pm_data.append({
            #             "id": int(f"{id}{iteration}"),
            #             "prompt": gpt_conversations[:gpt_step_before_length-2],
            #             "chosen": gpt_conversations[gpt_step_before_length-2:],
            #             "rejected": agent_conversations[agent_step_before_length-2: -1],
            #         })
            #     else:
            #          tie += 1
            else:
                tie += 1
    
    json.dump(pm_data, open(args.output_path, "w", encoding='utf-8'), indent=4)  
    print(f"win: {win}, tie: {tie}, lose: {lose}")
    print(f"global_traj: {global_traj}, local_step_traj: {local_step_traj}, local_entire_traj: {local_entire_traj}")          

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="alfworld",
        help="task name",
    )
    parser.add_argument(
        "--with_plan",
        action="store_true",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='test.json',
        help="output path of the trajectory preference dataset",
    )
    parser.add_argument(
        "--traj_path",
        type=str,
        default="experiments/Llama-2-7b-hf-alfworld-sft-with-plan/explored_traj/explored_traj_1.json",
        help="path of the trajectory to be judged",
    )
    parser.add_argument(
        "--global_traj",
        action="store_true",
        help="use global trajectory"
    )
    parser.add_argument(
        "--local_traj",
        action="store_true",
        help="use local trajectory"
    )
    parser.add_argument(
        "--step_threshold",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--traj_threshold",
        type=float,
        default=0.01
    )
    
    args = parser.parse_args()
    build_preference(args)

if __name__ == "__main__":
    main()
