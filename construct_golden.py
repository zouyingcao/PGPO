import json
import argparse

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
        if 'role' in item:
            if item['role'] == "assistant":
                message['from'] = "gpt"
                message['value'] = item['content'].strip()
            elif item['role'] == "user":
                message['from'] = "human"
                message['value'] = item['content'].strip()
        messages.append(message)
    return messages

def main():
    parser = argparse.ArgumentParser("Construct trajectory preference dataset")
    parser.add_argument(
        "--task",
        type=str,
        default="alfworld",
        help="task name",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='data/data_with_plan',
        help="output path of the constructed data",
    )

    args = parser.parse_args()

    golden_original = json.load(open(f"{args.output_path}/explored_traj_{args.task}.json", encoding='utf-8'))
    game_file_to_id = json.load(open(f"data/game_file_to_id.json"))
    golden_after = {}
    for item in golden_original:
        if args.task == "alfworld":
            game_file = item['game_file']
            id = game_file_to_id[game_file]
        else:
            id = item['id']

        iteration = item['iteration']  
        new_item = {
            "id": id,
            "iteration": iteration,
            "gpt_conversations": template_change(item["agent_conversations"]),
            "gpt_reward": item['agent_final_reward'],
            "gpt_step_conversations": template_change(item['agent_step_conversations']),
            "gpt_step_reward": item['agent_step_reward'],
            "monte_carlo_step_reward": item['monte_carlo_step_reward']
        } 
        if args.task == 'alfworld':
            new_item['game_file'] = item['game_file']
        
        golden_after[f"{id}_{iteration}"] = new_item
    json.dump(golden_after, open(f"{args.output_path}/sft_data_{args.task}.json", "w", encoding='utf-8'), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()