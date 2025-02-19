import os
import json

folder_path = './outputs/Llama-2-7b-hf-alfworld-sft-with-plan/alfworld-dev'

rewards = []
success_num = 0.
step_num = 0.
step1_num = 0.
reward1_num = 0.
error_num = 0
invalid_num = 0
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            rewards.append(data['meta']['reward'])
            step_num += data['meta']["steps"]
            if data['meta']['reward']==1:
                reward1_num += 1
                step1_num += data['meta']["steps"]
            if data['meta']["terminate_reason"]=="success":
                success_num += 1
            elif data['meta']["terminate_reason"]=="max_steps":
                print('max_steps:',filename)
            else:
                print(filename)
                error_num += 1
            for i in range(len(data['conversations'])):
                if i!=0 and 'invalid' in data['conversations'][i]['value'] or 'Nothing happens' in data['conversations'][i]['value']:
                    invalid_num+=1
                    break
print(folder_path)
print(error_num)
print(len(rewards))
print(sum(rewards) / len(rewards))
print(success_num / len(rewards))
print('总步数均值:', step_num / len(rewards))
print('成功案例的步数均值:', step1_num / reward1_num)
print('reward=1的比例:', reward1_num / len(rewards))
print('invalid_action_rate:', invalid_num / len(rewards))