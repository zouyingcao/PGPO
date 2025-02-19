import re
import json
import logging
from typing import Tuple

from eval_agent.envs import BaseEnv
from eval_agent.tasks import TextCraftTask
from eval_agent.prompt import prompt_with_icl
from eval_agent.utils.datatypes import State


from argparse import Action
import random
from pydoc import describe
import gymnasium as gym
import re
# from textcraft.utils import ActionFailed, ItemTag, ItemTagWithCount, Recipe, item_id_to_str
from dataclasses import dataclass
# from textcraft.crafting_tree import CraftingTree
from typing import List
import sys

from copy import deepcopy
from math import ceil
import os
from unittest import skip
from typing import Set, Dict
from numpy import rec


logger = logging.getLogger("agent_frame")


@dataclass(frozen=True)
class ItemTag:
    tag: str = None
    item_id: str = None

    @property
    def name(self):
        return self.item_id or self.tag

@dataclass
class ItemTagWithCount:
    item_tag: ItemTag
    count: int

@dataclass(frozen=True)
class Recipe:
    input_items: List[ItemTagWithCount]
    output_item: ItemTagWithCount

    @property
    def recipe_str(self):
        output_str = "craft {} {} using ".format(self.output_item.count,
                                                 item_id_to_str(self.output_item.item_tag.name))
        for input_itemtag_count in self.input_items:
            output_str += "{} {}, ".format(input_itemtag_count.count,
                                           item_id_to_str(input_itemtag_count.item_tag.name))
        output_str = output_str[:-2]
        return output_str
    
class ActionFailed(Exception):
    pass

def item_id_to_str(item_id: str):
    return item_id.replace("minecraft:", "").replace("_", " ")

class CraftingTree:

    def __init__(self, minecraft_dir):
        self.tag_recipes = {} # recipes for tags (i.e. item types)
        self.itemid_recipes: Dict[str, list[Recipe]] = {} # recipes for items
        self.tag_set = set() # set of tags
        self.itemid_set = set()
        self.item_id_to_tag = {} # mapping from item id to tag
        # all the items that could be used to craft an item (down to the base items). Useful to
        # remove cycles
        self.transitive_dependencies = {}
        # minimum depth of recipe tree to craft an item
        self.min_depth = {}
        self._load_recipes(minecraft_dir)
        self.clean_up_recipes()

    def clean_up_recipes(self):
        # make sure every recipe with input tag has craftable recipes or items
        new_items = set()
        for item, recipes in self.itemid_recipes.items():
            # for each recipe
            for recipe in recipes:
                # for each input item
                for input_item in recipe.input_items:
                    input_tag = input_item.item_tag.tag
                    # when only tag is specified
                    if input_item.item_tag.item_id is None:
                        assert input_tag is not None
                        # make sure that the tag is craftable or fetchable
                        item_list = list(self.get_items_with_tags(input_tag))
                        success = False
                        # if an item in this list has a recipe, we are good
                        for item_id in item_list:
                            if item_id in self.itemid_recipes:
                                success = True
                                break
                        # if not, this type can't be crafted, so we convert it to an item
                        if not success:
                            #print("No recipe found for tag: {}! Converting to item".format(input_tag))
                            input_item.item_tag = ItemTag(item_id=input_tag)
                            new_items.add(input_tag)

        # clean up itemid_set and tag_set
        for item in new_items:
            self.itemid_set.add(item)
            self.tag_set.remove(item)

    def _load_recipes(self, minecraft_dir):
        for f in os.listdir(os.path.join(minecraft_dir, 'recipes/')):
            with open(os.path.join(minecraft_dir, 'recipes/', f), "r") as fp:
                recipe_details = json.load(fp)
                input_items = []
                if recipe_details["type"] == "minecraft:crafting_shaped":
                    pattern = recipe_details["pattern"]
                    for key, item in recipe_details["key"].items():
                        count = 0
                        if isinstance(item, list):
                            # print("Ignoring lists for now: {}. Taking first item".format(item))
                            item = item[0]
                        for line in pattern:
                            count += line.count(key)
                        if "item" in item:
                            input_item = ItemTag(item_id=item["item"])
                            self.itemid_set.add(item["item"])
                        elif "tag" in item:
                            input_item = ItemTag(tag=item["tag"])
                            self.tag_set.add(item["tag"])
                        else:
                            print(recipe_details, item)
                            raise ValueError("Unknown item type")
                        input_items.append(ItemTagWithCount(input_item, count))
                elif recipe_details["type"] == "minecraft:crafting_shapeless":
                    item_name_idx = {}
                    for ingredient in recipe_details["ingredients"]:
                        if isinstance(ingredient, list):
                            # print("Ignoring lists for now: {}. Taking first item".format(ingredient))
                            ingredient = ingredient[0]
                        if "item" in ingredient:
                            item_name=ingredient["item"]
                            input_item = ItemTag(item_id=item_name)
                            self.itemid_set.add(item_name)
                        elif "tag" in ingredient:
                            input_item = ItemTag(tag=ingredient["tag"])
                            item_name = ingredient["tag"]
                            self.tag_set.add(ingredient["tag"])
                        else:
                            print(recipe_details)
                            raise ValueError("Unknown item type")
                        if item_name not in item_name_idx:
                            item_name_idx[item_name] = len(input_items)
                            input_items.append(ItemTagWithCount(input_item, 1))
                        else:
                            curr_count = input_items[item_name_idx[item_name]].count
                            # frozen dataclass so can't modify in place
                            input_items[item_name_idx[item_name]] = ItemTagWithCount(input_item,
                                                                                     curr_count + 1)
                else:
                    continue


                recipe_result = recipe_details["result"]
                if isinstance(recipe_result, str):
                    output_item_id = recipe_result
                    output_item_count = 1
                elif "item" in recipe_result:
                    output_item_id = recipe_result["item"]
                    output_item_count = recipe_result.get("count") or 1
                else:
                    print(recipe_details)
                    raise ValueError("Unknown item type")
                self.itemid_set.add(output_item_id)
                # Remove block recipes
                if len(input_items) == 1 and input_items[0].item_tag.name.endswith("_block"):
                    # print("Skipping recipe for {} using {}".format(output_item_id,
                    #                                               input_items[0].item_tag.name))
                    continue
                output_tag = None
                if "group" in recipe_details:
                    output_tag = "minecraft:" + recipe_details["group"]
                    # sometimes the group is the same as the output item id
                    if output_tag != output_item_id:
                        self.tag_set.add(output_tag)
                        self.item_id_to_tag[output_item_id] = output_tag
                
                output_item = ItemTagWithCount(ItemTag(tag=output_tag, item_id=output_item_id),
                                               output_item_count)
                recipe = Recipe(input_items, output_item)
                
                if output_item_id not in self.transitive_dependencies:
                    self.transitive_dependencies[output_item_id] = set()
                skip_recipe = False
                for input_itemtag_count in input_items:
                    input_item_name = input_itemtag_count.item_tag.name
                    if input_item_name in self.transitive_dependencies:
                        if output_item_id in self.transitive_dependencies[input_item_name]:
                            # print("Cycle detected for {} and {}".format(output_item_id, input_item_name))
                            # print("Skipping recipe: {}".format(recipe))
                            skip_recipe = True

                if not skip_recipe:
                    recipe_item_id = output_item.item_tag.item_id
                    if recipe_item_id not in self.itemid_recipes:
                        self.itemid_recipes[recipe_item_id] = [recipe]
                    else:
                        self.itemid_recipes[recipe_item_id].append(recipe)

                    for input_itemtag_count in input_items:
                        input_item_name = input_itemtag_count.item_tag.name
                        self.transitive_dependencies[output_item_id].add(input_item_name)
                        if input_item_name in self.transitive_dependencies:
                            self.transitive_dependencies[output_item_id].update(
                                self.transitive_dependencies[input_itemtag_count.item_tag.name])
                            
                    recipe_tag = output_item.item_tag.tag
                    if recipe_tag is not None:
                        if recipe_tag not in self.tag_recipes:
                            self.tag_recipes[recipe_tag] = [recipe]
                        else:
                            self.tag_recipes[recipe_tag].append(recipe)
                    

    def craft(self, recipe: Recipe) -> ItemTagWithCount:
        if recipe.output_item.item_tag.item_id not in self.itemid_recipes:
            return None
        target_recipes = self.itemid_recipes[recipe.output_item.item_tag.item_id]
        for target_recipe in target_recipes:
            success = True
            # check that input recipe items matches the target recipe items
            input_recipe_items_clone = deepcopy(recipe.input_items)
            for itemtag_count in target_recipe.input_items:
                itemtag = itemtag_count.item_tag
                input_itemtag_count = self.find_matching_item(itemtag, input_recipe_items_clone)
                if input_itemtag_count is None:
                    #raise ActionFailed("Missing Item: {}".format(itemtag.tag))
                    # print("Missing Item: {}".format(itemtag))
                    success = False
                    break
                    
                if input_itemtag_count.count != itemtag_count.count:
                    print("Wrong Item Count for: {}".format(input_itemtag_count))
                    success = False
                    break
                
                input_recipe_items_clone.remove(input_itemtag_count)

            if len(input_recipe_items_clone):
                # print("Extra Input items: {}".format(input_recipe_items_clone))
                success = False
            
            if success:
                return target_recipe.output_item        
            
        return None
        

    def find_matching_item(self, itemtag: ItemTag, input_recipe_items: List[ItemTagWithCount]):
        for input_itemtag_count in input_recipe_items:
            if itemtag.item_id is not None:
                if input_itemtag_count.item_tag.item_id == itemtag.item_id:
                    return input_itemtag_count
            elif itemtag.tag is not None:
                if input_itemtag_count.item_tag.tag == itemtag.tag or \
                    self.item_id_to_tag.get(input_itemtag_count.item_tag.item_id) == itemtag.tag:
                    return input_itemtag_count
        return None
    
    def is_craftable(self, item: str):
        return item in self.itemid_recipes or item in self.tag_recipes
    
    def is_valid_item(self, item: str):
        return item in self.itemid_set

    def is_tag(self, input: str):
        return input in self.tag_set
    
    def get_items_with_tags(self, input_tag: str):
        for item_id, tag in self.item_id_to_tag.items():
            if input_tag == tag:
                yield item_id

    def print_all_recipes(self):
        for item, recipes in self.itemid_recipes.items():
            for recipe in recipes:
                self.print_recipe(recipe)
        for tag, recipes in self.tag_recipes.items():
            for recipe in recipes:
                self.print_recipe(recipe)

    def print_recipe(self, recipe: Recipe):
        print(recipe.recipe_str)

    def traverse_recipe_tree(self, item_name: str, visited_names: Set[str]):
        if item_name in visited_names:
            print("Cycle detected for {}: {}".format(item_name, visited_names))
            return []
        recipes = self.itemid_recipes.get(item_name) or self.tag_recipes.get(item_name) or []
        for recipe in recipes:
            new_visited_names = deepcopy(visited_names)
            for input_itemtag_count in recipe.input_items:
                input_item_name = input_itemtag_count.item_tag.name
                new_visited_names.add(item_name)
                recipes.extend(self.traverse_recipe_tree(input_item_name, new_visited_names))
        return recipes
    
    def collect_item_uses(self):
        item_uses = {}
        for item, recipes in self.itemid_recipes.items():
            for recipe in recipes:
                for input_itemtag in recipe.input_items:
                    if input_itemtag.item_tag.name not in item_uses:
                        item_uses[input_itemtag.item_tag.name] = []

                    item_uses[input_itemtag.item_tag.name].append(recipe)
        for tag, recipes in self.tag_recipes.items():
            for recipe in recipes:
                for input_itemtag in recipe.input_items:
                    if input_itemtag.item_tag.name not in item_uses:
                        item_uses[input_itemtag.item_tag.name] = []
                    item_uses[input_itemtag.item_tag.name].append(recipe)
        return item_uses

    def get_min_depth(self, item_tag: str):
        if item_tag in self.min_depth:
            return self.min_depth[item_tag]
        
        if item_tag in self.itemid_recipes:
            self.min_depth[item_tag] = self.get_min_depth_recipes(self.itemid_recipes[item_tag])
        elif item_tag in self.tag_recipes:
            self.min_depth[item_tag] = self.get_min_depth_recipes(self.tag_recipes[item_tag])
        else:
            self.min_depth[item_tag] = 0
        
        return self.min_depth[item_tag]

    def get_min_depth_recipes(self, recipes):
        depths = []
        for recipe in recipes:
            recipe_depths = []
            for input_itemtag_count in recipe.input_items:
                recipe_depths.append(self.get_min_depth(input_itemtag_count.item_tag.name) + 1)
            # pick the max here since each item has to be built
            depths.append(max(recipe_depths))
        # pick the min here since the model could make the easiest recip
        return min(depths)

    def item_recipes_min_depth(self, min_depth:int):
        for item, recipes in self.itemid_recipes.items():
            item_depth = self.get_min_depth(item)
            if  item_depth >= min_depth:
                yield item, item_depth
    

    def item_recipes_min_items(self, min_items:int):
        for item, recipes in self.itemid_recipes.items():
            for recipe in recipes:
                if len(recipe.input_items) >= min_items:
                    yield item
    
    def item_recipes_min_closure(self, min_closure:int):
        for item, closure in self.transitive_dependencies.items():
            if len(closure) >= min_closure:
                yield item

    def create_recipe_set(self, item_name: str):
        item_uses = self.collect_item_uses()
        recipes = self.traverse_recipe_tree(item_name, set())
        distractors = []
        for recipe in recipes:
            for item in recipe.input_items:
                input_item_name = item.item_tag.name
                if input_item_name in item_uses:
                    input_item_uses_recipes = item_uses[input_item_name]
                    distractors.extend(random.sample(input_item_uses_recipes,
                                                     min(len(input_item_uses_recipes), 10)))

        return (recipes, distractors)


class TextCraft(gym.Env[str, str]):

    def __init__(self, minecraft_dir):
        self.inventory = {}
        self.action_regexes = {
            "craft": r"craft (.*) using (.*)",
            "get": r"get ([0-9]+) (.*)",
            "inventory": r"inventory",
        }
        self.count_regex = r"([0-9]+) (.*)"
        self.crafting_tree = CraftingTree(minecraft_dir=minecraft_dir)

    def step(self, action):
        observation = None
        reward = 0
        terminated = False
        truncated = False
        info = {}
        try:
            for action_type, regex in self.action_regexes.items():
                match = re.match(regex, action)
                if match:
                    if action_type == "craft":
                        recipe = self.extract_recipe(
                            match.group(1), match.group(2))
                        if recipe is None:
                            raise ActionFailed(
                                "Could not extract recipe from {}".format(action))
                        if not self.has_items(recipe.input_items):
                            raise ActionFailed(
                                "Could not find enough items to craft {}".format(recipe.output_item.item_tag.item_id))
                        output_itemtag_count = self.crafting_tree.craft(recipe)
                        if output_itemtag_count is None:
                            raise ActionFailed(
                                "Could not find a valid recipe for {}".format(recipe.output_item))
                        self.remove_items(recipe.input_items)
                        self.add_item(output_itemtag_count.item_tag, output_itemtag_count.count)
                        observation = "Crafted {} {}".format(output_itemtag_count.count,
                                                            output_itemtag_count.item_tag.item_id)
                        if output_itemtag_count.item_tag.item_id == self.goal:
                            reward = 1
                            terminated = True
                    elif action_type == "get":
                        (item, amt) = match.group(2), int(match.group(1))
                        item_obj = self.item_str_to_obj(item)
                        if self.crafting_tree.is_craftable(item_obj.name):
                            raise ActionFailed("Could not find {}".format(item))
                        if self.crafting_tree.is_tag(item_obj.item_id) or \
                            item_obj.item_id is None:
                            raise ActionFailed("Could not find {}".format(item))
                        if not self.crafting_tree.is_valid_item(item_obj.item_id):
                            raise ActionFailed("Could not find {}".format(item))
                        self.add_item(item_obj, amt)
                        observation = "Got {} {}".format(amt, item)
                    elif action_type == "inventory":
                        observation = "Inventory: "
                        if not len(self.inventory.items()): observation += 'You are not carrying anything.'
                        for item, amt in self.inventory.items():
                            observation += "[{}] ({}) ".format(item_id_to_str(item), amt)
                        # observation = observation.rstrip(', ')
                    else:
                        raise NotImplementedError(
                            "Action type {} not implemented".format(action_type))
            if observation is None:
                raise ActionFailed("Could not execute {}".format(action))

        except ActionFailed as e:
            observation = "{}".format(e.args[0])
            reward = 0
            info = {}

        return (observation, reward, terminated, truncated, info)
    
        
    def has_items(self, items:List[ItemTagWithCount]):
        for itemtag_count in items:
            if itemtag_count.item_tag.item_id not in self.inventory or \
                self.inventory[itemtag_count.item_tag.item_id] < itemtag_count.count:
                return False
        return True
    
    def add_item(self, item_tag: ItemTag, amt: int):
        if item_tag.item_id not in self.inventory:
            self.inventory[item_tag.item_id] = 0
        self.inventory[item_tag.item_id] += amt

    def remove_items(self, items: List[ItemTagWithCount]):
        for itemtag_amts in items:
            self.inventory[itemtag_amts.item_tag.item_id] -= itemtag_amts.count
            if self.inventory[itemtag_amts.item_tag.item_id] == 0:
                del self.inventory[itemtag_amts.item_tag.item_id]

    def extract_recipe(self, output_item_str, input_items_str) -> Recipe:
        # check if there is a number in the output item
        m = re.match("([0-9]+) (.*)", output_item_str)
        if m:
            output_item =  self.item_str_to_obj(m.group(2))
            output_item_count = int(m.group(1))
        else:
            output_item = self.item_str_to_obj(output_item_str)
            output_item_count = 1
        output_item_count = ItemTagWithCount(output_item, output_item_count)
        input_items = []
        for input_item_count in input_items_str.split(","):
            match = re.match(self.count_regex, input_item_count.strip())
            if match:
                count = int(match.group(1))
                item_str = match.group(2)
                input_item_obj = self.item_str_to_obj(item_str)
                input_items.append(ItemTagWithCount(input_item_obj, count))
            else:
                raise ActionFailed("Wrong item format: {}".format(input_item_count.strip()))
        return Recipe(input_items=input_items, output_item=output_item_count)
    
    def item_str_to_obj(self, item):
        item_id = "minecraft:" + item.replace(" ", "_")
        if self.crafting_tree.is_tag(item_id):
            return ItemTag(tag=item_id)
        else:
            return ItemTag(item_id=item_id)
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # clean inventory
        self.inventory = {}
        random.seed(seed)
        # item_depth_list = list(self.crafting_tree.item_recipes_min_depth(2))
        # use seed to deterministically select goal 
        # goal_depth = sorted(item_depth_list, key=lambda x: -x[1])[seed % len(item_depth_list)]
        # # self.goal = "minecraft:dark_oak_sign"
        # # print("Goal: {} with depth: {}".format(goal_depth[0], goal_depth[1]))
        # self.goal = goal_depth[0] 
        ######## for fair comparison 
        with open('data/id_to_craft_item.json', 'r') as f:
            id2item = json.load(f)
        self.goal = id2item[str(seed)]
        recipes_set = set()
        distractor_set = set()
        max_distractor = 10
        recipes, distractors = self.crafting_tree.create_recipe_set(self.goal)
        for recipe in recipes:
            recipes_set.add(recipe.recipe_str)
        for distractor in distractors:
            if distractor.recipe_str not in recipes_set:
                distractor_set.add(distractor.recipe_str)

        recipes_list = list(recipes_set) + random.sample(list(distractor_set),
                                                         min(len(distractor_set), max_distractor))
        random.shuffle(recipes_list)
        return "Crafting commands:\n{}\n\nGoal: craft {}.".format("\n".join(recipes_list), 
                                                        item_id_to_str(self.goal)), {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

class TextCraftEnv(BaseEnv):
    def __init__(
        self,
        task: TextCraftTask,
        env: TextCraft,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: TextCraftTask = task
        self.session_id = self.task.session_id
        self.session = {}
        self.env = env
        
        self.state = State()
        self.max_patience = 8
        self.pat_ctr = 0
    
    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        assert action is not None
        return action
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
        except:
            observation = f"Observation: Invalid format. The input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.state.reward = 0
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state
        try:
            observation, reward, done, _, info = self.env.step(action.lstrip('> '))
            observation = f"Observation: {observation}"
            # available_actions = self.env.get_available_actions()
            # observation = f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}"
        except AssertionError:
            observation = 'Observation: Invalid action!'
            done = False

        self.state.history.append({
            "role": "user",
            "content": f"{observation}",
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = 0
        
        # if 'task completed' in action.lower(): done = True; self.state.success = True
        # if 'task failed' in action.lower(): done = True; self.state.success = False
        # if action.startswith('think:'):
        #     observation = 'OK.'
        #     if 'task completed' in action.lower(): done = True; self.state.success = True
        #     if 'task failed' in action.lower(): done = True; self.state.success = False
        # if "Could not" in observation or observation == "OK.": 
        #     self.pat_ctr += 1
        #     if self.pat_ctr == self.max_patience: 
        #         self.state.finished = True
        #         done = True
        if reward > 0: 
            self.state.success = True
            self.state.finished = False
        if done:
            self.state.finished = True
            # self.state.success = True
            self.state.terminate_reason = "done"
            self.state.reward = reward

        return observation, self.state
    
    def reset(self) -> Tuple[str, State]:
        self.state = State()
        obs, info = self.env.reset(self.session_id)
        # commands, cur_task = obs.split('Goal: ')
        cur_task = obs
        observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 0)
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state
