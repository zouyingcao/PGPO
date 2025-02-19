# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


from collections import OrderedDict
import os
import glob
from os.path import join as pjoin
from shutil import copyfile, copytree, rmtree
from typing import Optional, Mapping

from textworld.logic import GameLogic
from textworld.generator.vtypes import VariableType, VariableTypeTree
from textworld.utils import maybe_mkdir, RegexDict

BUILTIN_DATA_PATH = os.path.dirname(__file__)
LOGIC_DATA_PATH = pjoin(BUILTIN_DATA_PATH, 'logic')
TEXT_GRAMMARS_PATH = pjoin(BUILTIN_DATA_PATH, 'text_grammars')


def _maybe_copyfile(src, dest, force=False, verbose=False):
    if not os.path.isfile(dest) or force:
        copyfile(src=src, dst=dest)
    else:
        if verbose:
            print("Skipping {} (already exists).".format(dest))


def _maybe_copytree(src, dest, force=False, verbose=False):
    if os.path.exists(dest):
        if force:
            rmtree(dest)
        else:
            if verbose:
                print("Skipping {} (already exists).".format(dest))
            return

    copytree(src=src, dst=dest)


def create_data_files(dest: str = './textworld_data', verbose: bool = False, force: bool = False):
    """
    Creates grammar files in the target directory.

    Will NOT overwrite files if they alredy exist (checked on per-file basis).

    Parameters
    ----------
    dest :
        The path to the directory where to dump the data files into.
    verbose :
        Print when skipping an existing file.
    force :
        Overwrite all existing files.
    """

    # Make sure the destination folder exists.
    maybe_mkdir(dest)

    # Knowledge base related files.
    _maybe_copytree(LOGIC_DATA_PATH, pjoin(dest, "logic"), force=force, verbose=verbose)

    # Text generation related files.
    _maybe_copytree(TEXT_GRAMMARS_PATH, pjoin(dest, "text_grammars"), force=force, verbose=verbose)


def _to_type_tree(types):
    vtypes = []

    for vtype in sorted(types):
        if vtype.parents:
            parent = vtype.parents[0]
        else:
            parent = None
        vtypes.append(VariableType(vtype.name, vtype.name, parent))

    return VariableTypeTree(vtypes)


def _to_regex_dict(rules):
    # Sort rules for reproducibility
    # TODO: Only sort where needed
    rules = sorted(rules, key=lambda rule: rule.name)

    rules_dict = OrderedDict()
    for rule in rules:
        rules_dict[rule.name] = rule

    return RegexDict(rules_dict)


class KnowledgeBase:
    def __init__(self, logic: GameLogic, text_grammars_path: str):
        self.logic = logic
        self.logic_path = "embedded in game"
        self.text_grammars_path = text_grammars_path

        self.types = _to_type_tree(self.logic.types)
        self.rules = _to_regex_dict(self.logic.rules.values())
        self.constraints = _to_regex_dict(self.logic.constraints.values())
        self.inform7_commands = {i7cmd.rule: i7cmd.command for i7cmd in self.logic.inform7.commands.values()}
        self.inform7_events = {i7cmd.rule: i7cmd.event for i7cmd in self.logic.inform7.commands.values()}
        self.inform7_predicates = {i7pred.predicate.signature: (i7pred.predicate, i7pred.source)
                                   for i7pred in self.logic.inform7.predicates.values()}
        self.inform7_variables = {i7type.name: i7type.kind for i7type in self.logic.inform7.types.values()}
        self.inform7_variables_description = {i7type.name: i7type.definition for i7type in self.logic.inform7.types.values()}
        self.inform7_addons_code = self.logic.inform7.code

    @classmethod
    def default(cls):
        return KB

    @classmethod
    def load(cls,
             target_dir: Optional[str] = None,
             logic_path: Optional[str] = None, grammar_path: Optional[str] = None) -> "KnowledgeBase":
        """ Build a KnowledgeBase from several files (logic and text grammar).

        Args:
            target_dir: Folder containing two subfolders: `logic` and `text_grammars`.
                        If provided, both `logic_path` and `grammar_path` are ignored.
            logic_path: Folder containing `*.twl` files that describe the logic of a game.
            grammar_path: Folder containing `*.twg` files that describe the grammar used for text generation.

        Returns:
            KnowledgeBase object.
        """
        if target_dir:
            logic_path = pjoin(target_dir, "logic")
            grammar_path = pjoin(target_dir, "text_grammars")

        if logic_path is None:
            logic_path = pjoin(".", "textworld_data", "logic")  # Check within working dir.
            if not os.path.isdir(logic_path):
                logic_path = LOGIC_DATA_PATH  # Default to built-in data.

        # Load knowledge base related files.
        paths = glob.glob(pjoin(logic_path, "*.twl"))
        logic = GameLogic.load(paths)

        if grammar_path is None:
            grammar_path = pjoin(".", "textworld_data", "text_grammars")  # Check within working dir.
            if not os.path.isdir(grammar_path):
                grammar_path = TEXT_GRAMMARS_PATH  # Default to built-in data.

        # Load text generation related files.
        kb = cls(logic, grammar_path)
        kb.logic_path = logic_path
        return kb

    def get_reverse_action(self, action):
        r_name = self.logic.reverse_rules.get(action.name)
        if r_name:
            return action.inverse(name=r_name)
        else:
            return None

    @classmethod
    def deserialize(cls, data: Mapping) -> "KnowledgeBase":
        logic = GameLogic.deserialize(data["logic"])
        text_grammars_path = data["text_grammars_path"]
        return cls(logic, text_grammars_path)

    def serialize(self) -> str:
        data = {
            "logic": self.logic.serialize(),
            "text_grammars_path": self.text_grammars_path,
        }
        return data

    def __str__(self) -> str:
        infos = []
        infos.append("logic_path: {}".format(self.logic_path))
        infos.append("grammar_path: {}".format(self.text_grammars_path))
        infos.append("nb_rules: {}".format(len(self.logic.rules)))
        infos.append("nb_types: {}".format(len(self.logic.types)))
        return "\n".join(infos)


# On module load.
KB = KnowledgeBase.load()
