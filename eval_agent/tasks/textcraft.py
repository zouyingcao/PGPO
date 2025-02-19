import json
import logging
import random
from typing import List, Tuple, Any

logger = logging.getLogger("agent_frame")

from eval_agent.tasks.base import Task


class TextCraftTask(Task):
    task_name = "textcraft"

    def __init__(
        self,
        session_id: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session_id = session_id
    
    @classmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int = -1) -> Tuple[List[Task], int]:
        if split == 'train':
            idxs = json.load(open("eval_agent/data/textcraft/textcraft_train.json"))
        elif split == 'other':
            idxs = json.load(open("eval_agent/data/textcraft/textcraft_other.json"))
        elif split == 'out':
            idxs = list(range(540, 801))
        else: # unseen
            idxs = json.load(open("eval_agent/data/textcraft/textcraft_test.json"))
        if part_num == 1:
            idxs = idxs
        else:
            assert part_idx != -1
            part_len = len(idxs) // part_num + 1
            idxs = idxs[part_len * part_idx: part_len * (part_idx + 1)]
        N_TASKS = len(idxs)
        def generator():
            for idx in idxs:
                session_id = idx
                yield cls(
                    task_id=idx,
                    session_id=session_id,
                )

        return generator(), N_TASKS
    