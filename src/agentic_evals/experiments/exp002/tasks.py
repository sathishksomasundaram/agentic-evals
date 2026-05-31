"""The coding battery for exp-002 — small, *executable* tasks.

Each task is a prompt plus a ``test`` snippet of `assert` statements. A run is
PASS only if the code extracted from the model's reply defines what the test
expects and the asserts all hold when executed in a fresh subprocess. This keeps
the "good enough for day-to-day coding" claim honest: correctness is run, not
eyeballed.

Tasks are tagged ``difficulty`` ("easy" / "hard") so the report can separate the
toy-function pass-rate from the harder, edge-case-heavy and bug-fix tasks that
better approximate the "replaces Cursor/ChatGPT" bar.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class CodingTask:
    name: str
    difficulty: str
    prompt: str
    test: str


_BUGGY_BINARY_SEARCH = """\
The function below should return the index of `target` in the sorted list \
`nums`, or -1 if it is absent, using binary search. It has a bug that makes it \
fail (it can loop forever or return the wrong index). Return a corrected \
version with the same signature. Respond with a single Python code block and \
no prose.

```python
def binary_search(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid
        else:
            hi = mid
    return -1
```"""


TASKS: tuple[CodingTask, ...] = (
    CodingTask(
        name="is_palindrome",
        difficulty="easy",
        prompt=(
            "Write a Python function `is_palindrome(s: str) -> bool` that returns "
            "True if the string is a palindrome, ignoring case, spaces and "
            "punctuation (consider only alphanumeric characters). Respond with a "
            "single Python code block and no prose."
        ),
        test=(
            "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
            "assert is_palindrome('race a car') is False\n"
            "assert is_palindrome('') is True\n"
            "assert is_palindrome('No lemon, no melon') is True\n"
        ),
    ),
    CodingTask(
        name="two_sum",
        difficulty="easy",
        prompt=(
            "Write a Python function `two_sum(nums: list[int], target: int) -> "
            "list[int]` that returns the indices of the two numbers that add up to "
            "target. Exactly one solution exists; you may not use the same element "
            "twice. Respond with a single Python code block and no prose."
        ),
        test=(
            "assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]\n"
            "assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]\n"
            "assert sorted(two_sum([3, 3], 6)) == [0, 1]\n"
        ),
    ),
    CodingTask(
        name="lru_cache_class",
        difficulty="easy",
        prompt=(
            "Write a Python class `LRUCache` with `__init__(self, capacity: int)`, "
            "`get(self, key: int) -> int` (returns -1 if absent), and "
            "`put(self, key: int, value: int) -> None`, evicting the least-recently-"
            "used item when over capacity. Respond with a single Python code block "
            "and no prose."
        ),
        test=(
            "c = LRUCache(2)\n"
            "c.put(1, 1); c.put(2, 2)\n"
            "assert c.get(1) == 1\n"
            "c.put(3, 3)  # evicts key 2\n"
            "assert c.get(2) == -1\n"
            "c.put(4, 4)  # evicts key 1\n"
            "assert c.get(1) == -1\n"
            "assert c.get(3) == 3\n"
            "assert c.get(4) == 4\n"
        ),
    ),
    CodingTask(
        name="merge_intervals",
        difficulty="hard",
        prompt=(
            "Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
            "that merges all overlapping intervals and returns the result sorted by "
            "start. Intervals that merely touch (e.g. [1,4] and [4,5]) count as "
            "overlapping. Respond with a single Python code block and no prose."
        ),
        test=(
            "def _n(r):\n"
            "    return [list(x) for x in r]\n"
            "assert _n(merge_intervals([[1,3],[2,6],[8,10],[15,18]])) == "
            "[[1,6],[8,10],[15,18]]\n"
            "assert _n(merge_intervals([[1,4],[4,5]])) == [[1,5]]\n"
            "assert _n(merge_intervals([])) == []\n"
            "assert _n(merge_intervals([[1,4],[0,2],[3,5]])) == [[0,5]]\n"
            "assert _n(merge_intervals([[1,4],[2,3]])) == [[1,4]]\n"
        ),
    ),
    CodingTask(
        name="coin_change",
        difficulty="hard",
        prompt=(
            "Write `coin_change(coins: list[int], amount: int) -> int` returning the "
            "fewest number of coins that make up `amount`, or -1 if it cannot be "
            "made. You may use each coin denomination any number of times. Respond "
            "with a single Python code block and no prose."
        ),
        test=(
            "assert coin_change([1, 2, 5], 11) == 3\n"
            "assert coin_change([2], 3) == -1\n"
            "assert coin_change([1], 0) == 0\n"
            "assert coin_change([1, 2, 5], 100) == 20\n"
            "assert coin_change([186, 419, 83, 408], 6249) == 20\n"
        ),
    ),
    CodingTask(
        name="min_stack",
        difficulty="hard",
        prompt=(
            "Write a class `MinStack` supporting `push(self, val: int) -> None`, "
            "`pop(self) -> None`, `top(self) -> int`, and `get_min(self) -> int` "
            "that returns the minimum element currently in the stack. All four "
            "operations must run in O(1) time. Respond with a single Python code "
            "block and no prose."
        ),
        test=(
            "s = MinStack()\n"
            "s.push(-2); s.push(0); s.push(-3)\n"
            "assert s.get_min() == -3\n"
            "s.pop()\n"
            "assert s.top() == 0\n"
            "assert s.get_min() == -2\n"
            "s.push(-5)\n"
            "assert s.get_min() == -5\n"
        ),
    ),
    CodingTask(
        name="group_anagrams",
        difficulty="hard",
        prompt=(
            "Write `group_anagrams(words: list[str]) -> list[list[str]]` that groups "
            "words which are anagrams of each other. The order of the groups and of "
            "the words within each group does not matter. Respond with a single "
            "Python code block and no prose."
        ),
        test=(
            "def _n(groups):\n"
            "    return sorted(sorted(g) for g in groups)\n"
            "assert _n(group_anagrams(['eat','tea','tan','ate','nat','bat'])) == "
            "_n([['bat'],['nat','tan'],['ate','eat','tea']])\n"
            "assert group_anagrams([]) == []\n"
            "assert _n(group_anagrams([''])) == [['']]\n"
        ),
    ),
    CodingTask(
        name="fix_binary_search",
        difficulty="hard",
        prompt=_BUGGY_BINARY_SEARCH,
        test=(
            "assert binary_search([1,2,3,4,5], 4) == 3\n"
            "assert binary_search([1,2,3,4,5], 1) == 0\n"
            "assert binary_search([1,2,3,4,5], 5) == 4\n"
            "assert binary_search([1,2,3,4,5], 6) == -1\n"
            "assert binary_search([], 1) == -1\n"
            "assert binary_search([5], 5) == 0\n"
        ),
    ),
)


def extract_code(reply: str) -> str:
    """Pull the first fenced Python block; fall back to the whole reply."""
    match = _CODE_FENCE.search(reply)
    return match.group(1) if match else reply


def run_task(task: CodingTask, reply: str, *, timeout_s: int = 15) -> tuple[bool, str]:
    """Execute the extracted code + the task's asserts in a fresh subprocess.

    Returns (passed, detail). `detail` is "ok" on success, else a short reason.
    """
    code = extract_code(reply)
    script = f"{code}\n\n# --- test ---\n{task.test}\nprint('PASS')\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(script)
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout>{timeout_s}s"
    if proc.returncode == 0 and proc.stdout.strip().endswith("PASS"):
        return True, "ok"
    err = (proc.stderr or proc.stdout).strip().splitlines()
    return False, err[-1] if err else f"exit={proc.returncode}"
