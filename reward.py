import re
import math
from typing import Optional, Callable


def accuracy_reward(
    completions: list[str],
    solution: list[str], 
    **kwargs
) -> list[Optional[float]]:
    def extract_points(text: str) -> Optional[list[tuple[int, int]]]:
        pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
        matches = re.findall(pattern, text)
        if not matches:
            return None
        return [(int(x), int(y)) for x, y in matches]

    def dist(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def path_length(path: list[tuple[int, int]]) -> float:
        return sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1))
 
    rewards = []

    for com, sol in zip(completions, solution):
        com_points = extract_points(com)
        sol_points = extract_points(sol)

        if com_points is None or len(com_points) < 2:
            rewards.append(0.0)
            continue

        # 起止点 reward
        rs = min(15 / (dist(sol_points[0], com_points[0]) + 1e-6), 1.0)
        rt = min(15 / (dist(sol_points[-1], com_points[-1]) + 1e-6), 1.0)
        r_key = (rs + rt) / 2

        # 路径长度 reward
        d1 = path_length(sol_points)
        d2 = path_length(com_points)
        r_dis = min(d1 / (d2 + 1e-6), 1.0)

        # 航点数量 reward
        n1 = len(sol_points)
        n2 = len(com_points)
        r_cnt = 1.0 if (0.7 * n2 <= n1 <= 1.5 * n2) else 0.0

        # 综合 reward
        w = [0.4, 0.4, 0.2]
        reward = (r_key * w[0] + r_dis * w[1] + r_cnt * w[2]) / sum(w)
        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    """format: [(x1, y1), (x2, y2), ..., (xn, yn)]"""
    def is_valid_format(text: str) -> bool:
        pattern = r"\(\s*\d+\s*,\s*\d+\s*\)"
        matches = re.findall(pattern, text)
        return len(matches) >= 2

    rewards = [1.0 if is_valid_format(text) else 0.0 for text in completions]
    return rewards

def get_reward_funcs() -> dict[Callable]:
    return [accuracy_reward, format_reward]