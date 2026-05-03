"""Quick tests for SuperTicTacToe environment rules.

Run:
    python test_environment.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ENV_PATH = Path(__file__).with_name("Gymnasium environment.py")
spec = importlib.util.spec_from_file_location("gymnasium_environment", ENV_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)
SuperTicTacToe = module.SuperTicTacToe
ActionMaskWrapper = module.ActionMaskWrapper


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def make_env():
    return ActionMaskWrapper(SuperTicTacToe())


def place_line(env, cells, player=1):
    for b, r, c in cells:
        env.env.board[b, r, c] = player


def test_reset_and_mask():
    env = make_env()
    obs, info = env.reset(seed=123)
    assert_true(obs.shape == (6, 4, 4), "Observation shape should be (6, 4, 4)")
    assert_true(info["action_mask"].shape == (96,), "Action mask should have 96 entries")
    assert_true(int(np.sum(info["action_mask"])) == 96, "All actions should be legal on empty board")


def test_basic_move_and_turn_switch():
    env = make_env()
    obs, info = env.reset(seed=123)
    next_obs, reward, terminated, truncated, info = env.step(0)

    assert_true(next_obs.shape == (6, 4, 4), "Next observation shape mismatch")
    assert_true(reward in (0.0, 1.0), "Reward should be 0 or 1 after one move")
    assert_true(not truncated, "Truncated should be False in this environment")
    assert_true(info["action_mask"].shape == (96,), "Action mask should be returned after step")


def test_occupied_action_forfeits():
    env = make_env()
    env.reset(seed=123)

    # Force a deterministic placement on action 0 by monkeypatching the RNG branch.
    env.env.np_random = np.random.default_rng(0)
    env.env.board[0, 0, 0] = 1
    obs, reward, terminated, truncated, info = env.step(0)

    assert_true(reward == 0.0, "Occupied move should not reward")
    assert_true(not terminated, "Occupied move alone should not terminate")
    assert_true(info["action_mask"][0] == 0, "Occupied action should be masked out")


def test_legal_actions_shrink():
    env = make_env()
    env.reset(seed=123)
    initial = int(np.sum(env.env.action_mask()))
    env.env.board[0, 0, 0] = 1
    after = int(np.sum(env.env.action_mask()))
    assert_true(initial == 96, "Initially all 96 actions should be legal")
    assert_true(after == 95, "One occupied cell should reduce legal actions to 95")


def test_horizontal_win_detected():
    env = make_env()
    env.reset(seed=123)
    place_line(env, [(0, 0, 0), (0, 0, 1), (0, 0, 2)])

    env.env.np_random = np.random.default_rng(0)
    obs, reward, terminated, truncated, info = env.step(3)

    assert_true(terminated, "Horizontal 4-in-a-row should terminate the game")
    assert_true(reward == 1.0, "Horizontal 4-in-a-row should reward a win")


def test_vertical_same_level_does_not_win():
    env = make_env()
    env.reset(seed=123)
    # Fill a vertical line that stays within a single level band.
    # This should NOT be counted as a win.
    place_line(env, [(0, 0, 0), (0, 1, 0), (0, 2, 0)])

    env.env.np_random = np.random.default_rng(0)
    obs, reward, terminated, truncated, info = env.step(12)

    assert_true(not terminated, "Same-level vertical line should not count as a win")
    assert_true(reward == 0.0, "Same-level vertical line should not reward a win")


def test_vertical_cross_level_can_win():
    env = make_env()
    env.reset(seed=123)
    # Fill a vertical line that spans multiple levels.
    # This should be counted as a valid win.
    place_line(env, [(0, 0, 0), (1, 1, 0), (3, 2, 0)])

    env.env.np_random = np.random.default_rng(0)
    obs, reward, terminated, truncated, info = env.step(80)

    assert_true(terminated, "Cross-level vertical line should terminate the game")
    assert_true(reward == 1.0, "Cross-level vertical line should reward a win")


def test_game_over_blocks_additional_moves():
    env = make_env()
    env.reset(seed=123)
    place_line(env, [(0, 0, 0), (0, 0, 1), (0, 0, 2)])
    env.env.np_random = np.random.default_rng(0)
    env.step(3)

    try:
        env.step(4)
    except RuntimeError:
        return
    raise AssertionError("Stepping after termination should raise RuntimeError")


def run_all_tests():
    test_reset_and_mask()
    test_basic_move_and_turn_switch()
    test_occupied_action_forfeits()
    test_legal_actions_shrink()
    test_horizontal_win_detected()
    test_vertical_same_level_does_not_win()
    test_vertical_cross_level_can_win()
    test_game_over_blocks_additional_moves()
    print("All environment tests passed.")


if __name__ == "__main__":
    run_all_tests()
