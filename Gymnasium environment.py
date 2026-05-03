import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SuperTicTacToe(gym.Env):
    """Super Tic-Tac-Toe environment with global-coordinate win detection.

    Board layout:
    - 6 independent 4x4 blocks
    - Each block maps to a specific set of global board coordinates
    - Win detection uses those global coordinates directly

    Rules implemented here:
    - Horizontal win: 4 consecutive cells in the same row
    - Vertical win: 4 consecutive cells in the same column
    - Diagonal win: 5 consecutive cells on a diagonal
    - All sequences are checked in the global coordinate system
    """

    def __init__(self):
        super().__init__()

        self.num_blocks = 6
        self.block_size = 4
        self.num_cells = self.num_blocks * self.block_size * self.block_size

        self.action_space = spaces.Discrete(self.num_cells)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_blocks, self.block_size, self.block_size),
            dtype=np.int8,
        )

        # Global-coordinate mapping for each block cell.
        # Coordinates are stored as (col, row).
        self.block_to_coords = {
            0: [
                (5, 12), (6, 12), (7, 12), (8, 12),
                (5, 11), (6, 11), (7, 11), (8, 11),
                (5, 10), (6, 10), (7, 10), (8, 10),
                (5, 9), (6, 9), (7, 9), (8, 9),
            ],
            1: [
                (6, 5), (6, 6), (6, 7), (6, 8),
                (5, 5), (5, 6), (5, 7), (5, 8),
                (4, 5), (4, 6), (4, 7), (4, 8),
                (3, 5), (3, 6), (3, 7), (3, 8),
            ],
            2: [
                (7, 8), (8, 8), (9, 8), (10, 8),
                (7, 7), (8, 7), (9, 7), (10, 7),
                (7, 6), (8, 6), (9, 6), (10, 6),
                (7, 5), (8, 5), (9, 5), (10, 5),
            ],
            3: [
                (1, 4), (2, 4), (3, 4), (4, 4),
                (1, 3), (2, 3), (3, 3), (4, 3),
                (1, 2), (2, 2), (3, 2), (4, 2),
                (1, 1), (2, 1), (3, 1), (4, 1),
            ],
            4: [
                (5, 4), (6, 4), (7, 4), (8, 4),
                (5, 3), (6, 3), (7, 3), (8, 3),
                (5, 2), (6, 2), (7, 2), (8, 2),
                (5, 1), (6, 1), (7, 1), (8, 1),
            ],
            5: [
                (9, 4), (10, 4), (11, 4), (12, 4),
                (9, 3), (10, 3), (11, 3), (12, 3),
                (9, 2), (10, 2), (11, 2), (12, 2),
                (9, 1), (10, 1), (11, 1), (12, 1),
            ],
        }

        # Row bands map to levels as follows:
        # rows 1-4 -> level 3
        # rows 5-8 -> level 2
        # rows 9-12 -> level 1
        self.block_to_level = {0: 1, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3}
        self.coord_to_block = {}
        self.coord_to_local = {}
        for block, coords in self.block_to_coords.items():
            for local_idx, coord in enumerate(coords):
                self.coord_to_block[coord] = block
                self.coord_to_local[coord] = local_idx

        self._win_lengths = {"horizontal": 4, "vertical": 4, "diagonal": 5}

        self.board = None
        self.current_player = None
        self._terminated = False
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.num_blocks, self.block_size, self.block_size), dtype=np.int8)
        self.current_player = 1
        self._terminated = False
        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_player == 1:
            return self.board.copy()
        return (-self.board).astype(np.int8)

    def legal_actions(self):
        empty = np.argwhere(self.board == 0)
        return [self._to_action_idx(b, r, c) for b, r, c in empty]

    def action_mask(self):
        mask = np.zeros(self.num_cells, dtype=np.int8)
        for idx in self.legal_actions():
            mask[idx] = 1
        return mask

    def _to_action_idx(self, b, r, c):
        return b * 16 + r * 4 + c

    def _from_action_idx(self, action):
        b = action // 16
        r = (action % 16) // 4
        c = action % 4
        return b, r, c

    def _neighbor_positions(self, b, r, c):
        del b
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.block_size and 0 <= nc < self.block_size:
                neighbors.append((r + dr, c + dc))
            else:
                neighbors.append(None)
        return neighbors

    def _sample_actual_position(self, action):
        b, r, c = self._from_action_idx(action)

        if self.np_random.random() < 0.5:
            return b, r, c

        neighbors = self._neighbor_positions(b, r, c)
        choice = neighbors[self.np_random.integers(0, 8)]
        if choice is None:
            return None
        nr, nc = choice
        return b, nr, nc

    def _occupied_global_coords(self, player):
        coords = set()
        for b in range(self.num_blocks):
            for r in range(self.block_size):
                for c in range(self.block_size):
                    if self.board[b, r, c] == player:
                        coords.add(self.block_to_coords[b][r * 4 + c])
        return coords

    def _check_win(self, player):
        coords = self._occupied_global_coords(player)
        if not coords:
            return False

        directions = {
            "horizontal": (1, 0),
            "vertical": (0, 1),
            "diag_down": (1, 1),
            "diag_up": (1, -1),
        }

        for x, y in coords:
            for name, (dx, dy) in directions.items():
                length = self._win_lengths["diagonal"] if "diag" in name else self._win_lengths["horizontal"]
                prev = (x - dx, y - dy)
                if prev in coords:
                    continue

                sequence = [(x + i * dx, y + i * dy) for i in range(length)]
                if not all(p in coords for p in sequence):
                    continue

                if name == "vertical":
                    # A vertical win is only valid if its 4 cells span multiple levels.
                    # Rows 1-4 are level 3, rows 5-8 are level 2, rows 9-12 are level 1.
                    levels = {3 if row <= 4 else 2 if row <= 8 else 1 for _, row in sequence}
                    if len(levels) == 1:
                        continue

                return True
        return False

    def step(self, action):
        if self._terminated:
            raise RuntimeError("Episode has terminated. Call reset() before stepping again.")

        reward = 0.0
        terminated = False
        truncated = False

        if action < 0 or action >= self.num_cells or self.board[self._from_action_idx(action)] != 0:
            actual_pos = None
        else:
            actual_pos = self._sample_actual_position(action)

        if actual_pos is not None:
            b, r, c = actual_pos
            if self.board[b, r, c] == 0:
                self.board[b, r, c] = self.current_player

                if self._check_win(self.current_player):
                    reward = 1.0
                    terminated = True
                elif not np.any(self.board == 0):
                    reward = 0.0
                    terminated = True

        if not terminated:
            self.current_player = 3 - self.current_player

        self._terminated = terminated
        return self._get_obs(), reward, terminated, truncated, {"action_mask": self.action_mask()}

    def render(self):
        symbols = {0: ".", 1: "O", 2: "X"}
        output = []
        for b in range(self.num_blocks):
            level = self.block_to_level[b]
            output.append(f"Block {b} (Level {level})")
            for r in range(self.block_size):
                output.append(" ".join(symbols[int(v)] for v in self.board[b, r]))
            output.append("")
        print("\n".join(output))


class ActionMaskWrapper(gym.Wrapper):
    """Optional wrapper that exposes action masks in observations/info for training."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["action_mask"] = self.env.action_mask()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["action_mask"] = self.env.action_mask()
        return obs, reward, terminated, truncated, info
