import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import random

style.use("ggplot")


class Tetris:
    # 方块颜色（RGB）
    piece_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)
    ]

    # 七种俄罗斯方块形状（用整数表示不同块）
    pieces = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]]
    ]

    def __init__(self, height=20, width=10, block_size=20):
        # 棋盘高度、宽度（以格子计）和像素块大小
        self.height = height
        self.width = width
        self.block_size = block_size
        # 文字颜色（绘制在棋盘左上角）
        self.text_color = (200, 20, 220)
        self.reset()

    def reset(self):
        # 初始化棋盘和状态
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        # 使用 7-bag 随机序列
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False
        return self.get_state_properties(self.board)

    def rotate(self, piece):
        # 顺时针旋转方块矩阵
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    def get_state_properties(self, board):
        # 返回用于 RL 的简单特征向量：消行数、空洞数、凹凸度、总体高度
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def get_holes(self, board):
        # 计算空洞（在列中，第一个非零块下方的零格数）
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
        # 计算列高度与相邻列差的总和（凹凸度）
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_next_states(self):
        # 返回当前方块所有可能放置（x, rotation）对应的状态特征
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        # 不同方块的独立旋转次数
        if piece_id == 0:  # O 方块
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_state_properties(board)
            curr_piece = self.rotate(curr_piece)
        return states

    def get_current_board_state(self):
        # 返回包含当前落下方块的棋盘视图（不改变原棋盘）
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board

    def new_piece(self):
        # 生成新方块并检查是否游戏结束
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0
                            }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        # 将 piece 放到 pos 后，判断是否与棋盘或底部冲突（检查下移一步是否越界/重叠）
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def truncate(self, piece, pos):
        # 当方块上方有障碍时，截断方块上部（用于碰撞时处理越界覆盖）
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def store(self, piece, pos):
        # 将方块写入棋盘并返回新棋盘（不修改原棋盘）
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        # 检查并删除已满行，返回删除行数和新的棋盘
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        # 删除指定行并在顶部补充空行
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def step(self, action, render=True, video=None):
        # 执行一次动作：action=(x, num_rotations)
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            if render:
                self.render(video)

        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True

        self.board = self.store(self.piece, self.current_pos)

        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        if not self.gameover:
            self.new_piece()
        if self.gameover:
            self.score -= 2

        return score, self.gameover

    def render(self, video=None):
        # 使用 PIL 构建主图并在 PIL 层绘制文字与网格，然后转换为 numpy 供 OpenCV 显示
        if not self.gameover:
            px = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else:
            px = [self.piece_colors[p] for row in self.board for p in row]

        arr = np.array(px).reshape((self.height, self.width, 3)).astype(np.uint8)
        arr = arr[..., ::-1]
        main_pil = Image.fromarray(arr, 'RGB')
        main_pil = main_pil.resize((self.width * self.block_size, self.height * self.block_size), Image.NEAREST)

        # 不再使用右侧信息栏，直接以主图为最终画布
        full_pil = main_pil

        # 在 PIL 上绘制网格线与文字
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(full_pil)
            # 网格线
            for i in range(self.height):
                y = i * self.block_size
                draw.line([(0, y), (main_pil.width - 1, y)], fill=(0, 0, 0))
            for j in range(self.width):
                x = j * self.block_size
                draw.line([(x, 0), (x, main_pil.height - 1)], fill=(0, 0, 0))

            # 文字（绘制在棋盘左上角）
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            pad = int(self.block_size * 0.2)
            text_x = pad
            text_y = pad
            draw.text((text_x, text_y), f"Score: {self.score}", fill=self.text_color, font=font)
            draw.text((text_x, text_y + int(self.block_size * 1.0)), f"Pieces: {self.tetrominoes}", fill=self.text_color, font=font)
            draw.text((text_x, text_y + int(self.block_size * 2.0)), f"Lines: {self.cleared_lines}", fill=self.text_color, font=font)
        except Exception:
            pass

        # 转换为 numpy 以供 OpenCV 显示并写入视频
        img = np.array(full_pil)
        img = np.ascontiguousarray(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if video:
            try:
                # 确保传入的是 numpy 数组（有时可能为 PIL.Image），并保证是连续的 uint8
                if not isinstance(img, np.ndarray):
                    img = np.asarray(img)
                img = np.ascontiguousarray(img)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)

                # 支持两种 writer：OpenCV 的 VideoWriter（具有 isOpened/write），
                # 以及 imageio 的 writer（具有 append_data）
                if hasattr(video, 'isOpened'):
                    if not video.isOpened():
                        print(f"Warning: VideoWriter 未打开，跳过写入视频帧。")
                    else:
                        video.write(img)
                elif hasattr(video, 'append_data'):
                    # imageio 要求 RGB numpy 数组，img 当前为 RGB（来自 PIL）
                    video.append_data(img)
                else:
                    print("Warning: 未知的视频写入器类型，跳过写入。")
            except Exception as e:
                # 打印更详细的调试信息，便于定位 video.write 失败的具体原因
                import traceback
                try:
                    print("Warning: 写入视频帧失败:", repr(e))
                    print("视频写入器类型:", type(video))
                    print("写入器方法(hasattr): isOpened=", hasattr(video, 'isOpened'),
                          "write=", hasattr(video, 'write'), "append_data=", hasattr(video, 'append_data'))
                    try:
                        if isinstance(img, np.ndarray):
                            print("帧信息: shape=", img.shape, "dtype=", img.dtype, "C_CONTIGUOUS=", img.flags['C_CONTIGUOUS'])
                        else:
                            print("帧不是 numpy 数组；类型:", type(img))
                    except Exception:
                        pass
                except Exception:
                    pass
                traceback.print_exc()

        # 保存最后一帧（以便外部保存为 GIF 回放）
        try:
            self._last_img = img.copy()
        except Exception:
            self._last_img = None

        # 使用 OpenCV 展示
        try:
            cv2.imshow("Deep Q-Learning Tetris", img)
            cv2.waitKey(1)
        except Exception:
            # 无法显示则忽略（例如无头服务器）
            pass
        return img
