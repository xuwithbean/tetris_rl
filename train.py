import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
try:
    from tensorboardX import SummaryWriter
except Exception:
    SummaryWriter = None

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque

# 用于绘图与生成 GIF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from PIL import Image as PILImage
import signal
import threading


def get_args():
    parser = argparse.ArgumentParser("实现用于玩俄罗斯方块的深度 Q 学习训练脚本（已中文注释）")
    parser.add_argument("--width", type=int, default=10, help="棋盘宽度（格子数）")
    parser.add_argument("--height", type=int, default=20, help="棋盘高度（格子数）")
    parser.add_argument("--block_size", type=int, default=30, help="渲染用像素块大小")
    parser.add_argument("--batch_size", type=int, default=512, help="每次训练的样本批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--initial_epsilon", type=float, default=1, help="初始 epsilon（探索率）")
    parser.add_argument("--final_epsilon", type=float, default=1e-3, help="最终 epsilon")
    parser.add_argument("--num_decay_epochs", type=float, default=2000, help="epsilon 衰减周期（epoch）")
    parser.add_argument("--num_epochs", type=int, default=3000, help="训练总 epoch 数")
    parser.add_argument("--save_interval", type=int, default=1000, help="保存模型的间隔（epoch）")
    parser.add_argument("--replay_memory_size", type=int, default=30000, help="经验回放最大容量")
    parser.add_argument("--log_path", type=str, default="tensorboard", help="tensorboard 日志路径（如果可用）")
    parser.add_argument("--saved_path", type=str, default="trained_models", help="训练结果保存路径")
    parser.add_argument("--no-render", dest='render', action='store_false',
                        help="禁用每步渲染（在无头服务器上推荐使用）")
    parser.add_argument("--gif-interval", type=int, default=100,
                        help="每多少个 epoch 合成一次中间 GIF（默认为 100）")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="保存曲线 PNG 的目录（默认：<saved_path>/plots）")
    parser.add_argument("--gif-dir", type=str, default=None,
                        help="保存中间 GIF 的目录（默认：<saved_path>/gifs）")

    args = parser.parse_args()
    return args


def train(opt):
    # 随机种子设置（可复现性）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # 准备日志目录
    if SummaryWriter is not None:
        if os.path.isdir(opt.log_path):
            shutil.rmtree(opt.log_path)
        os.makedirs(opt.log_path, exist_ok=True)
        writer = SummaryWriter(opt.log_path)
    else:
        writer = None

    # 创建游戏环境与模型
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    # 初始状态
    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0

    # 回合计数（用于决定是否记录该回合的回放）
    episode_count = 0
    # 是否记录当前回合的帧（回放）以及帧缓存
    recording_current_episode = (opt.gif_interval > 0 and episode_count % opt.gif_interval == 0)
    frames = []

    # 确保保存目录存在
    os.makedirs(opt.saved_path, exist_ok=True)
    # 创建用于保存曲线图和 gifs 的目录（可由 CLI 指定）
    plot_dir = opt.plot_dir if opt.plot_dir is not None else os.path.join(opt.saved_path, 'plots')
    gif_dir = opt.gif_dir if opt.gif_dir is not None else os.path.join(opt.saved_path, 'gifs')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)

    # 用于记录训练过程中的 loss 与 reward（每个训练 epoch 的值）
    loss_history = []
    reward_history = []

    # 中断控制：当接收到 SIGINT/SIGTERM 时设置该事件，训练循环会安全退出并保存当前状态
    stop_event = threading.Event()

    def _signal_handler(signum, frame):
        print(f"收到信号 {signum}，将在当前安全点停止训练并保存模型/曲线...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    def save_checkpoint(tag="interrupt"):
        try:
            # 模型保存（带 epoch 信息）
            epoch_tag = epoch if 'epoch' in locals() else 0
            model_path = os.path.join(opt.saved_path, f'tetris_{tag}_{epoch_tag}')
            torch.save(model, model_path)
            print(f'已保存模型: {model_path}')

            # 绘制并保存当前 loss 与 reward 曲线
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
                if len(loss_history) > 0:
                    ax1.plot(range(1, len(loss_history) + 1), loss_history, color='tab:blue')
                ax1.set_title('Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('MSE Loss')

                if len(reward_history) > 0:
                    ax2.plot(range(1, len(reward_history) + 1), reward_history, color='tab:orange')
                ax2.set_title('Reward (Final score per episode)')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Score')

                plt.tight_layout()
                plot_path = os.path.join(plot_dir, f'loss_reward_{tag}_{epoch_tag}.png')
                fig.savefig(plot_path, bbox_inches='tight')
                plt.close(fig)
                print(f'已保存曲线图: {plot_path}')
            except Exception as e:
                print('保存中断时曲线图出错：', e)

            # 刷新并关闭 tensorboard writer（如果有）
            try:
                if writer is not None:
                    writer.close()
            except Exception:
                pass
        except Exception as e:
            print('保存检查点时出错：', e)

    try:
        while epoch < opt.num_epochs:
            # 如果接收到中断信号，安全保存并退出训练循环
            if stop_event.is_set():
                save_checkpoint(tag='signal')
                break

            # 列举当前所有候选动作及其对应状态特征
            next_steps = env.get_next_states()

            # 计算当前 epsilon（随 epoch 线性衰减到 final_epsilon）
            epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                    opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
            u = random()
            random_action = u <= epsilon

            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            if torch.cuda.is_available():
                next_states = next_states.cuda()

            # 模型预测每个候选位置的 Q 值
            model.eval()
            with torch.no_grad():
                predictions = model(next_states)[:, 0]
            model.train()

            # 探索或利用
            if random_action:
                index = randint(0, len(next_steps) - 1)
            else:
                index = torch.argmax(predictions).item()

            next_state = next_states[index, :]
            action = next_actions[index]

            # 执行动作（落子）；如果当前回合需要录制回放，则强制渲染并保存帧
            render_flag = opt.render or recording_current_episode
            reward, done = env.step(action, render=render_flag)

            # 如果 render 产生了最近一帧，保存到 frames（用于回放 GIF）
            if recording_current_episode and hasattr(env, '_last_img') and env._last_img is not None:
                try:
                    frames.append(env._last_img.copy())
                except Exception:
                    pass

            if torch.cuda.is_available():
                next_state = next_state.cuda()
            replay_memory.append([state, reward, next_state, done])

            if done:
                # 回合结束，记录并重置环境
                final_score = env.score
                final_tetrominoes = env.tetrominoes
                final_cleared_lines = env.cleared_lines
                state = env.reset()
                # 如果本回合是需要录制的回合，保存回放 GIF
                if recording_current_episode and frames:
                    try:
                        pil_frames = [PILImage.fromarray(im) for im in frames]
                        base_w, base_h = pil_frames[0].size
                        norm_frames = [f.resize((base_w, base_h), PILImage.BICUBIC) for f in pil_frames]
                        ep_gif_path = os.path.join(gif_dir, f'episode_{episode_count:06d}.gif')
                        imageio.mimsave(ep_gif_path, [np.array(f) for f in norm_frames], duration=0.05)
                        print(f'已保存回放 GIF: {ep_gif_path}')
                    except Exception as e:
                        print('保存回放 GIF 时出错：', e)
                # 增加回合计数并为下一回合决定是否记录
                episode_count += 1
                recording_current_episode = (opt.gif_interval > 0 and episode_count % opt.gif_interval == 0)
                frames = []
                if torch.cuda.is_available():
                    state = state.cuda()
            else:
                state = next_state
                continue

            # 经验池积累到一定量后开始训练
            if len(replay_memory) < opt.replay_memory_size / 10:
                continue

            epoch += 1
            batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
            state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            state_batch = torch.stack(tuple(state for state in state_batch))
            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
            next_state_batch = torch.stack(tuple(state for state in next_state_batch))

            if torch.cuda.is_available():
                state_batch = state_batch.cuda()
                reward_batch = reward_batch.cuda()
                next_state_batch = next_state_batch.cuda()

            # 计算当前 Q 值和目标 y
            q_values = model(state_batch)
            model.eval()
            with torch.no_grad():
                next_prediction_batch = model(next_state_batch)
            model.train()

            # y = reward (若回合结束) 或 reward + gamma * next_prediction
            y_batch = torch.cat(
                tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                      zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

            # 反向传播更新模型
            optimizer.zero_grad()
            loss = criterion(q_values, y_batch)
            loss.backward()
            optimizer.step()

            # 记录当前 epoch 的 loss 与回合得分（reward）
            # 注意：loss 是对一个 batch 的整体损失；final_score 是本回合结束时的得分
            loss_history.append(loss.item())
            reward_history.append(final_score)

            print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
                epoch,
                opt.num_epochs,
                action,
                final_score,
                final_tetrominoes,
                final_cleared_lines))

            # 写入 tensorboard（如果可用）
            if writer is not None:
                writer.add_scalar('Train/Score', final_score, epoch - 1)
                writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
                writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

            # 定期保存模型
            if epoch > 0 and epoch % opt.save_interval == 0:
                torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    except KeyboardInterrupt:
        print('捕获 KeyboardInterrupt，正在保存当前模型与曲线...')
        save_checkpoint(tag='keyboard')
    except Exception as e:
        print('训练循环发生未处理异常：', e)
        try:
            save_checkpoint(tag='error')
        except Exception:
            pass

    # 训练结束后保存最终模型
    torch.save(model, "{}/tetris".format(opt.saved_path))
    # 训练结束后，绘制并保存 loss 与 reward 曲线（仅一次）
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        ax1.plot(range(1, len(loss_history) + 1), loss_history, color='tab:blue')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')

        ax2.plot(range(1, len(reward_history) + 1), reward_history, color='tab:orange')
        ax2.set_title('Reward (Final score per episode)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')

        plt.tight_layout()
        final_plot_path = os.path.join(plot_dir, 'loss_reward.png')
        fig.savefig(final_plot_path, bbox_inches='tight')
        plt.close(fig)
        print(f'已保存最终 loss/reward 曲线: {final_plot_path}')
    except Exception as e:
        print('保存最终曲线图时出错：', e)


if __name__ == "__main__":
    opt = get_args()
    train(opt)
