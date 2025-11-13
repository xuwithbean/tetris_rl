import argparse
import torch
import cv2
from src.tetris import Tetris
from src.deep_q_network import DeepQNetwork
import os
try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False


class MultiWriter:
    """包装器：优先使用 OpenCV VideoWriter，写入失败则回退到 imageio writer（若可用）。"""
    def __init__(self, primary=None, fallback=None):
        self.primary = primary
        self.fallback = fallback
        self._used_fallback = False

    def isOpened(self):
        if self.primary is None:
            return self.fallback is not None
        try:
            return self.primary.isOpened()
        except Exception:
            return False

    def write(self, img):
        # 尝试使用 primary 写入；若失败，使用 fallback.append_data
        if self.primary is not None:
            try:
                self.primary.write(img)
                return
            except Exception:
                # 记录并尝试修复后重试：有些 OpenCV 构建要求 UMat 或 BGR 格式
                self._used_fallback = True
                try:
                    # 尝试转换为 UMat
                    umat = cv2.UMat(img)
                    self.primary.write(umat)
                    return
                except Exception:
                    pass
                try:
                    # 尝试转为 BGR（当前 img 从 PIL 来是 RGB）
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    self.primary.write(bgr)
                    return
                except Exception:
                    pass
        if self.fallback is not None:
            try:
                # imageio expects RGB arrays
                self.fallback.append_data(img)
            except Exception as e:
                raise
        else:
            raise RuntimeError("没有可用的视频写入器")

    def release(self):
        if self.primary is not None:
            try:
                self.primary.release()
            except Exception:
                pass
        if self.fallback is not None:
            try:
                self.fallback.close()
            except Exception:
                try:
                    # older imageio versions may use .close()
                    self.fallback.close()
                except Exception:
                    pass


def get_args():
    parser = argparse.ArgumentParser("Play Tetris with a pretrained DQN model")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--model-path", type=str, default="/home/xu/best_model.pth",
                        help="要加载的模型文件路径（torch.save 保存的模型）")
    parser.add_argument("--output", type=str, default=None, help="可选的视频输出文件（mp4/avi）")
    return parser.parse_args()


def load_model(path):
    # 尝试加载模型：如果文件本身是保存的模型对象就直接 load，否则尝试构建网络再 load state_dict
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件未找到: {path}")

    # 先尝试直接用默认方式加载（weights_only=True 的安全加载）。
    # 对于 PyTorch 2.6+，如果 checkpoint 是用 pickle 保存的模型对象，默认会拒绝加载。
    # 我们按顺序尝试：
    # 1) 直接 torch.load（安全默认）
    # 2) 在 safe globals 中 allowlist 本地的 DeepQNetwork 并 load（更安全）
    # 3) 若仍失败，作为最后手段使用 weights_only=False 重新加载（仅在信任文件时）
    # 4) 如果得到的是 state_dict，则构建网络并 load_state_dict

    def _try_torch_load(load_kwargs=None):
        load_kwargs = load_kwargs or {}
        try:
            return torch.load(path, **load_kwargs)
        except Exception as e:
            return e

    # 1) 直接尝试（通常是 weights_only=True）
    res = _try_torch_load({'map_location': (lambda storage, loc: storage)})
    if not isinstance(res, Exception):
        model = res
        print("已直接加载模型对象（torch.load 返回）")
        model.eval()
        return model

    # 2) 尝试使用 safe_globals / add_safe_globals allowlist 本地类再加载（若可用）
    ser = getattr(torch, 'serialization', None)
    if ser is not None:
        ctx = None
        if hasattr(ser, 'safe_globals'):
            try:
                ctx = ser.safe_globals([DeepQNetwork])
            except Exception:
                ctx = None
        elif hasattr(ser, 'add_safe_globals'):
            try:
                ctx = ser.add_safe_globals([DeepQNetwork])
            except Exception:
                ctx = None

        if ctx is not None:
            try:
                with ctx:
                    res2 = _try_torch_load({'map_location': (lambda storage, loc: storage)})
                if not isinstance(res2, Exception):
                    model = res2
                    print("已在 allowlist 下加载模型对象（torch.load 返回）")
                    model.eval()
                    return model
            except Exception:
                pass

    # 3) 作为最后手段，尝试 weights_only=False 来允许完整反序列化（仅当你信任 checkpoint 时）
    try:
        res3 = _try_torch_load({'map_location': (lambda storage, loc: storage), 'weights_only': False})
    except TypeError:
        # 运行的 torch 版本可能不支持 weights_only 参数（向后兼容），直接再次尝试不带该参数
        res3 = _try_torch_load({'map_location': (lambda storage, loc: storage)})

    if isinstance(res3, Exception):
        # 最终尝试把文件作为 state_dict 读取并加载到新建网络
        # 这里再做一次尝试读取可能的 state_dict（不信任反序列化）
        try:
            state = torch.load(path, map_location=lambda storage, loc: storage)
        except Exception as e:
            raise RuntimeError(f"无法加载模型文件: {path}\n原始错误: {res}\n后续尝试错误: {e}")
        model = DeepQNetwork()
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)
        print("已从 state_dict 加载模型参数（回退方案）")
        model.eval()
        return model

    # 如果 res3 是 object（完整模型）或 state_dict
    if isinstance(res3, dict) or hasattr(res3, 'keys'):
        # 当 res3 是 state dict
        model = DeepQNetwork()
        if isinstance(res3, dict) and 'state_dict' in res3:
            model.load_state_dict(res3['state_dict'])
        else:
            model.load_state_dict(res3)
        print("已从 state_dict 加载模型参数（weights_only=False）")
    else:
        # res3 是反序列化得到的模型对象
        model = res3
        print("已直接加载模型对象（weights_only=False 反序列化）")

    model.eval()
    return model


def main():
    opt = get_args()
    # 加载模型
    model = load_model(opt.model_path)
    if torch.cuda.is_available():
        model.cuda()

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()

    # 可选的视频输出
    out = None
    if opt.output:
        out_path = opt.output
        ext = os.path.splitext(out_path)[1].lower()
        # 针对不同扩展名选择合适的 fourcc（注意有些 OpenCV 构建可能不支持 mp4）
        if ext in ['.mp4', '.m4v', '.mov']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        primary = cv2.VideoWriter(out_path, fourcc, opt.fps,
                       (int(opt.width * opt.block_size), int(opt.height * opt.block_size)))
        fallback = None
        if not primary.isOpened():
            print(f"Warning: VideoWriter 无法打开输出文件 {out_path}（检查 codec/后端支持）。尝试使用 imageio 回退。")
            try:
                primary.release()
            except Exception:
                pass
            primary = None
            if _HAS_IMAGEIO:
                try:
                    fallback = imageio.get_writer(out_path, fps=opt.fps)
                    print(f"使用 imageio 回退写入视频: {out_path}")
                except Exception as e:
                    print(f"Warning: 使用 imageio 写入失败: {e}，将不输出视频。")
                    fallback = None
        else:
            # primary 成功打开，同时根据可用性创建 fallback（可选）以便在写入失败时回退
            if _HAS_IMAGEIO:
                try:
                    fallback = imageio.get_writer(out_path, fps=opt.fps)
                except Exception:
                    fallback = None

        if primary is None and fallback is None:
            out = None
        else:
            out = MultiWriter(primary=primary, fallback=fallback)

    # 每一步列举所有可能动作并用模型评估，然后选择 Q 值最大的动作
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        # 前向推理得到每个候选位置的 Q 值
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)
        if done:
            if out:
                out.release()
            break


if __name__ == "__main__":
    main()
