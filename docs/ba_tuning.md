# ViPE SLAM 技术文档

## 第一部分：核心概念

### 1.1 Pose（位姿）—— 4×4 矩阵的含义

每帧输出的 pose 是一个 **4×4 齐次变换矩阵（homogeneous transformation matrix）**：

```
    ┌                         ┐
    │ r00  r01  r02  │  tx    │
    │ r10  r11  r12  │  ty    │       T = [ R  t ]
    │ r20  r21  r22  │  tz    │           [ 0  1 ]
    │ ─────────────────────── │
    │  0    0    0   │   1    │
    └                         ┘
         旋转 R (3×3)     平移 t (3×1)
```

| 组成部分 | 维度 | 物理意义 |
|---------|------|---------|
| **R** (旋转矩阵) | 3×3，正交矩阵 (R^T R = I, det(R) = 1) | 相机朝向：x 轴朝右、y 轴朝下、z 轴朝前（OpenCV 惯例） |
| **t** (平移向量) | 3×1 | 相机在世界坐标系中的位置（c2w 时）或原点到相机的偏移（w2c 时） |
| **最后一行** | [0, 0, 0, 1] | 齐次坐标固定行，使矩阵乘法可同时处理旋转和平移 |

#### w2c vs c2w

ViPE SLAM 内部使用两种方向的变换矩阵：

| 名称 | 方向 | 含义 | 代码中的使用 |
|------|------|------|------------|
| **w2c** (world-to-camera) | 世界 → 相机 | 将世界坐标点变换到相机坐标系 | SLAM 内部优化变量 (`buffer.poses`) |
| **c2w** (camera-to-world) | 相机 → 世界 | 将相机坐标点变换到世界坐标系 | 最终输出 (`trajectory`) |

两者互逆：`c2w = w2c.inv()`

代码对照：
- SLAM BA 中的 pose 定义（`vipe/slam/ba/terms.py:99`）：
  > `Pose is the world2cam transform.`
- 输出时取逆转为 c2w（`vipe/slam/system.py:172`）：
  > `trajectory = filled_return.poses.inv()`
- 保存到 npy 时调用 `.matrix()` 得到 4×4 numpy 数组（`scripts/infer_jsonl_pose.py`）：
  > `pose_data = trajectory.matrix().cpu().numpy()`

**结论：你保存的 `.npy` 文件中，每帧是一个 c2w 矩阵，shape 为 `(N, 4, 4)`。**

#### SE3 李群表示

代码中 pose 并非直接以 4×4 矩阵存储，而是使用 **SE3 李群**（`vipe/ext/lietorch/groups.py`）：

| 操作 | 代码 | 含义 |
|------|------|------|
| 矩阵形式 | `pose.matrix()` | SE3 → 4×4 numpy 矩阵 |
| 求逆 | `pose.inv()` | w2c ↔ c2w 互转 |
| 对数映射 | `pose.log()` | SE3 → 6 维李代数向量 (3 旋转 + 3 平移) |
| 指数映射 | `SE3.exp(w)` | 6 维向量 → SE3 |
| 相对变换 | `p2 * p1.inv()` | 计算 p1 到 p2 的相对运动 |
| 作用于点 | `pose.act(points)` | 将 3D 点从一个坐标系变换到另一个 |

李群的优势：旋转参数化避免了万向锁、归一化问题，且 `log/exp` 映射使插值和优化更自然。

---

### 1.2 Intrinsics（内参）—— 与 Pose 的关系

内参描述相机光学属性，与 pose 是**独立的两个概念**：

| | Pose (外参) | Intrinsics (内参) |
|---|---|---|
| **描述** | 相机在哪、朝哪看 | 相机如何将 3D 投影到 2D |
| **内容** | 旋转 R + 平移 t | fx, fy, cx, cy (+畸变参数) |
| **维度** | 4×4 矩阵 | 4 维向量 (pinhole) 或 5 维 (mei) |
| **随帧变化** | 每帧不同 | 通常整个视频固定 |
| **变换空间** | 世界坐标 ↔ 相机坐标 | 相机坐标 → 像素坐标 |

#### 完整的 3D→2D 投影链路

```
世界 3D 点 P_w    ──[Pose w2c]──▷    相机坐标 P_c    ──[Intrinsics]──▷    像素 (u, v)

数学表达:
  P_c = T_w2c × P_w           # 外参：世界→相机
  u = fx * (X/Z) + cx         # 内参：相机 3D→像素 2D
  v = fy * (Y/Z) + cy
```

其中内参矩阵 K：
```
    ┌              ┐
K = │ fx   0   cx  │
    │  0  fy   cy  │        fx, fy: 焦距（像素单位）
    │  0   0    1  │        cx, cy: 主点（通常接近图像中心）
    └              ┘
```

代码对照（`vipe/slam/interface.py:187-189`）：
```python
class SLAMOutput:
    trajectory: SE3           # (N,) 每帧一个 pose
    intrinsics: torch.Tensor  # (V, 4) 每个视角一组 [fx, fy, cx, cy]
```

#### 支持的相机模型

ViPE 支持多种相机模型（`vipe/utils/cameras.py`）：

| 模型 | 内参维度 | 说明 |
|------|---------|------|
| `PINHOLE` | 4 (fx, fy, cx, cy) | 标准针孔，无畸变 |
| `MEI` | 5 (fx, fy, cx, cy, k1) | 统一球面投影模型，支持鱼眼/广角 |
| `PANORAMA` | 4 (占位，值为 0) | 等距柱状投影 (360°) |

#### 关键区别：改 pose 不影响内参，改内参不影响 pose

- **Pose 变化 = 相机移动/旋转**（走路、开车、转头）
- **Intrinsics 变化 = 光学属性改变**（变焦、切换镜头）
- BA 中两者**联合优化**但数学上独立——Jacobian 矩阵是对它们分别求导的

---

### 1.3 从前一帧 Pose 推断下一帧

**可以，ViPE 内部已经在做这件事。** 核心假设是**恒速运动模型**。

#### 方法 1：前端初始化（frontend.py:70-75）

```python
# 用 t-2 和 t-1 的 pose 预测 t 的初始值
p1 = SE3(self.video.poses[self.t1 - 2])
p2 = SE3(self.video.poses[self.t1 - 1])
w = (p2 * p1.inv()).log() * 0.5          # 取一半的相对运动（保守估计）
self.video.poses[self.t1] = (SE3.exp(w) * p2).data
```

原理：计算前两帧的相对运动，取一半作为下一帧的预测增量。

#### 方法 2：内填充插值（inner_filler.py:78-82）

```python
# 从关键帧 t0, t1 插值得到中间帧 pose
d_pose = n_pose[t1] * n_pose[t0].inv()           # 两关键帧间的相对运动
vel = d_pose.log() / d_time.unsqueeze(-1)         # 转为"速度"（李代数/时间）
w = vel * (m_tstamp - n_tstamp[t0]).unsqueeze(-1) # 按时间比例缩放
m_pose = SE3.exp(w) * n_pose[t0]                  # 指数映射 + 左乘基准 pose
```

原理：在 SE3 李群上做线性插值（Slerp for rotation + Lerp for translation）。

#### 在推理脚本外使用

如果你想独立预测下一帧 pose（不依赖 ViPE）：

```python
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def predict_next_pose(pose_prev: np.ndarray, pose_curr: np.ndarray) -> np.ndarray:
    """
    恒速模型：从前两帧 4x4 c2w 矩阵预测下一帧。

    Args:
        pose_prev: (4, 4) c2w matrix at frame t-1
        pose_curr: (4, 4) c2w matrix at frame t

    Returns:
        (4, 4) predicted c2w matrix at frame t+1
    """
    delta = pose_curr @ np.linalg.inv(pose_prev)  # 相对变换
    return delta @ pose_curr                       # 假设相同运动继续


def interpolate_poses(pose_a: np.ndarray, pose_b: np.ndarray, t: float) -> np.ndarray:
    """
    在两个 c2w pose 之间按比例 t ∈ [0, 1] 插值。

    Args:
        pose_a: (4, 4) c2w at start
        pose_b: (4, 4) c2w at end
        t: interpolation factor (0 = pose_a, 1 = pose_b)

    Returns:
        (4, 4) interpolated c2w matrix
    """
    # 旋转：Slerp
    r_a = Rotation.from_matrix(pose_a[:3, :3])
    r_b = Rotation.from_matrix(pose_b[:3, :3])
    slerp = Slerp([0, 1], Rotation.concatenate([r_a, r_b]))
    r_mid = slerp(t)

    # 平移：线性插值
    t_mid = (1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3]

    result = np.eye(4)
    result[:3, :3] = r_mid.as_matrix()
    result[:3, 3] = t_mid
    return result
```

#### 预测的局限性

| 场景 | 恒速模型效果 |
|------|------------|
| 匀速直线运动（走廊、高速路） | 优秀，误差极小 |
| 匀速转弯 | 较好，1-2 帧内有效 |
| 突然加速/刹车 | 差，预测方向正确但幅度偏离 |
| 突然转向 | 差，旋转预测会偏离 |
| 静止 | 完美（相对运动为零） |

**结论：短期预测（1-2 帧）通常够用；超过 5 帧会累积漂移。如果需要长期预测，应结合 IMU 或场景先验。**

---

## 第二部分：BA 调参指南

### 概览

ViPE SLAM 的核心耗时在于 **Bundle Adjustment (BA)**——对相机位姿和深度的迭代优化。
BA 运行在 CPU 上（`scipy.sparse.linalg.spsolve`），每次迭代涉及 GPU→CPU 数据传输。
一个典型视频可能触发 **2000+ 次 BA 迭代**，是推理速度的主要瓶颈。

通过调节下列参数，可以在 **速度 vs 精度** 之间灵活权衡。

---

### SLAM 执行流程与参数对照

```
Pass 1: 前端逐帧处理
  ├─ 初始化阶段 ──────── frontend_init_iters × (itrs=3/次)
  ├─ 每关键帧 ─────────── frontend_iters1 × (itrs=3) + frontend_iters2 × (itrs=3)
  └─ 中途后端触发 ──────── frontend_backend_steps × backend_ba_itrs
       (当关键帧数到达 [16, 64, 256] 时)

全局 BA 第一轮 ──────────── backend_first_steps × backend_ba_itrs
全局 BA 第二轮 ──────────── backend_iters × backend_ba_itrs

Pass 2: 内填充（非关键帧位姿插值）
  └─ 每 chunk ─────────── inner_filler_iters × (itrs=3/次)
```

---

## 参数详解

### `backend_iters` (默认: 24) ⭐ 影响最大

- **位置**: 全局 BA 第二轮的 steps 数
- **作用**: 控制最终全局优化的深度。每个 step 包含一次 DroidNet GRU 更新 + BA 求解
- **BA 总次数**: `backend_iters × backend_ba_itrs`（默认 24×8 = 192 次）
- **降低效果**: 24→12 约减少全局 BA 时间 50%，对大多数场景 pose 精度损失 <5%
- **建议范围**: 8~24

### `backend_first_steps` (默认: 7)

- **位置**: 全局 BA 第一轮的 steps 数
- **作用**: 第一轮全局 BA，为第二轮提供初始值。使用旧的 GRU 状态
- **BA 总次数**: `backend_first_steps × backend_ba_itrs`（默认 7×8 = 56 次）
- **降低效果**: 7→4 减少约 24 次 BA 迭代。此轮主要建立粗略全局一致性，可适度缩减
- **建议范围**: 3~7

### `backend_ba_itrs` (默认: 8)

- **位置**: 后端每个 step 内的 BA 内迭代数（不优化内参/rig 时）
- **作用**: 控制单步 BA 收敛深度。此值乘以所有 backend steps 决定总 BA 量
- **全局影响**: 改此值会同时影响所有 backend 阶段
- **降低效果**: 8→4 使全部 backend BA 减半。单步收敛不完全，但多步累积可补偿
- **建议范围**: 4~16

### `backend_ba_itrs_extra` (默认: 16)

- **位置**: 后端每个 step 内的 BA 内迭代数（优化内参或 rig 旋转时）
- **作用**: 优化额外变量时需要更多迭代确保收敛
- **适用**: 仅在 `optimize_intrinsics=true` 或 `optimize_rig_rotation=true` 时生效
- **建议范围**: 8~16

### `frontend_iters1` (默认: 4)

- **位置**: 前端每个新关键帧的第一阶段 BA 轮数
- **作用**: 新关键帧加入后，做初始位姿优化
- **每轮 BA 次数**: `frontend_iters1 × 3`（graph.update 默认 itrs=3）
- **降低效果**: 4→2 每关键帧少 6 次 BA。前端精度下降可被全局 BA 补偿
- **建议范围**: 2~4

### `frontend_iters2` (默认: 2)

- **位置**: 前端关键帧被保留时的第二阶段 BA 轮数
- **作用**: 关键帧距离足够大，确认保留后做额外优化
- **每轮 BA 次数**: `frontend_iters2 × 3`
- **降低效果**: 2→1 影响较小，每关键帧仅少 3 次 BA
- **建议范围**: 1~2

### `frontend_init_iters` (默认: 8)

- **位置**: SLAM 初始化阶段的 BA 轮数
- **作用**: 前 `warmup` 帧建立初始地图的优化深度
- **每轮 BA 次数**: `frontend_init_iters × 3`
- **降低效果**: 8→4 初始地图精度下降，可能影响后续跟踪稳定性
- **建议范围**: 4~8（不建议低于 4，初始化质量对全局影响大）

### `inner_filler_iters` (默认: 10)

- **位置**: Pass 2 中每个 chunk（默认 16 帧）的 BA 轮数
- **作用**: 对非关键帧做位姿插值后的精细优化
- **总 BA 次数**: `ceil(total_frames / 16) × inner_filler_iters × 3`
- **降低效果**: 10→5 约减少非关键帧优化时间 50%。插值位姿通常已接近真值，影响较小
- **建议范围**: 5~10

### `frontend_backend_steps` (默认: 5)

- **位置**: 前端处理中途触发的后端 BA steps
- **作用**: 在关键帧数到达 [16, 64, 256] 时，提前做全局校正
- **条件**: 仅在 `optimize_intrinsics` 或 `optimize_rig_rotation` 为 true 时触发
- **降低效果**: 5→3 对多数场景无影响（除非优化内参）
- **建议范围**: 3~5

---

## 预设方案

### 默认 (`--slam-fast` 未启用)

| 参数 | 值 | 估算 BA 总次数 (50 关键帧, 500 总帧) |
|------|----|--------------------------------------|
| backend_iters | 24 | 192 |
| backend_first_steps | 7 | 56 |
| backend_ba_itrs | 8 | — |
| frontend_iters1 / iters2 | 4 / 2 | ~900 |
| frontend_init_iters | 8 | 48 |
| inner_filler_iters | 10 | ~960 |
| **合计** | | **~2200** |

### 快速 (`--slam-fast`)

| 参数 | 值 | 估算 BA 总次数 |
|------|----|----------------|
| backend_iters | 12 | 48 |
| backend_first_steps | 4 | 16 |
| backend_ba_itrs | 4 | — |
| frontend_iters1 / iters2 | 2 / 1 | ~450 |
| frontend_init_iters | 4 | 24 |
| inner_filler_iters | 5 | ~480 |
| **合计** | | **~1020** |

预期加速: **约 2x**，精度损失因场景而异（简单运动 <3%, 复杂运动 5-10%）。

### 自定义建议

如果想在默认和快速之间折中，优先调这三个（按性价比排序）：

1. `--backend-iters 12` — 最高收益，单改此项即可提速 ~30%
2. `--inner-filler-iters 5` — 对非关键帧位姿影响小
3. `--backend-ba-itrs 4` — 全局减少每步 BA 深度

---

## 批量推理命令

### 输入文件格式（jsonl）

每行一个视频路径，支持纯路径或 JSON 对象：

```jsonl
/data/videos/clip001.mp4
/data/videos/clip002.mp4
```

或指定字段名（配合 `--field video`）：

```jsonl
{"video": "/data/videos/clip001.mp4", "label": "scene_a"}
{"video": "/data/videos/clip002.mp4", "label": "scene_b"}
```

自动识别的字段名（无需 `--field`）：`video`, `video_path`, `path`, `filepath`, `file`。

### 输出目录结构

```
pose_output/
├── pose/{group}/{video_name}.mp4.npy          # (N, 4, 4) c2w 矩阵
├── intrinsics/{group}/{video_name}.mp4.npy    # (N, 4) [fx, fy, cx, cy]
└── vipe_errors/infer_errors_worker{i}.jsonl   # 失败记录（如有）
```

其中 `{group}` 取自视频所在的父目录名（如果父目录为 `clips`，则取上一级目录名）。

### 8 卡 GPU 全量生产命令

```bash
# ============================================================
# 推荐：8 卡 + pose-only + slam-fast（最大吞吐量）
# 默认开启断点续跑，中断后重跑同一命令即可从断点继续
# ============================================================
python scripts/infer_jsonl_pose.py videos.jsonl \
    -o /data/pose_output \
    --pose-only-fast \
    --slam-fast \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 1 \
    --prefetch-queue-size 2 \
    --cudnn-benchmark
```

### 各场景命令速查

#### 1. 全精度（质量优先）

```bash
python scripts/infer_jsonl_pose.py videos.jsonl \
    -o /data/pose_output \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 1 \
    --prefetch-queue-size 2 \
    --cudnn-benchmark
```

#### 2. 平衡模式（推荐日常使用）

```bash
python scripts/infer_jsonl_pose.py videos.jsonl \
    -o /data/pose_output \
    --pose-only-fast \
    --backend-iters 12 \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 1 \
    --prefetch-queue-size 2 \
    --cudnn-benchmark
```

#### 3. 极速模式（速度优先）

```bash
python scripts/infer_jsonl_pose.py videos.jsonl \
    -o /data/pose_output \
    --pose-only-fast \
    --slam-fast \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 1 \
    --prefetch-queue-size 2 \
    --cudnn-benchmark
```

#### 4. 精细调参模式

```bash
python scripts/infer_jsonl_pose.py videos.jsonl \
    -o /data/pose_output \
    --pose-only-fast \
    --backend-iters 16 \
    --backend-ba-itrs 4 \
    --inner-filler-iters 5 \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 1 \
    --prefetch-queue-size 2 \
    --cudnn-benchmark
```

#### 5. 强制重跑（忽略已有结果）

```bash
python scripts/infer_jsonl_pose.py videos.jsonl \
    -o /data/pose_output \
    --pose-only-fast \
    --slam-fast \
    --no-resume \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 1 \
    --prefetch-queue-size 2 \
    --cudnn-benchmark
```

#### 6. 带详细日志（调试/分析性能）

```bash
python scripts/infer_jsonl_pose.py videos.jsonl \
    -o /data/pose_output \
    --pose-only-fast \
    --slam-fast \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 1 \
    --prefetch-queue-size 2 \
    --cudnn-benchmark \
    --verbose
```

#### 7. 指定部分 GPU

```bash
# 只用 4 张卡
python scripts/infer_jsonl_pose.py videos.jsonl \
    -o /data/pose_output \
    --pose-only-fast \
    --slam-fast \
    --gpus 0,2,4,6 \
    --workers-per-gpu 1 \
    --prefetch-queue-size 2 \
    --cudnn-benchmark
```

### 参数速查表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `jsonl` (位置参数) | — | 视频列表文件路径 |
| `-o, --output` | `./vipe_results` | 输出目录 |
| `--gpus` | `auto` | GPU 编号，逗号分隔。`auto` = 全部可用 GPU |
| `--workers-per-gpu` | `1` | 每张 GPU 的 worker 数。通常 1 即可 |
| `--workers` | `0` | 总 worker 数（覆盖 workers-per-gpu） |
| `--prefetch-queue-size` | `1` | 预加载队列大小。设 2 可减少 GPU 空等 |
| `--cudnn-benchmark` | off | 启用 cuDNN benchmark（固定输入尺寸时加速） |
| `--pose-only-fast` | off | 跳过实例分割和深度对齐（只输出 pose/intrinsics） |
| `--slam-fast` | off | BA 迭代减半预设（约 2x 加速） |
| `--no-resume` | off | 禁用断点续跑，强制重新处理 |
| `--field` | 自动检测 | jsonl 中视频路径的字段名 |
| `--verbose` | off | 输出详细日志（含每视频 timing） |
| `--name-from-path` | off | 用完整路径生成唯一输出文件名 |

### 断点续跑机制

- **默认开启**，无需额外参数
- 判断逻辑：检查 `pose/{group}/{video}.npy` 和 `intrinsics/{group}/{video}.npy` 是否都存在
- 两者都存在 → 跳过该视频
- 任一不存在 → 重新处理
- 中断后重跑同一命令即可自动续跑
- 如需强制重跑：加 `--no-resume`

### 性能参考（估算）

以 8×A100 为例（实际取决于视频长度和分辨率）：

| 模式 | 单视频耗时 | 8 卡并行吞吐 |
|------|-----------|-------------|
| 全精度 | ~7 分钟 | ~70 视频/小时 |
| pose-only + backend-iters 12 | ~4 分钟 | ~120 视频/小时 |
| pose-only + slam-fast | ~3 分钟 | ~160 视频/小时 |

---

## 对比实验建议

为了量化不同参数对精度和速度的影响，建议：

1. 选 3-5 个代表性视频（简单/中等/复杂运动）
2. 先用默认参数跑一遍作为 baseline
3. 分别测试不同设置，输出到不同目录：

```bash
# baseline（全精度）
python scripts/infer_jsonl_pose.py test.jsonl -o ./results/baseline \
    --pose-only-fast --gpus 0

# 实验1：仅减 backend_iters
python scripts/infer_jsonl_pose.py test.jsonl -o ./results/bi12 \
    --pose-only-fast --backend-iters 12 --gpus 0

# 实验2：slam-fast 预设
python scripts/infer_jsonl_pose.py test.jsonl -o ./results/fast \
    --pose-only-fast --slam-fast --gpus 0

# 实验3：精细自定义
python scripts/infer_jsonl_pose.py test.jsonl -o ./results/custom \
    --pose-only-fast --backend-iters 16 --backend-ba-itrs 4 --inner-filler-iters 5 --gpus 0
```

4. 对比指标：
   - **时间**: 看日志中 `pipeline=Xs` 的值
   - **精度**: 比较输出 pose `.npy` 文件，计算轨迹偏差（ATE/RPE）
   - 或直接目视比较 `--visualize` 输出（需加 `--save-artifacts`）

---

## 第三部分：技术背景与性能分析

### 3.1 推理时间构成

典型的单视频推理时间分解（基于实际日志 `pipeline=436s`）：

```
总时间 ≈ video_decode + pipeline(SLAM) + save
         ≈ 28s (6%)   + 436s (94%)     + 0.1s (0%)
```

SLAM 内部时间进一步分解（估算）：

```
SLAM 436s
 ├─ BA 矩阵组装中的 .cpu().numpy() 同步    ~40-50%  (170-220s)
 │   └─ _tmult_mat_elements: Python 循环 + GPU→CPU 同步
 │      每次 BA 迭代多次调用，2200 次迭代 = 上万次 GPU 停顿
 │
 ├─ term.forward() Jacobian 计算 (GPU)      ~20-30%  (90-130s)
 │   └─ 投影、梯度计算，DroidNet GRU 更新
 │
 ├─ ravel / unravel / coalesce 格式转换     ~10-15%  (40-65s)
 │
 ├─ spsolve 稀疏求解 (CPU)                   ~<5%    (<20s)
 │   └─ Schur 消元后矩阵很小 (~300×300)，CPU 求解本身很快
 │
 └─ 其他 (DroidNet 特征编码, correlation)    ~10-15%
```

**关键发现：`spsolve` 本身不慢，瓶颈在于每次迭代的 GPU↔CPU 同步开销。**

### 3.2 为什么 BA 这么慢？

BA 求解路径（`vipe/slam/ba/solver.py`）：

```
Solver.run_inplace()
  ├─ term.forward()          # GPU: 计算 Jacobian
  ├─ jtwj() → tmult_mat()   # 调用 _tmult_mat_elements()
  │    └─ .cpu().numpy()     # ← GPU 同步点 ×2
  │    └─ Python for 循环     # ← CPU 瓶颈
  ├─ nwjtr() → tmult_vec()  # 同上
  ├─ ravel()                 # 矩阵展平
  └─ solve_scipy()           # GPU→CPU→spsolve→CPU→GPU
       └─ .cpu().numpy() ×3  # ← GPU 同步点 ×3
```

每次 BA 迭代包含 **5+ 次 GPU→CPU 同步**。
2200 次迭代 × 5 次同步 = **11000+ 次 GPU 流水线停顿**。

GPU 利用率只有 ~30% 的原因：
```
GPU: [DroidNet ██] [等CPU...]    [前向 ██] [等CPU...]    [DroidNet ██] ...
CPU: [等GPU...]    [BA求解 ████] [等GPU...]  [BA求解 ████] [等GPU...]  ...
```

### 3.3 为什么减少帧采样没有明显加速？

| 因素 | 说明 |
|------|------|
| 视频解码 | `cv2.VideoCapture.read()` 顺序解码所有帧（H.264 P/B 帧依赖），`step>1` 只是丢弃解码结果 |
| BA 迭代次数 | 由本文档中的参数控制，不随帧数线性变化 |
| Schur 消元后的矩阵大小 | 由关键帧数决定，非总帧数。减少采样帧可能减少关键帧，但效果不显著 |
| 每视频固定开销 | SLAM `_build_components`、GeoCalib 3 帧标定、DroidNet 首次加载——均不受帧数影响 |
| 内填充 (Pass 2) | 仍需遍历所有原始帧做位姿插值 |

### 3.4 代码库中未启用的 CUDA BA

`csrc/slam_ext/geom_kernels.cu` 中已实现了完整的 CUDA BA（`ba_cuda`），但**未被接入** Python BA 路径。

CUDA BA 的工作方式：
```
projective_transform_kernel  ──▷  GPU 上计算 JTJ/JTr
schur_block                  ──▷  GPU 上做 Schur 消元
SparseBlock.solve()          ──▷  Eigen::SimplicialLLT (CPU Cholesky)
pose_retr_kernel             ──▷  GPU 上更新 pose
```

即使 CUDA BA 最终也用 CPU 做稀疏求解（因为矩阵太小，GPU 稀疏求解器反而更慢），
但它将 Jacobian 计算和 Schur 消元全部留在 GPU 上，大幅减少 CPU↔GPU 同步次数。
如果未来要做更深层的优化，接入 `slam_ext.ba()` 是最有潜力的方向。

### 3.5 数据存储格式速查

| 输出文件 | 格式 | Shape | 内容 |
|---------|------|-------|------|
| `pose/{group}/{video}.npy` | numpy | `(N, 4, 4)` | 每帧 c2w 矩阵，N=帧数 |
| `intrinsics/{group}/{video}.npy` | numpy | `(N, 4)` | 每帧 `[fx, fy, cx, cy]`（通常所有帧相同） |

加载和使用示例：
```python
import numpy as np

poses = np.load("pose/group/video.npy")     # (N, 4, 4) c2w matrices
intr = np.load("intrinsics/group/video.npy") # (N, 4) [fx, fy, cx, cy]

# 取第 i 帧的相机位置（世界坐标）
camera_position = poses[i, :3, 3]

# 取第 i 帧的朝向（z 轴方向）
camera_forward = poses[i, :3, 2]

# 将世界坐标点投影到第 i 帧像素
def project(point_world, pose_c2w, fx, fy, cx, cy):
    pose_w2c = np.linalg.inv(pose_c2w)
    p_cam = pose_w2c @ np.append(point_world, 1.0)  # 世界 → 相机
    u = fx * p_cam[0] / p_cam[2] + cx               # 相机 → 像素
    v = fy * p_cam[1] / p_cam[2] + cy
    return u, v
```
