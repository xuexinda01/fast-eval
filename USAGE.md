# FastWAM VLN 评测使用说明

## 启动评测

### 自动检测最新 checkpoint（推荐）

```bash
cd /apdcephfs_tj5/share_302528826/xxd/fastwam_vln_eval
nohup bash run_fastwam_eval.sh > run_eval_launch.log 2>&1 &
```

自动扫描 `/apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4` 下最新的 `step_*.pt`。

### 指定 checkpoint

```bash
nohup bash run_fastwam_eval.sh /path/to/step_001200.pt > run_eval_launch.log 2>&1 &
```

### 启动 Waypoint 模式（极坐标连续推理）

第 4 个参数传 `1` 即可开启 waypoint 模式：

```bash
# 自动检测最新 checkpoint，waypoint 模式
nohup bash run_fastwam_eval.sh "" 1 9527 1 > run_eval_launch.log 2>&1 &

# 指定 checkpoint，waypoint 模式
nohup bash run_fastwam_eval.sh /path/to/step_001700.pt 1 9527 1 > run_eval_launch.log 2>&1 &
```

### 参数说明

```
bash run_fastwam_eval.sh [checkpoint_path] [gpu_id] [port] [waypoint_mode]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| checkpoint_path | 自动检测最新 | 主模型 checkpoint 路径 |
| gpu_id | 1 | 使用的 GPU 编号 |
| port | 9527 | server 端口 |
| waypoint_mode | 0 | `1` = 极坐标连续 waypoint 推理；`0` = 传统离散动作 |

---

## 推理模式说明

### 离散动作模式（默认，waypoint_mode=0）

每步输出 LEFT / RIGHT / FORWARD / STOP 之一，由 `fastwam_traj_to_actions()` 将 32 个 waypoint 映射得到。

停止条件：**超过 75%** 的 moving_flags < 0.5 → STOP。

### Waypoint 模式（waypoint_mode=1）

直接使用模型第一个 waypoint 计算连续目标点（极坐标转换），不经过离散动作映射。

计算方式：
```
r     = sqrt(x₀² + y₀²)       # 第一个 waypoint 的欧式距离
theta = trajectory[0][2]        # 模型输出的第 3 维
x_target = r * cos(theta)
y_target = r * sin(theta)
```

停止条件：**超过 90%** 的 moving_flags < 0.5 → STOP（更难停，走更远）。

返回值：`{'action': [x_target, y_target]}` 或 `{'action': [0]}`（STOP）。

---

## 查看日志和结果

### 日志位置

```
auto_eval_logs/server_<step>_<timestamp>.log   # server 加载模型、推理日志
auto_eval_logs/client_<step>_<timestamp>.log   # Habitat 评测日志
```

实时查看：

```bash
tail -f auto_eval_logs/server_step_001200_20260512_004826.log
```

### 结果位置

```
/apdcephfs_tj5/share_302528826/xxd/fastwam_nav_eval_results/<step>_<timestamp>/progress.json
```

---

## 修改停止条件

停止条件在 `traj_utils.py` 中。

### 离散模式（fastwam_traj_to_actions）

**当超过 75% 的 waypoint 的 `moving_flag < 0.5` 时，输出 STOP。**

```python
if stop_ratio > 0.75:   # ← 修改这里调整阈值
    return [STOP]
```

### Waypoint 模式（fastwam_traj_to_waypoint）

**当超过 90% 的 waypoint 的 `moving_flag < 0.5` 时，返回 None（STOP）。**

```python
if stop_ratio > 0.90:   # ← 修改这里调整阈值
    return None
```

### 调整 moving_flag 比例阈值

| 阈值 | 效果 |
|------|------|
| `> 0.5` | 超过一半说 stop → 停（更容易停） |
| `> 0.75` | 超过四分之三说 stop → 停（离散模式默认） |
| `> 0.9` | 超过九成说 stop → 停（waypoint 模式默认） |

### 启用 Stop Head（独立分类器）

Stop head 默认**关闭**。启用方式：在 `run_fastwam_eval.sh` 中修改 server 启动参数：

```bash
# 当前（关闭 stop head）：
--stop_head_scan_dir ""

# 改成（自动检测最新 stop head checkpoint）：
--stop_head_threshold 0.7 \
--stop_head_ensemble \
# 并删除 --stop_head_scan_dir "" 这行
```

Stop head checkpoint 扫描目录：`/apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4/stop_head`

| 参数 | 说明 |
|------|------|
| `--stop_head_threshold` | stop_prob 超过此值触发停止（默认 0.5） |
| `--stop_head_ensemble` | True = moving_flag 或 stop_head 任一触发即停；False = 仅 stop_head |

---

## 停止当前测试

### 查找正在运行的进程

```bash
ps aux | grep -E "(fastwam_server|torchrun|eval\.py)" | grep -v grep
```

输出示例：
```
xxd  1620166  ...  python fastwam_server.py --checkpoint ...
xxd  1630228  ...  torchrun --nproc_per_node=1 ...
xxd  1630374  ...  python scripts/eval/eval.py ...
```

### 一键 kill 所有相关进程

```bash
pkill -f fastwam_server.py; pkill -f "eval\.py.*fastwam"; pkill -f "torchrun.*2345"
```

或者逐个 kill：

```bash
kill <server_pid> <torchrun_pid> <eval_pid>
```

### 确认已停止

```bash
ps aux | grep -E "(fastwam_server|torchrun|eval\.py)" | grep -v grep
# 无输出则说明已全部停止
```

### 注意

- 如果 `auto_eval_watcher.sh` 也在运行（会每 30 分钟自动重启），需要一起 kill：
  ```bash
  pkill -f auto_eval_watcher.sh
  ```

---

## 注意事项

- **不要同时运行 `auto_eval_watcher.sh` 和 `run_fastwam_eval.sh`**，两者会抢占同一个 GPU 和端口（9527）
- Server 加载模型约需 **5 分钟**，期间日志无输出属正常，出现 `Server listening on` 后 client 自动启动
- `CUDA_VISIBLE_DEVICES=1` + `--device cuda:0` 是正确写法，实际使用的是物理 GPU 1


---

## 查看日志和结果

### 日志位置

```
auto_eval_logs/server_<step>_<timestamp>.log   # server 加载模型、推理日志
auto_eval_logs/client_<step>_<timestamp>.log   # Habitat 评测日志
```

实时查看：

```bash
tail -f auto_eval_logs/server_step_001200_20260512_004826.log
```

### 结果位置

```
/apdcephfs_tj5/share_302528826/xxd/fastwam_nav_eval_results/<step>_<timestamp>/progress.json
```

---

## 修改停止条件

停止条件在 `traj_utils.py` 的 `fastwam_traj_to_actions()` 中。

### 当前规则

模型每步输出 32 个 waypoint，每个 waypoint 有 `moving_flag` 值（0=停，1=走）。  
**当超过 75% 的 waypoint 的 `moving_flag < 0.5` 时，输出 STOP。**

```python
# traj_utils.py
stop_ratio = stop_steps / len(moving_flags)
if stop_ratio > 0.75:   # ← 修改这里调整阈值
    return [STOP]
```

### 调整 moving_flag 比例阈值

| 阈值 | 效果 |
|------|------|
| `> 0.5` | 超过一半说 stop → 停（更容易停） |
| `> 0.75` | 超过四分之三说 stop → 停（当前设置） |
| `> 0.9` | 超过九成说 stop → 停（更难停，走更远） |

### 启用 Stop Head（独立分类器）

Stop head 默认**关闭**。启用方式：在 `run_fastwam_eval.sh` 中修改 server 启动参数：

```bash
# 当前（关闭 stop head）：
--stop_head_scan_dir ""

# 改成（自动检测最新 stop head checkpoint）：
--stop_head_threshold 0.7 \
--stop_head_ensemble \
# 并删除 --stop_head_scan_dir "" 这行
```

Stop head checkpoint 扫描目录：`/apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4/stop_head`

| 参数 | 说明 |
|------|------|
| `--stop_head_threshold` | stop_prob 超过此值触发停止（默认 0.5） |
| `--stop_head_ensemble` | True = moving_flag 或 stop_head 任一触发即停；False = 仅 stop_head |

---

## 停止当前测试

### 查找正在运行的进程

```bash
ps aux | grep -E "(fastwam_server|torchrun|eval\.py)" | grep -v grep
```

输出示例：
```
xxd  1620166  ...  python fastwam_server.py --checkpoint ...
xxd  1630228  ...  torchrun --nproc_per_node=1 ...
xxd  1630374  ...  python scripts/eval/eval.py ...
```

### 一键 kill 所有相关进程

```bash
pkill -f fastwam_server.py; pkill -f "eval\.py.*fastwam"; pkill -f "torchrun.*2345"
```

或者逐个 kill：

```bash
kill <server_pid> <torchrun_pid> <eval_pid>
```

### 确认已停止

```bash
ps aux | grep -E "(fastwam_server|torchrun|eval\.py)" | grep -v grep
# 无输出则说明已全部停止
```

### 注意

- 如果 `auto_eval_watcher.sh` 也在运行（会每 30 分钟自动重启），需要一起 kill：
  ```bash
  pkill -f auto_eval_watcher.sh
  ```

---

## 注意事项

- **不要同时运行 `auto_eval_watcher.sh` 和 `run_fastwam_eval.sh`**，两者会抢占同一个 GPU 和端口（9527）
- Server 加载模型约需 **5 分钟**，期间日志无输出属正常，出现 `Server listening on` 后 client 自动启动
- `CUDA_VISIBLE_DEVICES=1` + `--device cuda:0` 是正确写法，实际使用的是物理 GPU 1
