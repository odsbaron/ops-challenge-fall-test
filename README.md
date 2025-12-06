# README

欢迎大家来到 AFT 首届算子挑战赛！

## Overview

* 算子比赛通过 Github 的 Action 机制模拟了一个非常简易 Online Judge 平台，每周这里会切换一个 Task，每个 Task 就是不同的算子，例如第一周将会是实现高效的 `rolling_rank()`
* 评分的 Action 需要通过提交 PR 来触发，所以这也是一个熟悉 Git/Github 的一个机会
    * 注意，需要加入 AFT25 Organization 才能够触发 Action # WIP
* 由于初次尝试，系统非常可能具有极大的不稳定性，如遇 bug 请反馈给会长，相信通过大家不断的使用，系统会快速迭代起来
* 任何问题都可以随意提 Issue 或者 Discussion

## Setup Env

```bash
# 0. A env with python=3.11

# op1. pip
pip install -r requirements.txt

# op2. poetry
poetry env use python3.11
poetry install
```

## Build ops

* 在 `src/solution.py` 是参考的基础 pandas 写法，需要完成读取+计算两个步骤
    * 注意只需要完成 `ops_rolling_regbeta()` 函数，且函数签名是 `(input_path: str, window: int = 20)`
    * 用于测试的 `main` 逻辑自行完成，提交时请只保留函数
    * Action 的 Runner 给大家配备了 20 个核心，所以在本地测试的时候也尽量把多核都用上，可以通过 `top` 或者 `htop` 来监测 CPU 利用率哦
    * 加速方法参考

* 测试数据可以从 [北大网盘](https://disk.pku.edu.cn/link/AA792794F02CAD41588EB1CCCB37085C18) 获取，需要校园网

* 或者通过 Google Drive

  ```bash
  cd testcase
  
  # data_for_rolling_regbeta.parquet
  gdown --fuzzy https://drive.google.com/file/d/1kyxl5bi0pJ_RZ-Dyqto9PE4Q1IU_BSQ6/view?usp=drive_link
  
  # rolling_regbeta_v1.npy
  gdown --fuzzy https://drive.google.com/file/d/1yaVW3CrW23-72Marz8pmBHU04LwCTO_V/view?usp=drive_link
  ```

* 以上数据集每周会更新～

## Test

* 参考的测试脚本在 `localTest.py`

  ```bash
  python localTest.py \
  	--entry_point ops_rolling_regbeta \
  	--input_path ./testcase/data_for_rolling_regbeta.parquet \
  	--ref_ans_path ./testcase/rolling_regbeta_v1.npy\
  	--window 20
  ```

## Submit PR

1. 点击 fork 将仓库到自己的 Github 下
2. 建立一个新分支，例如 `git checkout -b week1`
3. 从北大网盘下载 Testcase，在本地修改并测试 `src/solution.py` 后 push 回本地
4. 在你的本地仓库提交 PR
5. 提交一次即可，同一个 PR 可以反复修改然后重跑 Action，但后端的 LeaderBoard 可能只会计算前三次的成绩
6. 如果 Action 失败（非常有可能）可以截图发给学术副会

## Timeline

1. week 1：`rollling_rank()`
2. week 2: `rolling_regbeta()`
    *  之后改为使用 float64 数据集
