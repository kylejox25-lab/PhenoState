# PhenoState

RxRx3-core 六通道图像的自监督表征学习与下游评价。编码器只输入图像，基因、guide、化合物和浓度用于评价分组。

## 环境

```bash
conda env create -f environment.yml
conda activate phenostate
python -m pip install -e . --no-deps --no-build-isolation
```

## 代码

```text
src/phenostate/
  models/             MAE、CA-MAE、DiffMAE、CNN-MAE 与模型工厂
  data/module.py      LightningDataModule，固定 MDS 划分
  train/module.py     LightningModule，loss、日志、优化器及调度
  train/mae.py        Trainer 命令入口
  data/source.py      原始数据读取与按实验内整板划分
  data/convert.py     Parquet → MDS
  data/inference_loader.py  本地 MDS 有界线程读取
  infer.py            流式导出 well embedding
  evaluate.py         遮挡重建指标
  retrieval.py        独立 embedding 下游评价
```

| 模型配置 | 输入 | 编码器 | Decoder | embedding |
| --- | --- | --- | --- | --- |
| mae_comparison.json | 6×512×512，patch8 | 512维／12层／8 heads | 256维／4层 | 512维 |
| ca_mae.json | 通道 patch token | 共享编码器，无 channel ID | 六个通道 decoder | 512维 |
| diffmae.json | 干净可见 patch | 512维／12层／8 heads | 带噪隐藏 query，预测 x0 | 512维 |
| cnn_mae.json | 空间 patch 遮挡 | 12个残差卷积块 | 4个残差块及转置卷积 | 512维 |

`mae_small.json` 是 patch16、384维配置。模型实际结构以 checkpoint 的 `hyper_parameters.model_config` 为准。四模型均以隐藏像素重建 MSE 训练；`norm_pix_loss=true` 按 patch、按通道标准化目标，编码器图像输入为 float32 `[0,1]`。DiffMAE 的随机时间步去噪 loss 与普通 MAE 的重建难度不同，正式模型比较使用共同下游协议。

## 数据

```bash
phenostate-convert --source /path/rxrx3_core --output /path/rxrx3_mds
phenostate-check-loader --mds-root /path/rxrx3_mds --split val --batch-size 1
```

每个样本是一个 well 的六通道 s1 中心裁剪，没有单细胞分割。转换保留 JP2 字节，写入未额外压缩的 MDS 分片。每个 experiment 内整块 plate 划分：val、test 各 `max(1, round(板数×0.1))`，其余 train，默认 seed=2026。具体归属保存为 plate_splits.json，数量与元数据哈希保存为 manifest.json。不要修改已有清单内容。

## Lightning 训练

```bash
phenostate-train \
  --mds-root /path/rxrx3_mds --output /path/runs/mae \
  --model-config configs/mae_comparison.json \
  --epochs 50 --batch-size 1 --accumulation-steps 64 \
  --precision 16-mixed --checkpoint-every 500
```

Trainer 负责 AMP、反向传播、梯度累积和范数1裁剪；LightningModule 配置 AdamW、5% warmup＋cosine 调度。默认单 GPU，`--accelerator cpu --precision 32-true` 可做小样本验证。训练 DataLoader 默认 `--num-workers 0`，避免在 NFS 临时目录创建多进程 IPC 文件；增加进程数前确认节点临时目录合适。

```text
RUN_DIR/
  checkpoints/best.ckpt       验证 val_loss 最低
  checkpoints/last.ckpt       最近验证结束的状态
  checkpoints/step-*.ckpt     最近的周期保存点
  logs/metrics.csv            train_loss_step、train_loss_epoch、val_loss、学习率
  logs/hparams.yaml           模型、数据和训练超参数
```

续训使用相同模型、数据与训练设置，并添加 `--resume /path/runs/mae/checkpoints/last.ckpt`。优化器、scheduler、循环和有状态 loader 由 Lightning 恢复。总调度步数按完整训练孔位数、batch size、梯度累积及 epochs 预计算并保存；设置 max-steps 时以其作为训练步数上限。`best.ckpt` 只代表重建目标最佳，不代表生物学表征最佳。

## 导出与重建评价

```bash
phenostate-embed \
  --checkpoint /path/runs/mae/checkpoints/best.ckpt \
  --mds-root /path/rxrx3_mds --split train \
  --output /path/embeddings/mae/embedding_train.parquet \
  --num-workers 4 --skip-existing

phenostate-evaluate \
  --checkpoint /path/runs/mae/checkpoints/best.ckpt \
  --mds-root /path/rxrx3_mds --split val \
  --output /path/results/reconstruction.json
```

冻结模型只做前向；导出和重建评价通过有界线程读取本地 MDS，`--num-workers` 表示线程数。导出检查 manifest/checkpoint SHA、有限值、唯一 ID 与完整数量；成功后才发布正式 Parquet。`.incomplete` 不能续接或直接用于评价。

四模型批量导出不启动训练：

```bash
export RXRX3_MDS_ROOT=/path/rxrx3_mds
export PHENOSTATE_COMPARISON_ROOT=/path/runs
export PHENOSTATE_EMBEDDING_ROOT=/path/embeddings
export PHENOSTATE_MODELS="mae ca_mae diffmae cnn_mae"
bash scripts/export_embeddings.sh
```

默认导出 train、val、test；已有新格式文件校验通过后复用。新训练和导出只支持原生 Lightning `.ckpt`，不再加载以前的自定义 `.pt`。已有 embedding 文件不改写，仍可作为通用特征独立评价；不带新来源标签的文件记录为来源未验证。

## 下游任务

```bash
phenostate-retrieval \
  --model mae /path/embeddings/mae/embedding_train.parquet /path/embeddings/mae/embedding_val.parquet \
  --metadata /path/metadata_rxrx3_core.csv --mds-root /path/rxrx3_mds \
  --control-rules configs/negative_controls.example.json \
  --query-split val --output-dir /path/results/downstream_val
```

三类任务、四个计分项：activity（相对对照 mAP、energy、显著活性比例）、same-guide 和 compound（mAP、Recall、percent replicating）、cross-guide（MRR、Recall、mAP）。train 作 gallery，val/test 作 query。阴性对照配置须先核实，缺少严格匹配对照时报告不可评价。详见[任务协议](docs/representation_evaluation.md)。

数据集提供的 OpenPhenom、MAE-L/8、MAE-G/8 可以直接评价；无需重新 embedding，也无需复制为三个 split 文件。`phenostate-split` 可只从 metadata 生成相同的整板划分；`--embedding` 可与 `--model` 混合比较。详见[公开特征说明与运行命令](docs/public_embeddings.md)。

Notebook 01 展示可选 well 的 CRISPR、药物浓度及匹配阴性对照图像，并检查公开特征 schema；02 读取 Lightning CSV 日志及重建结果；03 只读取下游结果，展示指标、置信区间、覆盖率和诊断。设置 `PHENOSTATE_RESULTS_DIR` 指定03的结果目录。测试运行 `python -m pytest -q`；真实 MDS 测试需要安装 Streaming 依赖。

### RxRx3-core 论文评估与 Notebook 04

论文的三项评估共用 PCA-CS 校正，使用输入文件中各模型共同覆盖的 well。将自训练模型已有的 train、val Parquet 连续传入即可，评估会合并这些孔位；同时传入 test 文件可扩大共同覆盖范围。全表公开特征只需传一个文件。论文评估使用共同孔位构建扰动表征，Notebook 03 的 train-gallery / val-query 协议单独报告。

七模型脚本使用 `mae_patch8_head8`、`ca_mae`、`diffmae`、`cnn_mae` 的 train 和 val 文件，以及 OpenPhenom、MAE-L/8、MAE-G/8 的全表。它会先检查所有输入文件。运行环境需要 `analysis` 依赖，以及 Recursion EFAAR 仓库中的关系标签。

```bash
git clone --depth 1 https://github.com/recursionpharma/EFAAR_benchmarking.git ../data/EFAAR_benchmarking
bash scripts/evaluate_paper_seven_models.sh
```

默认从 `../data/embeddings` 读自训练向量，从 `/home/xkj2006/CellPainting/rxrx3_core` 读元数据和公开向量。路径不同时可设置 `PHENOSTATE_EMBEDDING_ROOT`、`RXRX3_CORE_ROOT`、`PHENOSTATE_PUBLIC_EMBEDDING_ROOT` 或 `PHENOSTATE_EFAAR_ANNOTATIONS_DIR`。如已导出 test，可设置 `PHENOSTATE_PAPER_SPLITS='train val test'`。相对路径以项目根目录为基准。

结果默认写入 `results/downstream_val_all/paper_7_models`，该目录在运行前需不存在。打开 [Notebook 04](notebooks/04_rxrx3_core_paper_benchmarks.ipynb) 即可读取这份结果。使用自定义目录时，设置 `PHENOSTATE_PAPER_RESULTS_DIR`，或在 Notebook 首格填写 `RESULT_DIR`。单模型或自选组合可直接调用 `python -m phenostate.paper_benchmarks --help` 查看输入参数。

| 任务 | 口径 | 输出 |
| --- | --- | --- |
| 扰动信号 | 论文 §4.1 的有偏欧氏能量距离。沿用 EFAAR 通用函数的至少 5 个处理孔和最多 1,000 个全局对照。本实现按查询基因或药物汇总，药物合并剂量，从 `EMPTY_control` 固定抽样，报告均值、MAD 及同维度高斯随机基线。 | `energy_groups.csv`、`energy_summary.csv` |
| 药物靶点 | 沿用 EFAAR 的已知活性阈值、候选抽样与绝对余弦相似度。按剂量和跨剂量最大相似度计算 AP、AUROC，保留逐药物指标的均值和中位数两种汇总。随机序列由本项目固定。 | `dti_seeds.csv`、`dti_summary.csv` |
| 基因关系 | 聚合到基因后按余弦相似度两端各 5% 计算已知关系召回。分别报告 CORUM、hu.MAP、Reactome 和 STRING 的关系覆盖。 | `gene_relationships.csv` |

`protocol.json` 记录模型文件、注释哈希、共同孔位数和评估参数，`well_manifest.csv` 列出实际使用的孔位。论文表格的分数还取决于所用孔位、对照和随机基线。参见 [RxRx3-core 论文 §4](https://arxiv.org/html/2503.20158v2#S4) 和 [EFAAR 官方 Notebook](https://github.com/recursionpharma/EFAAR_benchmarking/tree/trunk/RxRx3-core_benchmarks)。
