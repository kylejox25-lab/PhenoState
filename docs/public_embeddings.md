# RxRx3-core 已有 embedding：方法与统一评价

这三种 Parquet 文件已经包含每个 well 的图像表征。直接读取特征即可评价，不需要 checkpoint、GPU、重新推理或重新训练，也不使用基因、guide、化合物或浓度作为编码器输入。

## 三种方案是什么

| 数据集文件名中的方案 | 架构 | 编码器层数 | 原生维度 | 模型规模与训练来源 |
| --- | --- | --- | --- | --- |
| OpenPhenom | CA-MAE，ViT-S/16 | 12 | 384 | 约25M参数；RxRx3 + JUMP/cpg0016 的公开子集 |
| MAE-L/8，RPI-93M | 多通道联合 patch 的 MAE，ViT-L/8 | 24 | 1024 | Phenom-1，约307M参数；RPI-93M |
| MAE-G/8，PP-16M trimmed | 多通道联合 patch 的 MAE，ViT-G/8 | 原48层，发布表征取第38层 | 1664 | Phenom-2，完整编码器约1.86B参数；Phenoprints-16M |

S、L、G 表示不同大小的 ViT；`/8`、`/16` 表示 patch 的空间边长，不是 embedding 维数。G 的参数量指完整模型，不能当成裁剪到第38层之后的精确参数量。架构与训练来源见 [RxRx3-core 论文 §3](https://arxiv.org/html/2503.20158v1#S3)，层数和维度见 [ViTally Consistent 附录 A.2](https://arxiv.org/html/2411.02572v2#A1.SS2)。Notebook 01 会读取本地文件的实际行数与特征列数，避免只信名称。

### OpenPhenom

它是 channel-agnostic MAE：每个通道分别划成16×16 patch，共享 patch 投影和 Transformer 编码器，各通道 token 可以互相提供上下文。训练时遮挡部分 token，利用通道 decoder 重建。与把六通道一起压成一个 token 的标准 MAE 相比，它更便于处理不同通道数，但同样图像大小下 token 更多。

以256×256、六通道 crop 为例，总 token 为 `6 × (256/16)^2 = 1536`；75%遮挡时编码器看384个 token。推理不做随机遮挡，公开实现对非 CLS tokens 求平均，得到384维向量；按通道输出的拼接模式维度不同，不应与数据集提供的384维表征混淆。见 [模型卡](https://huggingface.co/recursionpharma/OpenPhenom) 和 [公开推理实现](https://github.com/recursionpharma/maes_microscopy/blob/trunk/huggingface_mae.py)。

### MAE-L/8

它把同一空间位置的六通道8×8像素一起投影为 token。256×256 crop 有1024个空间 token，训练时只编码其中25%的可见部分，decoder 学习恢复隐藏区域；推理使用完整干净图像的编码特征，输出1024维向量。

这是大规模显微图像预训练模型，而不是本项目 encoder_dim=512 的 MAE。其训练规模、归一化和训练目标也不能仅凭“MAE”名称视为相同。MAE-L/8 RPI-93M 与 MAE-L/8 PP-16M 是不同预训练版本；这里接入的文件明确为 RPI-93M。

### MAE-G/8

它继续扩大标准 MAE 编码器。重要特点是提供的 `trimmed` 表征取自中间第38层，而非最后第48层：研究通过逐层线性探测选择更适合生物表征的层，说明重建网络越靠后的层并不必然产生更好的下游特征。

Phenoprints-16M 是经筛选、增加有一致形态信号扰动占比的训练数据。其优势可能同时来自数据选择、规模和取层方式，不能把与本项目的差异仅归因为 Transformer 更大。详见 [ViTally Consistent §3–4](https://arxiv.org/html/2411.02572v2)。

## 与本项目模型比较时的边界

- 这是同一套孔位上的预训练参考比较，不是同预算的模型架构消融。OpenPhenom 已在 RxRx3 上训练；其它预训练数据的逐孔重叠尚未审计。当前整板 split 无法证明这些编码器从未见过 query 图像。
- RxRx3-core 论文描述的推理将512×512图像拆成四个256×256 crop，再平均 crop embeddings；本项目模型直接处理完整512×512。这也是方法差异，不能称作完全相同的输入协议。
- 论文评测还使用 PCA-CS 批次校正。当前程序的 `raw` 只表示“不额外执行本项目标准化”，不保证提供方文件从未预处理。不能用本项目分数直接对照论文表格。
- 三种维度可以直接各自做 cosine 检索，不需要强行统一维度。先比较 raw，再用 train-only 标准化作为后处理消融；不在 val/test 上拟合均值、方差或 PCA。

上述 crop 聚合及官方预处理见 [RxRx3-core 论文 §4](https://arxiv.org/html/2503.20158v1#S4)。

## 如何划分已有 embedding

划分依据是元数据中的 `(experiment_name, plate)`，之后按 `well_id` 连接特征；不能按 Parquet 行位置随机切分。优先直接复用已有 MDS 目录的 `plate_splits.json`。无需物理拆成三份特征文件，评价时只保留 train 和当前 query split。

没有已有划分时，可以单独生成，与图像转换使用完全相同的 `make_plate_splits`：

```bash
python -m phenostate.data.splits \
  --metadata /path/rxrx3_core/metadata_rxrx3_core.csv \
  --output /path/evaluation/plate_splits.json --seed 2026
```

安装更新后的项目后也可使用 `phenostate-split`。每个 experiment 内按整板分配：val、test 各 `max(1, round(板数×0.1))`，其余 train。已有文件不会被覆盖。不要给每个模型分别生成随机划分，也不要重新生成一个不同 seed 的文件代替训练时划分。

## 评价三个公开 embedding

先核实阴性对照 JSON，尤其 COMPOUND 的车辆/背景条件。运行完整四任务示例：

```bash
DATA=/path/rxrx3_core
python -m phenostate.retrieval \
  --embedding OpenPhenom "$DATA/OpenPhenom_rxrx3_core_embeddings.parquet" \
  --embedding MAE-L-8 "$DATA/MAE-L_8_RPI-93M_rxrx3_core_embeddings.parquet" \
  --embedding MAE-G-8 "$DATA/MAE-G_8_PP-16M_trimmed_rxrx3_core_embeddings.parquet" \
  --metadata "$DATA/metadata_rxrx3_core.csv" \
  --splits /path/evaluation/plate_splits.json \
  --control-rules configs/negative_controls.example.json \
  --tasks activity same-guide compound cross-guide \
  --query-split val --output-dir /path/results/public_val_raw
```

若已有完整 MDS，将 `--splits ...` 换成 `--mds-root /path/rxrx3_mds`，还能核验 metadata 与 manifest SHA。两者二选一。不做 activity 时可以去掉该任务和 control-rules，先验证三个检索项。

可以在同一个命令内追加自训练模型，保证共同孔位与配对差值可比：

```bash
  --model mae:local_MAE /path/mae/embedding_train.parquet /path/mae/embedding_val.parquet
```

`--embedding` 和 `--model` 可以同时、多次使用，模型名必须唯一。整表支持 `well_id + feature_0...`，自训练导出支持 `well_id + embedding` 向量列；不同方法允许不同维度。

程序检查 ID 唯一性、元数据归属、整板划分、完整覆盖与有限值。模型间取共同有效孔位，使用相同 query、候选和对照规则。相似度分块计算，不建立全量 well×well 矩阵；多种特征同时保存在内存中仍可能占用数 GB，尤其1664维特征。`--allow-partial` 只用于明确的部分覆盖实验，不能静默省略缺失孔位。

train 是检索 gallery，val/test 是 query，不是重新训练三个公开模型。`--preprocessing standardize` 只从共同 train wells 拟合每维均值和方差。结果会记录每种来源的实际维度、文件哈希和来源类型；没有本项目 checkpoint 来源标签的公开特征显示为来源未验证，这不表示文件无效。

## 看什么结果

| 任务 | 主指标 | 辅助指标与解释 |
| --- | --- | --- |
| activity | 对严格匹配阴性对照的 mAP | energy distance，AP/energy 各自的显著活性比例；按 CRISPR/COMPOUND 分开 |
| same-guide | mAP | Recall@K、percent replicating；同 guide 的跨板重复 |
| compound | mAP | Recall@K、percent replicating；化合物＋浓度，不跨剂量混组 |
| cross-guide | MRR | Recall@1/5、mAP；不同 guide 的同基因表型一致性 |

完整协议、排除条件和指标定义见 [representation_evaluation.md](representation_evaluation.md)。输出仍使用同一套 `summary.csv`、`protocol.json`、`coverage.json`、配对差值和逐任务明细。

运行后设置：

```bash
export PHENOSTATE_RESULTS_DIR=/path/results/public_val_raw
jupyter lab
```

打开 Notebook 03 对七模型共同孔位进行重复/对照相似度、表型强度与重复性、PCA/可选 UMAP、批次近邻富集及扰动/plate 差值诊断，已有主指标 CI、Recall、活性/重复比例和配对差值保留在附录。输入配置和输出见 [七模型诊断说明](embedding_diagnostics.md)。Notebook 01 负责原图探索，02 负责自训练模型的训练日志与重建；已有向量不能单独恢复重建图。
