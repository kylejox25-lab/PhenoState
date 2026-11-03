# 下游任务与独立 embedding 评测

当前取并集为三类任务、四个计分项。编码器只接收图像；元数据用于评价，不编码扰动条件。

| 参数 | 任务 | 分组／正例 | 指标 |
| --- | --- | --- | --- |
| activity | 相对阴性对照的表型活性 | CRISPR 按 guide，药物按化合物＋浓度 | 相对对照 mAP、分层 energy distance、显著活性扰动比例 |
| same-guide | 同 guide、同 experiment 跨板重复检索 | 相同 guide；同基因其他 guide 忽略 | mAP、Recall@K、percent replicating |
| compound | 同化合物、同剂量、同 experiment 跨板检索 | 化合物＋浓度；候选固定相同剂量 | mAP、Recall@K、percent replicating |
| cross-guide | 跨 experiment、跨 guide 基因检索 | 同基因原型，完全排除 query guide | MRR、Recall@1/5、mAP |

cross-guide 每次仅一个正例，所以 AP=RR、Recall@K=Hit@K，并非多份独立证据。不同基因功能关系检索仍需外部真值，本阶段不运行。

## query/gallery 与公平比较

val/test 板的单 well 作 query，train 板作固定参考库。验证用于确定方案，测试在规则冻结后运行。不能用测试活性筛选好样本，再把该子集检索分数当作无偏主结果。

same-guide 和 compound 均要求相同 experiment、不同 plate。cross-guide 的候选来自另一个 experiment，同一候选 experiment 单独构成任务；先平均 guide 内重复，再等权平均不同 guide，得到每基因一个原型。

CRISPR 检索先平均同 well 的任务，再平均同 guide 的 well、同基因的 guide，最后基因宏平均；药物按化合物＋浓度宏平均。报告候选规模、正例数和覆盖率。全部模型按 well_id 对齐到共同孔位，不依赖文件行顺序。NaN/Inf 默认报错，显式 --drop-nonfinite 后共同剔除；零向量／Pearson 常量向量共同剔除。退化基因原型严格报错。模型差值采用相同分组的配对 bootstrap；不能把该区间当作训练 seed 不确定性。

## 活性与严格匹配阴性对照

活动 mAP 的正例是 train 参考库中的同处理重复，负例仅为这些参考板上匹配的阴性对照。query 板的对照用于分层检验。其他普通处理不充当阴性对照。

控制规则必须通过 --control-rules 显式给出，不按名称包含 control 自动选择，不把 EMPTY_control 等同 DMSO，不使用已标注阳性对照。强制匹配 experiment、plate、perturbation_type；额外字段由 match_fields 指定，例如 cell_type。缺少合格对照时记录不可评价原因，不跨板回退。

[negative_controls.example.json](../configs/negative_controls.example.json) 是选择模板，需核实服务器实际标签和阴性对照语义。模板 CRISPR 选择 Intron controls；COMPOUND 选择 treatment=EMPTY_control 且 well_type_label 为空。CRISPR_control 所在的 CRISPR Gene Positive Controls 不能作为负例。药物剂量用于处理分组；无药物浓度的对照不强行匹配处理浓度。若元数据有载体等字段，可增加匹配要求。

统计口径：

1. Energy 使用规范化向量的欧氏距离，分别计算每板 treated/control 的 biased energy distance，再按板平均，称为分层 energy。它不是官方 pooled ED 的直接复现，有限样本偏差由匹配零分布校准。
2. 每块板内部置换 treated/control 标签，保留处理数量及 query/gallery 角色；每次同时生成伪 query、伪参考正负例，重算完整 mAP 和 energy。
3. 同一唯一 guide／化合物＋浓度先跨 experiment 汇总观测值和置换值，再分别在 CRISPR、COMPOUND 内做 BH 校正。
4. percent_active 按 AP 的 q<0.05 定义；energy 的 q 和活性比例另报。分母是可评价的唯一扰动，必须同时报告全部扰动的覆盖率。
5. p=(1+null>=observed 的次数)/(B+1)。报告置换次数、最小 p、检验数量；大规模 BH 可能受分辨率限制。不能把零显著直接解释为没有表型。
6. 控制抽样和置换仅由元数据和 seed 决定，各模型固定相同规则。输出按扰动类型分层。

活性 AP 对精确相同的相似度采用并列顺序的解析期望，观测和置换使用同一公式，避免常量 embedding 因 well_id 排序误报活性。显式匹配字段的空值不能作为已知条件相等。

活性检验包含 train 参考样本，属于参考库支持的新板评价，不能称所有检验图像均独立留出。

## percent replicating

每个处理×experiment 只对跨板 query-gallery 重复对计算相似度中位数。零分布匹配角色、板、扰动类型、剂量和重复数，从相互不同的其他处理采样；同 guide 的零分布要求其他基因相互不同，排除同基因其他 guide。观测值严格高于零分布 95% 分位数才记作 replicating，再按处理汇总。

这是本项目的 query-gallery PR 变体，不混入纯 train-train 重复对。其他处理不足时标记不可评价，不放宽标准。记录采样上限、seed 和实际覆盖率。

## 独立运行

无需 checkpoint、GPU、原始图像或固定模型架构。运行依赖 NumPy、PyArrow 及本项目 src 包。支持唯一 well_id 加 embedding 向量列，或 well_id 加宽表数值特征列；不同来源可有不同维度。

~~~bash
python -m pip install -e . --no-deps

phenostate-retrieval \
  --embedding encoder_A /srv/features/A.parquet \
  --embedding encoder_B /srv/features/B.parquet \
  --metadata /srv/data/metadata_rxrx3_core.csv \
  --splits /srv/data/plate_splits.json \
  --control-rules configs/negative_controls.example.json \
  --query-split val --output-dir /srv/results/all_tasks_val
~~~

划分可为 JSON/CSV 的 experiment_name,plate,split，或 well_id,split；也可直接使用 metadata 的 split 列。禁止同一板跨 split。使用原项目数据可用 --mds-root 代替 --splits，额外核验 manifest、元数据哈希和转换划分。

分开导出的 train/query 文件继续用 --model，可重复指定不同来源：

~~~bash
phenostate-retrieval \
  --model mae /srv/mae/embedding_train.parquet /srv/mae/embedding_val.parquet \
  --model ca_mae /srv/ca_mae/embedding_train.parquet /srv/ca_mae/embedding_val.parquet \
  --model diffmae /srv/diffmae/embedding_train.parquet /srv/diffmae/embedding_val.parquet \
  --metadata /srv/data/metadata_rxrx3_core.csv \
  --mds-root /srv/data/rxrx3_mds \
  --control-rules configs/negative_controls.example.json \
  --query-split val --output-dir /srv/results/three_models_val
~~~

原训练脚本只导出了 val/test，需用同一份冻结 checkpoint 补导 train。native 文件核验已有来源声明；外部 embedding 不要求 native 标签，记录其来源验证状态。也可运行 python -m phenostate.retrieval，参数相同。

默认 CLI 执行全部四项；--tasks cross-guide same-guide compound 可只运行不需要阴性对照配置的任务。--tasks activity 可单独检测活性。

参数：

- --preprocessing raw|standardize：后者只在共同 train wells 上拟合均值／标准差，保存参数。
- --n-permutations：活性置换次数；--n-null：PR 零分布次数；--n-bootstrap：分组 CI 次数。
- --max-controls：活性每板对照上限；--max-replicates：PR 重复上限。
- --allow-partial：明确允许不完整导出，使用共同孔位并报告覆盖率。
- --query-split test：最终评价时换成 test 文件及新输出目录，不重新选规则。

输出目录须不存在；失败写入保留 .incomplete，不覆盖已有结果。

## 输出

summary.csv / summary.md 汇总主辅指标；summary_by_type.csv 单列 CRISPR／COMPOUND 活性，paired_differences.csv 给配对差值和 CI。MODEL/TASK/metrics.json 保留完整协议、类型分层、覆盖率与排除原因。queries.csv、groups.csv 对应 well 检索与分组结果；activity 的 blocks.csv 记录匹配、抽样和实验详情；重复任务的 replicates.csv 记录 PR 观测及零阈值。protocol.json、coverage.json、well_manifest.csv 保留输入哈希、参数与共同孔位。标准化模式另保存 standardization.npz。

## 后续消融

现有三个模型是完整方法对照。CA-MAE 优先做 2×2：有／无图像通道身份 × 当前 channel-token 掩码／共享空间掩码，保持其余结构、可见 token 数和训练设置一致；已有一格，新增三次训练形成单 seed 实验。通道身份不包含扰动条件。次优先在 MAE 内比较 norm_pix_loss=true/false。

无需重训的消融先比较 raw/训练标准化，再补相同候选规模的同实验／跨实验跨 guide 检索。当前不修改训练架构。

方法依据：[mAP 框架](https://www.nature.com/articles/s41467-025-60306-2)、[RxRx3-core](https://arxiv.org/html/2503.20158v1)。本项目使用明确的 train-gallery 分层协议，不将其分数直接当作官方基准分数。

## 从冻结权重导出 embedding

`phenostate-embed` 与 `python -m phenostate.infer` 使用相同入口。输入是已有 checkpoint 和训练时的完整 MDS 目录；`--split` 选择原有 train、val 或 test，不重新划分，也不更新权重。同一模型的各 split 使用同一份冻结 checkpoint。

导出现在通过 `LocalDataset` 直接读取已有 MDS 分片，使用有界线程预取；`--num-workers 4` 表示 4 个读取／解码线程，`0` 表示同步读取。它不启动 DataLoader 子进程，不依赖 Streaming 的 `/tmp` 锁或多进程 IPC 临时目录，避开 NFS `.nfs*` 清理异常。训练继续使用原 Streaming 加载器。解码、归一化、模型前向及 Parquet 向量格式保持一致；输出行顺序可能变化，下游按 well_id 对齐。

从项目根目录运行四模型导出脚本：

~~~bash
export RXRX3_MDS_ROOT="/path/rxrx3_mds"
export PHENOSTATE_COMPARISON_ROOT="/path/runs/comparison_patch8_50epochs"
export PHENOSTATE_EMBEDDING_ROOT="/path/embeddings"
export PHENOSTATE_MODELS="mae_patch8_head8 ca_mae diffmae cnn_mae"
bash scripts/export_embeddings.sh
~~~

模型列表对应训练目录名称，默认 `mae ca_mae diffmae cnn_mae`；CNN 没有权重时移出列表。脚本先检查全部模型的权重，每个模型优先使用 best_reconstruction.pt，缺少时回退 latest.pt。相对路径以项目根目录为基准。MDS 和输出目录必须显式设置；训练总目录默认 runs/comparison_patch8_50epochs。`PHENOSTATE_SPLITS` 默认 `train val test`，可改为 `train val`；`PHENOSTATE_BATCH_SIZE` 默认 1，`PHENOSTATE_NUM_WORKERS` 默认 4，`PHENOSTATE_PYTHON` 可指定 Python 可执行文件。

脚本启用 `--skip-existing`：已有正式 Parquet 仅在 manifest SHA、checkpoint SHA、split、pooling、固定长度 float32 schema、非空唯一 well_id、有限值及完整数量均通过检查后复用。原生成功文件无需新的 loader 标签；来源不完整或不同的文件会报错，不覆盖。复用检查信任原生来源标签，不重新解码全部 MDS 孔位来逐项核对 ID，也不加载模型权重检查向量宽度。

残留 `.incomplete` 仍会阻止导出。需确认旧进程退出后，人工保留或处理失败文件；脚本不删除 NFS 残留或临时目录。把本地改动同步到服务器后再启动新进程，旧进程仍运行旧代码。已完成的 embedding 可以保留并校验复用。

这项修复限于 embedding 导出。即使 `TMPDIR` 位于 NFS，新的读取链路也不使用上述进程临时文件；训练及重建评价仍沿用原加载器。
