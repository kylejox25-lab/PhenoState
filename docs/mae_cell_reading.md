# MAE microscopy 论文精读：归一化、下游任务与研究设计补充

阅读日期：2026-10-29。论文为 Kraus 等，*Masked Autoencoders for Microscopy are Scalable Learners of Cellular Biology*，CVPR 2024，arXiv:2404.10242v1。本地原文：`D:\XKJ_Proj\cell_painting\papers\mae_cell.pdf`，共 17 页。下文页码均为 PDF 文件页码；正文、补充材料和作者公开代码分别标明。公开代码核对时间为本次阅读日期，当前代码不能保证与论文全部历史训练配置逐项一致。

本补充用于完善研究思路，不表示已执行新训练或新增评测。当前正式范围仍以 [representation_evaluation.md](representation_evaluation.md) 的 T1/T2 为准：三个已有模型、只用图像和自带元数据、固定 train gallery 与 val/test query。外部数据库、JUMP-CP 和 CellProfiler 目标回归均为论文方法说明或未来可选扩展。

## 1. 先明确论文实际证明了什么

论文的主线是：掩码重建预训练 → 冻结编码器提取图像表征 → 聚合、批次校正 → 用外部生物学关系验证表征。它比较普通 MAE、通道无关 CA-MAE、掩码 U-Net 和弱监督分类模型，并研究模型与数据规模的影响。

- 普通 MAE 的六通道空间 patch 共同形成一个 token；CA-MAE 的每个通道分别形成 patch tokens，使用共享单通道投影、重复的固定二维正余弦位置编码，不加入通道身份编码；对总计 C×N 个 tokens 随机掩码，训练时采用独立通道解码器（第 3–5 页，§4.2、§4.2.2、Fig. 4）。
- MAE 使用编码器最终 patch tokens 的平均值作为 crop embedding。训练后提取表征不需要解码器，不再进行随机掩码。CA-MAE 也讨论了 CLS、全体 token 平均、逐通道平均后拼接三种输出；最后一种维度和排序依赖通道数及通道顺序（第 3–5 页）。
- 标准 MAE 探索 16×16 patch/25% mask 和 8×8 patch/75% mask；CA-MAE 报告 ViT-B/16 的 75% 和 ViT-L/16 的 85% mask。不能认为全文所有模型都是 75% 掩码（第 3–4 页）。
- 最佳普通 MAE 与 CA-MAE 的配置、训练数据和计算量不同，不能从 Table 3 直接得出“CA-MAE 在所有任务优于 MAE”或“某一模块单独造成全部增益”。
- JUMP-CP 上 CA-MAE 的扰动检测很好，但相关扰动的 siblings retrieval 弱于 CellProfiler；论文并未证明所有生物学任务都同步提升（第 7–8 页，Table 4）。

## 2. 归一化：必须分四个位置讨论

### 2.1 输入图像：明确采用逐裁剪、逐通道自标准化

正文 §4 与附录 A.3 明确说明，送入模型前对每个图像 crop 做 channel-wise self-standardization（第 2、14 页）。作者公开 `mae_modules.py` 的 `SelfStandardize` 及 `huggingface_mae.py` 的输入链进一步明确实现：

1. uint8 转浮点并除以 255；
2. `InstanceNorm2d` / `LazyInstanceNorm2d`，`affine=False`，`track_running_stats=False`。

对单个 crop、单个通道，可写成：

\[
u_{c,h,w}=x_{c,h,w}/255,\qquad
\widetilde u_{c,h,w}=\frac{u_{c,h,w}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}.
\]

均值与方差在该 crop 的 H×W 空间位置上计算，不跨 batch，也不把六通道混在一起。推理时同样使用当前输入的统计量；它不依赖训练集运行均值。除以 255 只是范围缩放，单独做这一步不能称为已复现论文的自标准化。

此处理有助于抑制整体亮度、染色和成像条件差异，但会减弱每通道整体均值/尺度包含的信息；其收益不能直接外推到本项目。论文本身仍能回归部分强度类 CP 特征，也不意味着预处理保留了每张原图的绝对亮度。

### 2.2 网络内部：LayerNorm 与 QK-normalization

附录 A.2.3 明确列出 MAE-ViT 使用 stochastic depth、LayerScale、flash attention、parallel scaling blocks、QK-normalization，以及 no QK-bias（第 13–14 页）。公开 `vit.py` 对编码器设置 `block_fn=vit.ParallelScalingBlock`、`qk_norm=True`、`qkv_bias=False`。公开解码器模块也采用 LayerNorm。

因此，“架构有没有归一化”的回答是有，而且不能只写一句“沿用 Transformer 的 LayerNorm”。除 token 特征上的 LayerNorm，还在 Q、K 投影之后、注意力点积之前做归一化。可用下式表示其作用位置：

\[
Q_n=\operatorname{LN}(Q),\quad K_n=\operatorname{LN}(K),\quad
\operatorname{Attn}=\operatorname{softmax}\!\left(Q_n K_n^\top/\sqrt{d_h}\right)V.
\]

核对的 timm v0.9.12 `ParallelScalingBlock` 使用每个 attention head 的 `head_dim` 做 Q/K LayerNorm。该版本用于解释代码机制，不代表已确认论文训练所用的精确 timm 版本；作者当前依赖文件未固定 timm 版本。复现时应自行固定依赖、归一化维度、epsilon 和 affine 设置。

LayerScale 是残差分支的可学习缩放，flash attention 是高效注意力实现，两者不能当作图像标准化或批次校正。已有输入 LayerNorm 也不能替代投影后的 QK-Norm。

AdaBN / Ghost BatchNorm 出现在 DenseNet 弱监督对照模型中（第 3、13 页），不能据此说论文的 MAE-ViT 使用 AdaBN。

### 2.3 重建目标：不能把本项目的 patch 标准化说成论文方案

论文 Eq. (1) 是掩码 patch 的像素 MSE。公开 `huggingface_mae.py` 的 `compute_MAE_loss` 先对原图执行相同 `input_norm`，然后切 patch，与预测计算 MSE；所核对实现没有再对每个 patch 单独做均值/方差标准化。

所以要区分：

| 位置 | 论文 / 作者公开代码 | 本项目当前实现 |
| --- | --- | --- |
| 编码器输入 | crop 逐通道自标准化 | JP2 解码后除以 255，保留整体强度 |
| 像素重建目标 | 对经过输入自标准化的图像切 patch | `norm_pix_loss=true` 时，每个 patch 内每个通道单独标准化 |
| 编码器内部 | LayerNorm、QK-Norm、parallel scaling 等 | 共享 TransformerBlock 有 pre-LayerNorm，尚无投影后的 QK-Norm |
| 下游校正 | TVN，关系评估另作染色体臂偏差校正 | 正式 CLI 提供 raw 与 train-only standardize；尚未实现 TVN |

本项目定位依据：`src/phenostate/data/decode.py`、`src/phenostate/models/mae.py`、`src/phenostate/models/CA-MAE.py`、`src/phenostate/models/DiffMAE.py`、`src/phenostate/retrieval.py`。模型的 `reconstruction_target` 归一化作用于损失目标，不会自动改变 encoder 看到的图像。

论文的“+”是额外的 Fourier 幅度重建损失，不是一种归一化。它将像素 MSE 与频域 L1 组合：`L=(1−α)L_pixel+αL_FT`，论文采用 α=0.01，两个分量按论文公式在隐藏 patch 上计算（第 4 页，Eq. 1–3）。若未来实现，需明确 FFT 的空间轴、patch/通道布局、FFT 缩放和隐藏位置处理；当前项目不同目标空间上的 loss 数值不能与论文直接比较。

### 2.4 表征后处理：TVN 是主要实验组成部分

正文 §5.1 在特征空间使用 Typical Variation Normalization（TVN）减轻批次差异，并以阴性对照均值为原点，还校正 CRISPR 染色体臂偏差。TVN 的作用是利用对照中的典型变异改善特征空间，不等同于单个向量的 L2 归一化或简单逐维 z-score。

Table 2（第 5 页）对同一 RPI-93M MAE ViT-L/8+ 展示：

| 后处理 | CORUM | hu.MAP | Reactome | StringDB |
| --- | ---: | ---: | ---: | ---: |
| 无变换 | 0.124 | 0.124 | 0.096 | 0.135 |
| PCA | 0.126 | 0.122 | 0.102 | 0.134 |
| 按 plate 中心化 | 0.449 | 0.361 | 0.184 | 0.350 |
| 按 experiment 标准化 | 0.460 | 0.370 | 0.188 | 0.359 |
| PCA + 按 plate 标准化 | 0.614 | 0.435 | 0.261 | 0.477 |
| TVN | 0.622 | 0.443 | 0.267 | 0.484 |

这说明该实验的生物学关系召回明显依赖特征校正。不能把完整管线的得分全部归因于编码器，也不能把这些数字解释为基因分类准确率。本文引用的正文与作者模型仓库并未完整展开 TVN 的全部拟合边界、协方差正则化和染色体臂校正步骤；完整复现仍需核对引用方法和对应实现版本，不能自行补一个 z-score 就命名为 TVN。

## 3. 下游任务到底怎么做

### 3.1 共同入口：冻结编码器与分层聚合

论文预训练随机抽取 256×256×6 crop 并同步随机水平/垂直翻转。验证使用训练未见实验的中心裁剪图；附录 A.1 按 experiment 隔离训练/验证（第 13–14 页）。

RxRx3 下游推理的一个 well 是 2048×2048×6：平铺成 64 个互不重叠的 256×256 crop，各自标准化并编码，再对 64 个 crop embeddings 做算术平均，得到 well embedding。之后才进行下游的特征变换与扰动聚合（附录 A.3）。

须区分两层平均：`patch tokens → crop embedding` 与 `64 crops → well embedding`。扰动重复之间的 spherical mean 又是后续的另一层聚合。论文方法不能直接套用为“本地每孔也有 64 张图”：本项目每孔只有数据集提供的一张 512×512 中心裁剪图。

当前主实验继续完整 512×512 输入，得到一个 well embedding，保留现有公平比较。如果未来比较 4 个 256 crop 的平均，必须作为独立消融，所有方法使用同一聚合规则，并报告训练/推理尺度改变与计算成本；这不恢复原始 2048 视野。

### 3.2 下游 A：基因—基因已知生物学关系召回

这是论文最核心的下游任务（§5.1、Table 2/3）。流程为：

1. 在 RxRx3 CRISPR wells 上提取 well embeddings。
2. 进行 TVN、阴性对照中心化和染色体臂偏差校正。
3. 合并同一扰动的重复表征，使用 spherical mean 形成扰动/基因表征。可将其理解为对方向向量求均值再归一化；完整复现仍要固定实际实现的聚合顺序与权重。
4. 计算不同基因表征之间的 cosine similarity。
5. 用 CORUM、hu.MAP、Reactome、StringDB 的独立关系真值计分，StringDB 使用 combined score >95% 的关系。

主要指标是在所有候选基因对相似度的最高 5% 和最低 5% 两个尾部中，找回多少已知关系：

\[
\operatorname{Recall}_{tails}=\frac{|R_{known}\cap R_{tails}|}{|R_{known}|}.
\]

分母是候选基因全集中可评估的已知关系。因为选取两端合计约 10% 的候选对，随机基线约为 0.10。负向相似也被纳入该指标的关系发现范围，但不能据此把负余弦解释成确定的拮抗机制。Table 5 的“5% false positive rate”是另一个操作点，不等同于这里的双尾百分位，更不等同于 Recall@5。

它验证的是不同基因之间的功能联系，而本项目 T1 验证同一个基因能否跨 guide、跨实验匹配。两者互补但不是同一项任务；gene 身份标签不能构造不同基因间的功能真值。

### 3.3 下游 B：JUMP-CP 的扰动检测与 siblings retrieval

JUMP-CP 未包含在论文预训练数据中，属于外部迁移验证（§5.3、Table 4、附录 A.7）。编码器冻结，比较 cosine 检索表现：

- 扰动检测/检索：相对阴性对照，判断同一扰动的表型是否可以稳定找回。
- siblings retrieval：检索相关但不相同的扰动，例如相关化合物或相关 CRISPR 处理；正例定义应跟随该基准的官方元数据，不能仅靠图像相似度反过来定义。
- 对 AP 做 permutation test，进行 FDR 校正，以 q=0.05 为阈值；Table 4 报告通过显著性标准的扰动比例，`fraction retrieved`，并非直接报告平均 AP 或分类准确率。

固定六通道模型需要匹配染色并复制一个通道来接收 JUMP 的五通道 Cell Painting 输入。CA-MAE 能接收五个 Cell Painting 加三个 brightfield 共八个通道，不需要把输入硬改成六通道。

Table 4 中 CA-MAE 的扰动检索为 0.95±0.05，而 CellProfiler 为 0.53±0.30；siblings 则为 0.02±0.02 对 0.13±0.07。这里的 ± 是跨细胞类型、模态、时间点的标准差，不能解释成训练 seed 置信区间。结论是任务依赖的：稳定识别扰动，不自动意味着能识别相关扰动。作者提出未来需要研究微调或对齐，并未在本工作中完成一个统一的下游微调方案。

### 3.4 下游 C：冻结表征预测 CellProfiler 形态特征

正文 §5.5 与附录 A.6（第 8、14、16 页）给出一个真正训练下游预测器的例子：冻结图像编码器，以 well embedding 为自变量，分别拟合 955 个 CP 特征。类别为 AreaShape、Intensity、Neighbors、RadialDistribution、Texture。

- 训练：一个内部 experiment，12 plates×1380 wells=16,560 wells。
- 测试：另一个同等大小的独立 experiment；共有 121 个对照扰动，试剂重叠不足 10%。
- CP 特征先按偏度处理：skew>0.5 时 log 变换，skew<−0.5 时平方，再标准化；embeddings 同样进行中心化和单位方差缩放。
- 每个 CP 目标独立拟合 `ElasticNetCV`；5 折交叉验证，L1 ratio 搜索 0.1、0.6、0.9、0.95、0.99，alpha 自动选择。
- 独立测试集报告 R²，并按形态类别汇总中位数及 MAD。Intensity 类的 median R²：MAE 0.737，WSL 0.297（Table 7）。

这是冻结编码器的线性探测，不是端到端微调，也不是对每个细胞做分割/分类。若未来在本项目做此任务，应使用本地实际可用的 CP 字段与共同 well 交集，而不是硬编码论文的 955 维；现有项目记录本地 CP 为 952 维且覆盖不足全部 wells。本地同一图像来源的 CP 目标只说明形态信息可解码，不应替代独立的生物学关系真值。当前阶段不新增该任务。

## 4. 对当前研究思路的具体完善

### 4.1 将研究问题从“重建更好”改写成可检验链条

建议表述：在 RxRx3-core 域内，比较不同掩码预训练方法能否得到同时具有跨 guide 生物一致性、跨板重复稳定性和批次稳健性的单孔表征，并通过受控消融区分通道建模、归一化与后处理的作用。

重建 MSE 是优化与预训练任务指标；正式主终点继续用 val T1 MRR，T2 mAP 作辅助。三个已有模型是完整方法比较，不能直接解释成通道掩码或扩散目标的单因素效果。`checkpoints/best.ckpt` 名字只说明在其自身目标下重建最佳，不说明表征最优。若已有多个 epoch 的保存点，可在 val 上做表征检查点选择，再锁定 test；所有导出必须来自选定的同一个 checkpoint。

### 4.2 先用已有表征完成主任务，补齐后处理敏感性

最低可执行流程：

1. 为 MAE、CA-MAE、DiffMAE 用同一份各自冻结 checkpoint 导出 train、val、test well embeddings，核对完整覆盖与身份。
2. 在三个模型共同孔位上运行 T1/T2 的 `raw + cosine`；再运行 `train-only standardize + cosine`。这两种已由正式 CLI 支持。
3. val 冻结后处理和模型选择规则；test 只做最终计分。主表给出 T1 MRR/Hit、T2 mAP/Recall、覆盖率、基因配对 bootstrap 和训练成本。
4. 保留对照类型一致的同板、跨板、跨实验批次诊断；诊断不替代生物学任务。

论文提示未来可增加阴性对照中心化与 TVN，但现有 CLI 未实现，不能写成已有结果。若开展，需要预先区分：

| 拟合协议 | 可使用的统计量 | 应如何解释 |
| --- | --- | --- |
| 当前严格 train-only | 训练参考 wells/合格训练对照的统计量，冻结后作用于 query | 不使用新板统计量；适合现有主表 |
| 新板对照适配 | 使用 query 所在板的匹配阴性对照估计板偏移/变异 | 使用新批次参考的适配协议；单列结果与对照样本需求 |

第二种不是编码器更新，但确实利用了新批次观测。不能混用 query 处理孔拟合后处理再称为严格 train-only，也不能把不同 control 类型一并当作 DMSO。拟合 PCA、白化或协方差时需要训练范围、样本量、正则化、降维维数与失败回退规则；在本项目定义并验证前，不称为论文 TVN 的复现。

### 4.3 将归一化安排成单因素消融，避免连带改变

优先级仍为：现有 T1/T2+E1 → CA 通道身份×掩码方式 A1 → 重建目标标准化 A2。下面是论文阅读后新增的候选，均需重训，不自动替换已确认的核心消融。

| 候选实验 | 固定什么 | 只改变什么 | 回答的问题 |
| --- | --- | --- | --- |
| QK-Norm | 先在普通 MAE 内固定输入、目标、串行 block、QKV bias、预算 | 投影后 Q/K 的归一化开/关 | 除 pre-LayerNorm 外，QK-Norm 对稳定性和 T1 是否有益 |
| 输入自标准化 | 固定模型与 patch 目标标准化开关 | /255 与 /255+逐图逐通道 InstanceNorm | 抑制整体强度差异是否利于跨实验检索 |
| 目标标准化 A2 | 固定 /255 输入与模型 | `norm_pix_loss=true/false` | 逐 patch 目标标准化是否有益 |
| Fourier | 固定输入、目标、模型与优化设置 | α=0 与预先指定的 α=0.01 | 纹理重建约束是否带来表征收益 |

输入自标准化实验中，若重建目标由预处理后的图像生成，应保持这一数据流不变，只切换输入预处理；明确目标也随输入空间变化，不能称为“只改变 encoder 输入、目标数值完全相同”。想分别归因输入与目标时，应再做两者的交叉设计，预算不足就不作单独因果宣称。

QK-Norm 的首轮实验不要同时更换 parallel block、LayerScale、Lion 或移除 QKV bias，否则无法归因。若最终选择升级共同编码器，应给所有方法相同升级，并重新训练公平对照。论文的巨型 batch、Lion 和大模型稳定化经验可作候选依据，不直接要求单张 3090 照搬。

### 4.4 CA-MAE 的价值要单独验证

除 T1/T2 外，未来可验证完整六通道与预先固定的缺失通道组合；同一候选库/查询使用一致的输入组合并另报覆盖。全 token 均值在共享投影、无通道 ID、重复空间位置编码下应对通道置换近似不变，可作为实现自检；逐通道拼接不具备同一性质。通道 ID 消融若提高完整六通道表现，应同时说明其对任意通道顺序/新染色协议泛化的限制。

当前 512×512、patch=8：普通 MAE 完整 4096 tokens、75% mask 后 1024 可见；CA-MAE 完整 24576、约 6144 可见。相同图像数、epochs 不代表相同 FLOPs。论文 CA-MAE 的主要价值还包括可变通道迁移，不能只用六通道重建误差判定其设计是否有效。

### 4.5 证据分层与结果解释

正式表：三个已有模型 × 已冻结后处理 × T1/T2，附覆盖率、训练预算和配对置信区间。后续消融另表，避免把新归一化模型与旧结构模型混排为单因素结果。

- T2 好、T1 差：技术重复可识别，但跨 guide/实验的共同基因表型尚弱。
- 后处理明显提升 T1：说明特征可用性受批次几何影响；须同时报告 raw，不能把增益全部归给编码器。
- 重建更好、T1 不变/变差：预训练目标的改进没有转化为当前任务收益。
- 将来 CP R² 高但功能关系召回弱：形态细节可解码，与高层生物学关系可检索仍有区别。
- 单次训练 seed 的基因 bootstrap 不能代表训练随机性；最终核心消融仍应补配对 seeds。

## 5. 一句话研究路线

冻结已有三个编码器，先完成统一孔位的跨实验/跨 guide 检索、跨板重复检索与训练集拟合的后处理比较；再在同一架构内分别检验通道机制、QK-Norm 和输入/目标标准化。外部功能关系与跨数据集迁移留作独立扩展，每项主张都对应一项可计算的下游证据。

## 6. 核对来源

- 本地完整 PDF：第 2–5 页（方法、输入处理、CA-MAE、Table 2）、第 6–8 页（关系召回、JUMP、CP 回归）、第 13–17 页（划分、超参数、推理、回归协议与补充结果）。
- [论文主页](https://arxiv.org/abs/2404.10242)。
- [作者仓库与编码器设置](https://github.com/recursionpharma/maes_microscopy)。
- [输入 SelfStandardize 与编码/解码模块](https://github.com/recursionpharma/maes_microscopy/blob/trunk/mae_modules.py)。
- [公开输入链、目标构造和推理聚合](https://github.com/recursionpharma/maes_microscopy/blob/trunk/huggingface_mae.py)。
- [公开 ViT 配置](https://github.com/recursionpharma/maes_microscopy/blob/trunk/vit.py)。
- [用于解释 QK-Norm 的 timm v0.9.12 ParallelScalingBlock](https://github.com/huggingface/pytorch-image-models/blob/v0.9.12/timm/models/vision_transformer.py)。
- 当前正式协议与现有代码：见本笔记 §2.3 及 [representation_evaluation.md](representation_evaluation.md)。
