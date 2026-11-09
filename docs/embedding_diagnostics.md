# Notebook 03：七模型 embedding 诊断

打开 `notebooks/03_representation_comparison.ipynb`，按顺序执行。使用已有 Parquet embedding、元数据、既有 split 和显式阴性对照规则，不读取图像或训练模型。

固定比较七种表征：`mae`、`ca_mae`、`diffmae`、`cnn_mae`、`OpenPhenom`、`MAE-L-8`、`MAE-G-8`。缺文件会在输入表中显示并停止，不静默改成少于七种。不同模型允许不同特征维数。

## 输入

Notebook 配置单元可直接修改 `MODEL_FILES`、`METADATA_PATH`、`SPLITS_PATH` 和 `CONTROL_RULES`。也可使用下列环境变量：

| 变量 | 用途 |
|---|---|
| `PHENOSTATE_PROJECT_ROOT` | 项目根目录；默认从当前目录向上寻找 |
| `PHENOSTATE_RESULTS_DIR` | 现有 retrieval 结果目录；默认 `results/downstream_val_all` |
| `RXRX3_CORE_ROOT` | 元数据和公开 embedding 所在数据目录 |
| `PHENOSTATE_METADATA_PATH` | 直接指定 `metadata_rxrx3_core.csv` |
| `PHENOSTATE_EMBEDDING_ROOT` | 四个本地模型 embedding 根目录，默认 `../data/embeddings` |
| `PHENOSTATE_PUBLIC_EMBEDDING_ROOT` | 三个公开模型文件目录；默认数据目录 |
| `PHENOSTATE_SPLITS_PATH` | 已有 plate JSON/CSV 或含 `well_id,split` 的清单 |
| `RXRX3_MDS_ROOT` | 复用该目录中的 `plate_splits.json` |
| `PHENOSTATE_CONTROL_RULES` | 阴性对照 JSON；不提供时优先使用正式协议中的规则，否则显示示例规则供核对 |
| `PHENOSTATE_DIAGNOSTIC_INPUTS` | 包含七模型路径列表的 JSON，可覆盖全部模型路径 |
| `PHENOSTATE_DIAGNOSTIC_OUTPUT` | 新诊断输出目录；默认结果目录中的 `embedding_diagnostics_7_models` |

Notebook 会优先复用现有 `protocol.json` 中存在的模型来源文件，迁移目录后可直接改配置或提供模型路径 JSON：

```json
{
  "mae": ["/data/mae/embedding_train.parquet", "/data/mae/embedding_val.parquet"],
  "ca_mae": ["/data/ca_mae/embedding_train.parquet", "/data/ca_mae/embedding_val.parquet"],
  "diffmae": ["/data/diffmae/embedding_train.parquet", "/data/diffmae/embedding_val.parquet"],
  "cnn_mae": ["/data/cnn_mae/embedding_train.parquet", "/data/cnn_mae/embedding_val.parquet"],
  "OpenPhenom": ["/data/OpenPhenom_rxrx3_core_embeddings.parquet"],
  "MAE-L-8": ["/data/MAE-L_8_RPI-93M_rxrx3_core_embeddings.parquet"],
  "MAE-G-8": ["/data/MAE-G_8_PP-16M_trimmed_rxrx3_core_embeddings.parquet"]
}
```

默认仅保留 `train,val`，可改 `SELECTED_SPLITS=('val',)`；不新生成 split。若明确选择 `selected_splits=None`，只能称已提供孔位的探索分析，不能推断留出表现。

## 分析和含义

1. **覆盖**：七模型共同 well_id、共同无效向量剔除、扰动×plate 矩阵。每类默认抽至 200 个扰动，优先跨批次覆盖，每扰动每板至多 3 wells，对照至多 32；抽样由元数据和固定 seed 决定，不按某模型效果选组。
2. **相似度分布**：严格同扰动跨板、合格非重复跨板/同板、同板扰动—对照、同类对照跨板/跨 experiment。药物匹配浓度，CRISPR 非重复排除同基因其他 guide；controls 不混入普通扰动。主图使用跨七模型统一的实际数值范围放大差异，同时保存完整 `[-1,1]` 版本；强度—重复性散点的七面板也统一坐标尺度。
3. **强度—重复性**：横轴 `d(T,C)-d(C,C')`，纵轴重复相似度减跨板非重复相似度。弱扰动接近对照合理，横轴不作为统一模型分数。
4. **PCA/UMAP**：七模型同一套样本；每个模型每子集只计算一次坐标，分别按 experiment、plate、扰动着色。全部、阴性对照、共享扰动子集分别展示；共享扰动在两类 assay 间轮流选择，类内优先批次覆盖。各类还展示一个控制覆盖最多的 experiment 内的 plate 图。`umap-learn` 可选，未安装时仍完整运行 PCA 和所有高维指标。
5. **相似度热图**：小样本、同一矩阵按批次与扰动两种顺序展示，固定 `[-1,1]` 色标并隐藏对角线。
6. **扰动/plate 对比**：B 为同扰动跨板、C 为异扰动同板、D 为异扰动跨板，报告 B-D、B-C、C-D。C/D 使用同 anchor、同参考板、两板共享的非重复身份。无共享身份时竞争和 plate 优势为 NA。
7. **控制近邻富集**：实际同批次近邻比例除以排除自身后的随机比例，参照值 1。plate 比较限制同 experiment；experiment 比较限制同生物 context/控制家族。并列近邻按期望计数，按批次再按匹配子集等权。

孔对先在 anchor/候选批次块内取中位数，再对扰动×context 汇总，避免孔对数量主导宏平均。差值的 95% CI 使用基因/化合物聚类 bootstrap，保留同基因多个 guide、同化合物多个剂量的相关性，不是训练 seed 的置信区间。

这些图是描述性结构诊断，原始向量只做余弦所需单位化，不执行批次校正。PCA/UMAP 在画图子集拟合，只用于探索。图中分群不能证明因果或百分比方差归属；扰动与 experiment 不交叉时不可单独归因。阴性对照的跨批次比较也要求生物条件一致。

## 输出与原评估的关系

诊断输出独立于既有 retrieval 成绩，包括输入/孔对覆盖、排除原因、共同 well 清单、固定 pair plan、组/块级相似度、差值、95% CI、近邻基线、降维坐标、PNG 图和 `diagnostic_protocol.json`。抽样上限、实际数目、输入路径和 seed 均有记录。

附录保留正式 `val-query/train-gallery` 的 mAP、Recall、活性、重复比例和配对差值；无既有结果时跳过附录。包含 train 的探索图与留出成绩分开解释。公开模型的上游预训练重叠未知，不能把七模型差异解释为相同预算的架构消融。
