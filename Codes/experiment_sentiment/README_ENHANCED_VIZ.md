# 🎨 Enhanced Visualization System

## 概述

增强的可视化系统为伪文明检测实验提供了美观、现代化的图表和自动版本化的结果管理。该系统包含：

- 🎨 **现代化图表设计** - 使用现代配色方案和样式
- 📊 **丰富的可视化类型** - 混淆矩阵、ROC曲线、PR曲线、嵌入可视化等
- 🌐 **交互式仪表板** - 基于Plotly的交互式图表
- 📄 **综合HTML报告** - 美观的实验报告
- 📁 **自动版本管理** - 每次实验自动创建带时间戳的结果文件夹

## 🚀 快速开始

### 1. 运行演示

```bash
cd Codes/experiment_sentiment
python demo_enhanced_viz.py
```

这将创建一个完整的演示，展示所有可视化功能。

### 2. 在实验中使用增强可视化

```bash
# 基础增强可视化
python enhanced_main.py --quick-test --enhanced-viz

# 包含嵌入可视化
python enhanced_main.py --quick-test --enhanced-viz --embedding-viz

# 包含交互式仪表板
python enhanced_main.py --quick-test --enhanced-viz --interactive-dashboard

# 完整功能
python enhanced_main.py --quick-test --enhanced-viz --embedding-viz --interactive-dashboard
```

## 📁 结果目录结构

每次运行实验时，系统会自动创建带时间戳的结果目录：

```
results/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── charts/                    # 静态图表
    │   ├── enhanced_confusion_matrix.png
    │   ├── enhanced_roc_curve.png
    │   ├── enhanced_pr_curve.png
    │   ├── metrics_comparison.png
    │   └── embedding_visualization.png
    ├── interactive/               # 交互式内容
    │   └── dashboard.html
    ├── reports/                   # HTML报告
    │   └── comprehensive_report.html
    ├── models/                    # 保存的模型（可选）
    ├── embeddings/                # 保存的嵌入（可选）
    ├── config.json               # 实验配置
    ├── experiment_summary.json    # 实验摘要
    └── experiment.log            # 实验日志
```

## 🎨 可视化功能

### 1. 增强混淆矩阵
- 🎨 现代化热力图设计
- 📊 百分比注释
- 🌈 优雅的配色方案
- 📱 高分辨率输出

### 2. 增强ROC曲线
- 📈 渐变填充效果
- 🎯 AUC分数显示
- 📊 网格背景
- 🔍 清晰的轴标签

### 3. 增强PR曲线
- 📉 精确率-召回率曲线
- 🎨 渐变填充
- 📊 AUC分数
- 🔍 详细的轴标签

### 4. 模型性能对比
- 📊 多模型对比柱状图
- 🎨 不同颜色区分
- 📈 数值标签
- 📱 响应式设计

### 5. 嵌入可视化
- 🔍 t-SNE/PCA降维可视化
- 🎨 类别颜色区分
- 📊 清晰的图例
- 🌐 高质量输出

### 6. 交互式仪表板
- 🌐 基于Plotly的交互图表
- 📊 多子图布局
- 🔍 缩放和悬停功能
- 📱 响应式设计

## 📄 HTML报告特性

### 现代化设计
- 🎨 优雅的CSS样式
- 📱 响应式布局
- 🌈 现代配色方案
- 📊 卡片式设计

### 内容组织
- 📊 性能指标卡片
- 📈 图表展示
- 🔍 错误分析
- 💾 实验信息

### 用户体验
- 🎯 清晰的导航
- 📱 移动端友好
- 🔍 易于阅读
- 📊 直观的数据展示

## 🔧 配置选项

### 命令行参数

```bash
# 可视化相关
--enhanced-viz          # 启用增强可视化（默认开启）
--interactive-dashboard  # 创建交互式仪表板
--embedding-viz         # 创建嵌入可视化

# 实验配置
--quick-test           # 快速测试模式
--experiment-name      # 自定义实验名称
--output-dir           # 自定义输出目录
```

### 配置文件设置

在配置文件中可以设置：

```python
# 可视化配置
evaluation:
  save_predictions: True      # 保存预测结果
  save_probabilities: True    # 保存概率
  feature_importance: True    # 特征重要性
  error_analysis: True        # 错误分析

# 输出配置
save_embeddings: True         # 保存嵌入
save_models: True            # 保存模型
save_results: True           # 保存结果
```

## 📊 依赖项

### 核心依赖
- `matplotlib` - 静态图表
- `seaborn` - 统计可视化
- `plotly` - 交互式图表
- `numpy` - 数值计算
- `pandas` - 数据处理

### 可选依赖
- `scikit-learn` - 机器学习指标
- `umap-learn` - UMAP降维（可选）

## 🎯 使用示例

### 示例1：基础实验
```bash
python enhanced_main.py \
  --config-type binary \
  --datasets chnsenticorp \
  --classifier logistic_regression \
  --enhanced-viz
```

### 示例2：完整功能实验
```bash
python enhanced_main.py \
  --config-type ensemble \
  --datasets chnsenticorp civil_comments \
  --classifier xgboost \
  --enhanced-viz \
  --embedding-viz \
  --interactive-dashboard \
  --experiment-name "full_experiment"
```

### 示例3：快速演示
```bash
python demo_enhanced_viz.py
```

## 🔍 查看结果

### 静态图表
- 查看 `charts/` 目录下的PNG文件
- 高分辨率，适合论文和报告

### 交互式仪表板
- 打开 `interactive/dashboard.html`
- 支持缩放、悬停、筛选等交互功能

### 综合报告
- 打开 `reports/comprehensive_report.html`
- 包含所有实验结果和可视化

### 实验摘要
- 查看 `experiment_summary.json`
- 机器可读的实验数据

## 🎨 自定义样式

### 修改配色方案
在 `enhanced_visualizer.py` 中修改 `_get_color_palette()` 方法：

```python
def _get_color_palette(self) -> Dict[str, str]:
    return {
        'primary': '#2E86AB',      # 主色调
        'secondary': '#A23B72',    # 次要色调
        'accent': '#F18F01',       # 强调色
        # ... 更多颜色
    }
```

### 修改图表样式
在 `setup_style()` 方法中调整matplotlib参数：

```python
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 16,
    # ... 更多样式设置
})
```

## 🚀 扩展功能

### 添加新的可视化类型
1. 在 `EnhancedVisualizer` 类中添加新方法
2. 遵循现有的命名和样式约定
3. 在 `create_comprehensive_report()` 中集成

### 添加新的交互功能
1. 使用Plotly创建新的交互图表
2. 在 `create_interactive_dashboard()` 中添加
3. 更新HTML模板以显示新内容

## 📝 最佳实践

### 实验管理
- 📁 每次实验使用不同的名称
- 📊 保存重要的实验配置
- 🔍 定期清理旧的结果目录

### 可视化优化
- 🎨 保持一致的配色方案
- 📱 确保图表在不同设备上显示良好
- 📊 添加清晰的标签和说明

### 性能考虑
- ⚡ 对于大数据集，限制嵌入可视化的样本数量
- 💾 定期清理不需要的嵌入文件
- 📊 使用适当的图表分辨率

## 🔧 故障排除

### 常见问题

1. **图表不显示**
   - 检查matplotlib后端设置
   - 确保所有依赖项已安装

2. **交互式仪表板无法打开**
   - 检查Plotly是否正确安装
   - 确保浏览器支持JavaScript

3. **样式显示异常**
   - 检查字体是否可用
   - 验证CSS文件路径

### 调试模式
```bash
python enhanced_main.py --quick-test --log-level DEBUG
```

## 📚 相关文档

- [主实验文档](README_MODULAR.md)
- [配置说明](config.py)
- [评估模块](evaluation.py)
- [模型管理](models.py)

---

🎨 **享受美观的可视化体验！**
