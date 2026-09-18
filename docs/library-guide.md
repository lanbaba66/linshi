# Library Guide

本项目当前默认使用以下数据可视化与分析库。

## Streamlit

用途：快速构建交互式数据应用、筛选器、指标卡、侧边栏和页面布局。

常用导入：

```python
import streamlit as st
```

适合：数据看板、交互演示、模型结果展示。

## Pandas

用途：表格数据读取、清洗、过滤、聚合和统计。

```python
import pandas as pd
```

适合：CSV/Excel/DataFrame 数据处理。

## NumPy

用途：数值数组、随机数据、向量化计算。

```python
import numpy as np
```

## Matplotlib

用途：静态图、精确排版、科研风格图形。

```python
import matplotlib.pyplot as plt
```

Docker 中默认设置 `MPLBACKEND=Agg`，便于无显示器环境生成图片。

## Plotly

用途：交互式图表、悬浮提示、缩放、3D 图和 Web 友好可视化。

```python
import plotly.express as px
import plotly.graph_objects as go
```

## scikit-learn

用途：回归、分类、误差指标、预处理等机器学习演示。

截图中已出现：

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
```

## 标准库

例如：

```python
import time
```

这类模块由 Python 自带，不写进 `requirements.txt`。

## 新增库规则

以后如果新增第三方库，必须同时：

1. 更新 `requirements.txt`
2. 在本文件写用途和导入方式
3. 更新 `examples/import_smoke_test.py`
4. 重新验证 Docker 构建

不要在业务代码里临时 `pip install`，优先把依赖固化进镜像。
