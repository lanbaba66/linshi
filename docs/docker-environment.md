# Docker Environment

## 目标

把数据可视化依赖放入可重复使用的 Docker 环境，让 GitHub Actions 运行代码时不依赖本地电脑。

## 基础镜像

```text
python:3.12-slim
```

## 预装 Python 库

- streamlit
- pandas
- numpy
- matplotlib
- plotly
- scikit-learn

## 运行模型

```text
GitHub Actions runner
      ↓
Docker build / pull
      ↓
Python 3.12 + 已声明依赖
      ↓
运行脚本或 Streamlit 应用
      ↓
输出图片、HTML、CSV、日志等 artifact
```

## 注意

- 不使用用户手机或电脑上的 Python 环境。
- Docker 构建以 `requirements.txt` 为依赖真源。
- Matplotlib 采用 `Agg` 后端以支持无 GUI 服务器。
- 后续如果需要浏览器自动化、地图库、视频编码或系统字体，再按需求补充系统依赖。
