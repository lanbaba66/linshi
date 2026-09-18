# Data Visualization Lab

一个用于数据可视化、交互式数据应用和简单机器学习演示的 GitHub 工作区。

## 当前基础栈

- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- scikit-learn

`time` 等 Python 标准库不需要单独安装。

## 使用方式

后续代码默认在 Docker 环境中运行。新增第三方库时，请同时更新：

1. `requirements.txt`
2. `docs/library-guide.md`
3. Docker 镜像
4. `examples/import_smoke_test.py`

## 目录

```text
skills/data-visualization/   数据可视化 Skill
docs/                        环境和库说明
examples/                    测试与示例代码
.github/workflows/           GitHub Actions
Dockerfile                   运行环境
requirements.txt             Python 依赖
```

## Docker

构建：

```bash
docker build -t data-viz .
```

运行导入测试：

```bash
docker run --rm data-viz python examples/import_smoke_test.py
```

运行 Streamlit 应用时可使用：

```bash
docker run --rm -p 8501:8501 data-viz streamlit run app.py --server.address=0.0.0.0
```
