# RAGAnything 数据流 Notebook

从两页[图书借阅说明](data/library-guide.pdf)出发，用 RAGAnything 和 LightRAG 的官方接口观察解析结果、图片与表格描述、文本块、向量、知识图谱、检索证据、提示词和回答。示例问题是“学生每次最多能借几本书、能借多久、怎么续借？” 教学步骤见 [rag_anything_walkthrough.ipynb](src/rag_anything_walkthrough.ipynb)。

## 启动

在项目根目录运行以下命令。项目使用 `uv` 和 Python 3.14。

```sh
uv sync --locked
cp -n src/.env.example src/.env
```

在 `src/.env` 中填入 `OPENAI_API_KEY`，然后启动 JupyterLab。

```sh
uv run --env-file src/.env jupyter lab
```

打开 `src/rag_anything_walkthrough.ipynb`，选择 Python 3 内核，按顺序运行。模型、API 地址和向量维度从 `src/.env` 读取。入库和提问会向云端发送示例内容并产生 API 用量。

API Key 留空时可以运行到 PDF 解析单元。Docling 在本地解析 PDF，首次运行可能下载版面和表格识别所需的模型文件。文本生成、图片理解和嵌入使用云端 API。参见 [Docling 模型文件说明](https://docling-project.github.io/docling/usage/advanced_options/)和 [RAGAnything 官方文档](https://github.com/HKUDS/RAG-Anything)。

## 产物和重跑

- `outputs/parsed/` 保存 Docling 导出的文件和提取的图片。
- `outputs/library_rag_storage/` 保存图书借阅示例的文档状态、文本块、向量、知识图谱和缓存。

结束时运行 `finalize_storages()` 单元。重新启动内核后从头运行，可复用同一目录中的索引和缓存。缓存命中时，本次的 `calls` 列表中可能没有相应模型调用。要观察全新的入库过程，或者更换嵌入模型或维度，在配置单元给 `storage_dir` 指定一个新的目录。
