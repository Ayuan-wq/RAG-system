
RAG 知识库问答系统

> 上传文档，AI 只依据资料回答，不瞎编

基于 fastAPI + FAISS + DeepSeek构建的 RAG（检索增强生成）知识库问答系统。

---

✨ 核心特性

| 特性 | 说明 |
| :--- | :--- |
| 📄 多格式支持 | 支持 .txt / .md / .pdf / .docx |
| 🔍 语义检索 | 基于 BGE 向量模型，理解语义而非关键词 |
| 🎯 答案可溯源 | 返回答案的同时标注来源段落和相关度 |
| 🚫 杜绝幻觉 | 资料外问题会回答「无法回答」，不瞎编 |
| ⚡ 轻量部署 | 基于 FAISS，无需 C++ 编译器，开箱即用 |

---

🛠️ 技术栈

| 组件 | 选型 | 说明 |
| :--- | :--- | :--- |
| 后端框架 | FastAPI | 轻量异步 Web 框架 |
| 向量库 | FAISS | 高性能向量检索，纯 Python 实现 |
| Embedding | BAAI/bge-small-zh-v1.5 | 本地中文向量模型，数据不出本地 |
| 大模型 | DeepSeek API | 生成最终答案 |
| 前端 | 原生 HTML + CSS + JS | 无需构建工具 |

---

🚀 快速开始

```bash
git clone https://github.com/Ayuan-wq/RAG-system.git
cd RAG-system/my-rag-system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

📊 工作流程

```
用户上传文档 → 文档切分 → 向量化 → 存入 FAISS
用户提问 → 向量检索 → 召回相关段落 → DeepSeek 生成 → 返回答案 + 来源
```
