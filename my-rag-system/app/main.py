"""FastAPI 主入口：提供 REST API 和前端页面"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 加载环境变量（从 .env 文件）
load_dotenv()

from app.rag_engine import RagEngine

# ---- 初始化 ----
app = FastAPI(title="RAG 知识库问答系统")

# 目录配置
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
DOCS_DIR = BASE_DIR / "data" / "docs"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# 创建必要的目录
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# 初始化 RAG 引擎（启动时加载，常驻内存）
engine = RagEngine(index_path=str(CHROMA_DIR / "faiss.index"))


# ---- 数据模型 ----
class AskRequest(BaseModel):
    """提问请求体"""
    question: str


# ---- 接口 1：首页 ----
@app.get("/")
def index():
    """返回前端页面"""
    return FileResponse(STATIC_DIR / "index.html")


# ---- 接口 2：状态查询 ----
@app.get("/api/status")
def get_status():
    """获取知识库状态"""
    return engine.status()


# ---- 接口 3：上传文档 ----
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并入库"""
    if not file.filename:
        raise HTTPException(400, "没有文件名")
    
    # 检查文件类型
    suffix = Path(file.filename).suffix.lower()
    allowed = (".txt", ".md", ".pdf", ".docx")
    if suffix not in allowed:
        raise HTTPException(400, f"只支持 {', '.join(allowed)}，收到: {suffix}")
    
    # 保存文件到 docs 目录
    save_path = DOCS_DIR / file.filename
    save_path.write_bytes(await file.read())
    
    # 调用 RAG 引擎入库
    try:
        chunks = engine.ingest_file(save_path)
        return {"filename": file.filename, "chunks": chunks}
    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整错误堆栈到终端
        raise HTTPException(400, f"文档解析失败: {str(e)}")


# ---- 接口 4：问答 ----
@app.post("/api/ask")
def ask(request: AskRequest):
    """提问，返回基于知识库的答案"""
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")
    
    return engine.answer(question)


# ---- 接口 5：重置 ----
@app.post("/api/reset")
def reset():
    """清空知识库"""
    engine.reset()
    return {"ok": True}


# ---- 挂载静态文件 ----
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---- 启动入口（可直接运行） ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)