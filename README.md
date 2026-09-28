# LLM Learning Assistant

后端使用 FastAPI、LangGraph、PostgreSQL 和 Milvus。PostgreSQL 保存用户、会话与图状态；Milvus 保存 LLM 工程书籍的向量。前端仍是 `/login` 和 `/chat` 两个简易页面，书籍管理位于聊天页的“知识库”弹窗。

理解项目的设计顺序、请求流程和各组件职责，可先阅读 [项目一运行原理与设计思路](docs/项目一运行原理与设计思路.md)。

## 启动

1. 启动 PostgreSQL，将 `backend/.env.example` 复制为 `backend/.env`，并配置 `DATABASE_URL`、`JWT_SECRET_KEY`、`GuiJiLiuDong_API_KEY` 和 `GuiJiLiuDong_BASE_URL`。`DATABASE_URL` 使用 `postgresql+asyncpg://...` 格式。不要提交 `.env`。已有 `.env` 的项目只需补充示例中的 Milvus/Embedding 配置，默认值也可直接使用。
2. 安装 Docker Desktop，在项目根目录启动 Milvus Standalone：

   ```powershell
   cd deploy
   .\milvus-standalone.bat start
   docker ps --filter name=milvus-standalone
   ```

   首次启动会下载 Milvus 镜像，等待容器显示 `healthy`。本地地址默认是 `http://127.0.0.1:19530`；数据保存在 Docker 命名卷 `llm-staudy-milvus-data`。日常停止可执行 `.\milvus-standalone.bat stop`。

   后端启动后可访问 `http://127.0.0.1:8000/health/milvus` 确认连接。如果容器异常退出，先查看 `docker logs milvus-standalone`，然后使用 `docker start milvus-standalone` 重启。本地容器配置为 `unless-stopped` 自动重启；遇到数据卷 I/O 错误时先备份和检查 Docker Desktop，不要直接删除数据卷。

   旧版 Windows 路径数据已在本机复制到命名卷；原目录 `deploy/volumes/milvus` 和已停止的 `milvus-standalone-bind-backup` 容器暂作备份。新环境直接运行上述启动命令即可，不需要旧目录。
3. 安装并启动后端：

   ```powershell
   cd backend
   python -m pip install -r requirements.txt
   python -m app.infrastructure.database.init_db
   python run.py
   ```

4. 另开终端启动前端：

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

打开 `http://localhost:5173/login`。开发时 Vite 将 `/api` 代理到 `http://127.0.0.1:8000`。注册成功会自动登录；也可以用用户名或邮箱登录。聊天请求使用 Bearer token。回答在书籍依据核对通过后，再通过 SSE 分段发送；因此首段文字需要等待检索、生成和核对完成。

## 书籍知识库

登录后在聊天页右上角打开“知识库”，上传 PDF、Markdown 或 UTF-8 TXT。单文件上限 25 MB，导入状态会自动刷新；失败后可重试。扫描版 PDF 需要先做 OCR。每本书最多切分 4000 段，超出时请拆分文件。书籍文件保存在 `backend/data/books`，仅所有者的书籍会进入检索；聊天回答展示引用书名、PDF 页码或文本段号。删除书籍会同时删除对应的 Milvus 向量和本地文件。项目不附带书籍，请只导入你有权使用的文件。

第一次验证流程时可上传 `backend/evals/demo_corpus.md`；它是项目自编的演示资料，不计入书籍评测。真实评测所用 PDF 的哈希和页数记录在 `backend/evals/corpus_manifest.json`，原文文件不随代码分发。

导入流程按层划分：`rag/parser.py` 解析和切分，`rag/embeddings.py` 调用 Embedding 模型，`rag/milvus_store.py` 存取向量，`services/knowledge_service.py` 协调导入。所有聊天问题均通过 `agents/chat_graph.py` 的书籍检索路径；`agents/tools.py` 提供受控检索工具。没有候选片段、回答缺少有效引用，或独立的证据核对判定结论不受引用片段支持时，系统统一说明知识库依据不足，不使用模型自身知识直接回答。用户长期记忆不会作为书籍证据注入回答。更换 Embedding 模型或维度时，需使用新的 `MILVUS_COLLECTION` 名称并重新导入书籍。模型核对仍可能出错，因此不能把这一机制视为绝对无幻觉保证。

检索还会过滤相似度低于 `RAG_MIN_SCORE`（默认 0.55）的片段。这个阈值基于当前小型语料的正反例初步选定，应在更换书籍或 Embedding 模型后重新评估。

## 评测

准备 UTF-8 JSONL 文件，每行一个问题及人工标注的相关页码。`evals/llm_books_questions.jsonl` 包含针对本地两本 PDF 的 10 道正例和 5 道无依据反例，使用文件 SHA-256 标识语料，避免重新上传后书籍 ID 改变：

```json
{"question":"为什么 QA 抽取需要保存中间结果？","relevant_pages":[{"book_sha256":"书籍文件的 SHA-256","pages":[4]}]}
```

运行前确保样例对应的原始 PDF 已由同一用户导入且状态为 `ready`。其他书籍可建立自己的标注集；也支持 `relevant_book_ids`、`relevant_book_sha256` 和 `relevant_chunk_ids`。运行：

```powershell
cd backend
python -m app.rag.evaluate evals/llm_books_questions.jsonl --user-id 你的用户ID --k 5 --output evals/retrieval_result.json
```

结果包含 Hit@K、MRR、P50/P95 检索延迟和逐题命中情况。`--user-id` 必须是导入这些书籍的用户。当前样例集较小，同一本书可能在其他页重复同一知识点，因此页码级指标只应作为初始基线。

无依据反例使用 `{"question":"问题","should_abstain":true}` 标注，评测还会输出弃答准确率。

回答评测可将实际聊天结果保存为 JSONL，每行含 `question`、`answer`、`sources`，以及人工标注的 `required_facts`（应出现的短语）或 `relevant_book_ids`：

```json
{"question":"Checkpoint 有什么作用？","answer":"Checkpoint 保存图状态 [1]。","sources":[{"book_id":"书籍 ID"}],"required_facts":["保存图状态"],"relevant_book_ids":["书籍 ID"]}
```

运行 `python -m app.rag.evaluate_answers evals/answers.jsonl --output evals/answer_result.json`。它计算事实短语覆盖率、引用编号有效率、相关书籍命中率；这些只能作为可重复的初步检查，事实是否真实受到书籍支持仍需人工审阅。

若需自动运行实际 LangGraph 回答链并保存评分：

```powershell
python -m app.rag.evaluate_live evals/llm_books_answers.jsonl --user-id 你的用户ID --output evals/answers_live.jsonl
```

原始回答文件默认不纳入 Git；同目录的 `answers_live.metrics.json` 保存汇总与逐题评分。此命令会调用真实模型 API，产生实际请求费用。评测每道题使用独立的临时图状态，不使用生产会话中的长期记忆。

最终回答的弃答情况另用 `evals/llm_books_abstention.jsonl` 评估：

```powershell
python -m app.rag.evaluate_live evals/llm_books_abstention.jsonl --user-id 你的用户ID --output evals/abstention_live.jsonl
```

样本以 `should_abstain: true` 标注；只有回答明确说知识库依据不足且未返回来源，才记为正确弃答。这个指标与检索层“是否返回片段”分开统计。

在 2026-09-28 的本地快照中，用户已导入五份资料，共 228 个文本段。**面向用户的结果应优先看完整回答链**：5 道无依据题均明确拒答且未返回来源，回答层弃答准确率为 5/5；5 道正例回答的事实短语覆盖率、引用编号有效率、相关书籍命中率均为 1.00，回答延迟 P50 为 5.38 秒、最大约 19.09 秒。检索层指标保留作内部诊断：针对最初两份 PDF 的 10 道页码标注题，Hit@5 为 1.00、MRR 为 0.6533；5 道无依据反例中有两题仍检索到主题相近但不足以回答的片段，所以检索层弃答准确率为 3/5。两道误检均被后续回答核对挡住。回答先核对再发送会增加等待时间。这些都是小样本和格式指标，不等于完整的事实准确率；新增书籍后应持续扩充标注集并人工复核答案。模型服务曾出现一次约 281 秒的异常延迟；目前模型请求超时设为 60 秒并限制重试，端到端超时仍可能略高于 60 秒。

生产构建运行 `npm run build`。若前后端分开部署，把 `VITE_API_BASE_URL` 设为后端地址，并把前端来源加入后端的 `FRONTEND_ORIGINS`（逗号分隔）。

后端接口测试可运行 `python -m pip install -r requirements-dev.lock`，然后执行 `python -m pytest tests -q`。测试使用临时 SQLite 和模拟图输出，无需真实 PostgreSQL 或模型密钥。正常聊天和书籍向量化需要配置模型密钥、PostgreSQL 与 Milvus。`backend/requirements.lock` 和 `backend/requirements-dev.lock` 固定了在 Windows / Python 3.13 上验证过的依赖版本；使用 `python -m pip install -r requirements.lock` 可以安装同一版本集合。
