# LLM Learning Assistant

后端使用 FastAPI、LangGraph、PostgreSQL 和 Milvus。PostgreSQL 保存用户、会话与图状态；Milvus 保存 LLM 工程书籍的向量。前端仍是 `/login` 和 `/chat` 两个简易页面，书籍管理位于聊天页的“知识库”弹窗。

理解项目的设计顺序、请求流程和面试讲法，可先阅读 [项目一从零学习计划](docs/项目一从零学习计划.md)。

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

   旧版 Windows 路径数据已在本机复制到命名卷；旧备份容器已清理。原目录 `deploy/volumes/milvus` 仍保留在磁盘上，但当前容器不再挂载它。新环境直接运行上述启动命令即可，不需要旧目录。
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

打开 `http://localhost:5173/login`。开发时 Vite 将 `/api` 代理到 `http://127.0.0.1:8000`。注册成功会自动登录；也可以用用户名或邮箱登录。聊天请求使用 Bearer token。检索完成后，回答模型通过 `model.astream()` 边生成边输出，`graph.astream()` 将文字增量转成 SSE，前端标明内容待核对。生成结束后仍进行书籍依据核对：通过时确认最终答案，失败时将草稿替换为依据不足的回复并清除来源。历史记录只保存核对后的最终结果；用户可能在核对结束前看到不受证据支持的草稿内容。

## 混合检索、RRF 与重排

默认检索现在使用 **Milvus 原生 BM25＋稠密向量召回 → RRF 融合 → 硅基流动 rerank → 证据筛选**。BM25 使用 jieba 中文分词与 lowercase 过滤，向量仍使用原 Embedding 配置。两路查询均限制当前用户及 ready 书籍；普通问题最终取合格的前 K 个片段。符合条件的短书数量问题会先补取并合并候选，再整体重排，附加的同书上下文明确标记 `is_expansion`，仍受图中的证据预算限制。

### 升级现有索引

先停止所有导入、删除和后端写入进程，再执行（Milvus 保持运行）：

```powershell
cd backend
python -m app.rag.migrate_index --report evals/hybrid_index_migration.json
python -m app.rag.migrate_index --verify-only
```

旧集合由 `MILVUS_COLLECTION` 指定，新入口由 `MILVUS_HYBRID_COLLECTION` 指定。迁移把原片段 ID、原文、元数据和已有向量复制到新暂存集合，由 Milvus 自动生成 BM25 稀疏向量；校验所有行的内容摘要、原始 ID 和混合检索后，才发布目标别名。**不删除旧集合，不重新调用 Embedding，不要求重新上传原文件。** 目标已存在时拒绝覆盖；新环境没有旧数据可直接导入。迁移只检查本机 8000 端口是否停止，其他主机或自定义端口的写入须自行暂停。迁移后不双写旧集合，旧集合是迁移时的快照，不包含后续新导入。

### 配置和故障

- `RAG_RETRIEVAL_MODE`：`dense` 纯向量基线、`hybrid` 仅混合融合、`hybrid_rerank` 默认完整链路。
- 每路召回默认 `RAG_RECALL_K=20`，融合最多 `RAG_FUSION_K=40`，RRF 参数 `RAG_RRF_K=60`，含短书补取的重排候选上限默认 64。
- `RERANK_MODEL=BAAI/bge-reranker-v2-m3`。`RERANK_BASE_URL`、`RERANK_API_KEY` 留空时沿用硅基流动服务配置；与前端选择的聊天模型相互独立。
- `RERANK_MIN_SCORE=0.30` 是初始工作阈值，应在更换语料或模型后重新校准；`RAG_MIN_SCORE=0.55` 仅用于 dense 基线，不能用于 RRF 或重排分数。
- `score_type` 解释当前 `score` 的含义，同时保留 `dense_score`、`fusion_score`、`rerank_score`。低分 `is_expansion` 片段是同书上下文，不代表独立通过了相关性筛选。
- 重排增加一次服务请求及费用；超时、鉴权失败或返回编号不完整时明确报错，不悄悄退回无重排结果。需要人工排障时可切到 `hybrid` 或 `dense`，报告会记录实际模式。

### 对比评测

```powershell
python -m app.rag.evaluate evals/llm_books_questions.jsonl --user-id 你的用户ID --strategy dense --k 5 --output evals/dense.json
python -m app.rag.evaluate evals/llm_books_questions.jsonl --user-id 你的用户ID --strategy hybrid --k 5 --output evals/hybrid.json
python -m app.rag.evaluate evals/llm_books_questions.jsonl --user-id 你的用户ID --strategy hybrid_rerank --k 5 --output evals/hybrid_rerank.json
```

2026-10-22 本机 5 本 ready 资料、15 道标注题（10 正例、5 负例）的验证结果如下，原报告保存在 `backend/evals/*_20261022.json`。这是同一小型数据集上的比较，不是独立留出集或普遍质量保证，也未证明最终回答零幻觉：

| 模式 | Hit@5 | MRR@5 | 检索层弃答准确率（5 负例） | P50 / P95 检索耗时 |
| --- | --- | --- | --- | --- |
| dense | 1.00 | 0.6533 | 0.60 | 271 / 805 ms |
| hybrid（无重排） | 0.90 | 0.6750 | 0.00 | 286 / 1015 ms |
| hybrid_rerank | 1.00 | 1.0000 | 1.00 | 1004 / 1420 ms |

融合阶段用于扩大候选范围，不使用余弦阈值，因此仅融合模式可能包含无关候选。真正重排和阈值筛选是单独阶段。列举题可能额外返回同书上下文，但 Hit@K/MRR@K 只统计前 K 项，`returned_count` 另记返回总数，避免把扩展后的全部结果冒充前 K 命中。

## 模型选择与自定义参数

问题输入框左侧提供模型下拉框和“添加 / 管理模型”。可以选择服务端默认模型、额外预设和自己保存的模型配置。点击管理入口后，填写显示名称、模型标识、兼容 OpenAI 的 API 根地址及 API Key；无需鉴权的本地模型可以留空密钥。也可导入不超过 64 KB 的 JSON 参数文件，格式示例在弹窗内。这里导入的是接入参数，不是模型权重；本地模型须已通过 Ollama、vLLM 等服务提供兼容接口。

支持上下文窗口、最大输出 token、temperature、top_p、超时及结构化输出方式。建议先点击“测试连接”，它会分别验证结构化输出和流式文字，会发起两次短模型请求。若模型不支持默认的工具调用，可在服务支持的前提下选择 JSON Schema 或 JSON 模式再测试。保存后自动选择新模型；配置按用户持久化，只能查看和删除自己的配置。

切换模型只影响下一次聊天调用，不清空页面、删除消息或创建新会话。后端复用原 `thread_id`、checkpoint 和 Store，为所选模型创建独立客户端和输入预算；回答、改写、摘要、核对和记忆提炼都使用所选聊天模型。Embedding 模型及 Milvus 向量库保持不变，不需要重新导入书籍。模型配置中的窗口须按实际服务规格填写，小窗口会缩小历史、摘要和证据预算。

API Key 经 Fernet 加密后存入 `model_profiles` 表，不出现在模型列表或聊天历史中，也不写入浏览器持久存储。可配置独立的 `MODEL_CREDENTIAL_SECRET`；未配置时从 `JWT_SECRET_KEY` 派生。更换用于加密的秘密后，旧配置无法解密，需要重新添加。额外服务端预设使用 `CHAT_MODEL_OPTIONS`（逗号分隔的模型标识），沿用服务端 SiliconFlow 凭据；应用不自动承诺这些模型在服务商上可用。

升级后安装后端依赖并重启：`python -m pip install -r requirements.txt`、`python run.py`。启动初始化会创建新配置表；已有会话和 checkpoint 不迁移、不清空。自定义配置 API 为 `GET/POST /api/v1/models`、`DELETE /api/v1/models/{id}`、`POST /api/v1/models/test`；聊天请求增加可选 `model_id`，未传时与原默认行为一致。

## 图片与 Word 生成工具

聊天中可以直接输入：

- `请生成一张图片：蓝色背景、一本书和一个机器人，方形插画。`
- `把刚才的回答导出为 Word，标题为学习笔记。`
- `请根据知识库生成一份 Word 文档，介绍 RAG 的工作流程。`

新增 `agents/artifact_tools.py` 的 `generate_image` 和 `generate_word` 两个工具。生成/导出候选先由本地规则筛选，再由所选聊天模型通过 `bind_tools(tool_choice="auto")` 选择工具及参数。普通 RAG 提问沿用原来的检索路径；每轮最多各生成一张图片和一份 Word。此功能要求聊天模型支持原生工具调用，配置成 JSON 结构化输出也不代表其一定支持工具调用。

Word 支持导出最近一条已核对回答，或先检索知识库、生成并核对正文后导出；无依据时保留弃答，不创建知识文档。正文、来源、用户身份、会话编号和幂等键由后端注入，模型不能指定保存路径。Word 保留 Markdown 标题、强调、列表、代码、表格与书籍参考位置。图片为生成式创作，独立调用硅基流动图片模型，不充当书籍证据。

图片默认使用 `Kwai-Kolors/Kolors`，从 `IMAGE_API_KEY/IMAGE_BASE_URL` 读取配置，留空时沿用服务端硅基流动账号；切换聊天模型不切换图片服务。可通过 `IMAGE_MODEL` 配置其他兼容图片模型。接口约定见[硅基流动官方文档](https://api-docs.siliconflow.cn/docs/api/images-generations-post)。服务商返回的临时 URL 会被后端立即下载并验证，前端只访问本地受保护文件。`IMAGE_DOWNLOAD_HOSTS` 限定可信图片存储域名；仅在服务商正式更换域名后调整。

新增 PostgreSQL 表 `generated_artifacts`，记录所属用户、会话、文件类型、文件名、大小与存储位置；启动时自动创建。文件默认保存到 `backend/data/generated/<user_id>/`，生成目录已被 Git 忽略。`GET /api/v1/files/{id}/content` 每次校验登录身份及所有者，前端携带 Bearer 获取 Blob，显示图片预览或触发 Word 下载。SSE 新增 `artifacts` 事件，附件也写入 AIMessage 并随 checkpoint 恢复历史。

升级后在后端目录运行 `python -m pip install -r requirements.txt`，重启后端和前端。收费的图片生成及文件写入节点不配置自动重试；同一工具请求复用已有文件结果。当前为请求内生成，未接入持久任务队列，断开连接不提供后台任务进度恢复。测试位于 `tests/test_artifacts.py`，实际模型烟测位于 `evals/artifact_tools_smoke_20261022.json`，浏览器联调结果位于 `evals/artifact_tools_ui_20261022.json`。

## 对话上下文与历史摘要

工作流现在先执行 `manage_context`，再进入问题改写与书籍检索。完整聊天消息仍由 PostgreSQL checkpoint 保存，历史接口继续返回原文；模型输入单独使用 `conversation_summary` 和 `context_messages`。摘要进度 `summarized_count` 随会话保存，只总结新增的较早消息，旧会话无需清空或迁移关系表。

默认未摘要历史加摘要超过 6000 个保守估算单位时触发摘要；保留预算内的近期原文，旧消息每批最多 6000 单位总结，摘要最多 1800 单位。问题改写默认先经本地规则筛选：独立问题直接检索，有“它、刚才、继续”等指代或省略的追问才参考摘要和近期消息补全；专业词或数量本身不是跳过依据。摘要只帮助理解问题，不能作为书籍事实依据，也不会写入用户跨会话的长期记忆 Store。

### LangGraph 节点自动重试

`agents/retry.py` 定义 `RetryPolicy(max_attempts=2)`，通过 `builder.add_node("retrieve", retrieve, retry_policy=...)` 挂到只读检索节点。遇到网络超时、429 限流、部分临时服务端错误或明确的连接故障时，等待 0.5 秒后最多再执行一次。空检索结果、鉴权失败、参数错误、集合/索引配置错误和重排结果格式错误直接按原逻辑处理。两次都失败时，一次性接口返回 HTTP 502，SSE 接口返回中文 error，不发送最终答案或成功 done。

图内检索的 Embedding 客户端设置 `max_retries=0`，由 LangGraph 控制节点尝试次数；书籍导入与独立检索仍保留 Embedding SDK 默认两次重试。Milvus SDK 自身可能在单次调用的超时范围内重试，节点两次尝试不等于所有依赖的 HTTP/RPC 总请求数最多两次。其他聊天模型仍使用原来的 SDK 重试配置。

流式生成和长期记忆写入没有配置节点重试，避免文字重复输出或重复写入。问题改写、摘要与依据核对继续使用原有的回退/弃答处理；核对不通过不会循环重新生成。测试见 `tests/test_graph_retry.py` 和 `tests/test_chat_stream.py`。

### 减少辅助模型请求

`agents/call_policy.py` 决定是否需要问题改写和长期记忆提炼。默认 `CHAT_QUERY_REWRITE_MODE=auto`、`CHAT_MEMORY_MODE=selective`：普通知识问答只调用一次回答模型和一次依据核对模型；含指代的追问增加一次改写；学习目标、偏好或明确要求记住的信息才增加一次记忆提炼。没有检索证据时直接弃答。摘要仍只在历史超过阈值后增量执行，Embedding 和 rerank 是另外的模型服务请求，不计入这里的聊天模型次数。

自动筛选是启发式规则，可能漏掉没有明显指代词的省略表达或隐含的用户偏好。可把改写设为 `always` 恢复所有多轮问题改写，把记忆设为 `always` 恢复每轮判断，或设为 `off` 关闭自动提炼；手动记忆接口仍可使用。修改配置后重启后端。依据核对没有关闭，引用有效性也不等于事实正确性。

可用下列命令测量首字、核对后的最终答案与整轮耗时，并按节点统计聊天模型调用次数（会产生真实 API 费用；状态和记忆使用内存，不写生产会话）：

```powershell
cd backend
python -m app.rag.benchmark_chat --user-id YOUR_USER_ID --question "Checkpoint 是什么？" --question "什么是 Milvus 混合检索？" --output evals/chat_latency.json
```

本次两题顺序实测保存在 `evals/chat_latency_before_20261022.json` 和 `evals/chat_latency_after_20261022.json`：聊天模型请求从 3/4 次降为 2/2 次，第二轮首字由 16.23 秒降为 3.05 秒；但首轮首字由 3.34 秒升到 5.40 秒，整轮耗时分别为优化前 9.57/21.88 秒、优化后 10.67/20.54 秒。网络、输出长度与服务商负载会影响结果，这个小样本证明调用减少，不能据此声称总耗时稳定下降。

配置见 `backend/.env.example`：部署窗口预算默认 32768，回答输出上限 2048，安全余量 1024，书籍证据上限 12000。文本使用 UTF-8 字节数保守估算，通常会比实际 token 数更早触发摘要，不能把它当作精确 tokenizer。窗口必须按所选模型服务实际规格配置；如果减小窗口，也需降低历史、摘要、证据及输出预算。回答、问题改写、摘要、核对和记忆提炼都检查输入预算并限制输出。摘要服务失败时，暂时使用已有摘要和有限近期消息；摘要游标保留，下轮可重试。当前问题本身超过近期消息预算时返回中文提示要求拆分，不静默截断它。

上下文管理位于 `backend/app/agents/context.py`，回归测试位于 `backend/tests/test_context.py`。摘要会带来额外模型调用及一定信息损失；保留原始聊天历史不代表每次将全部历史发送给模型。

## 书籍知识库

登录后在聊天页右上角打开“知识库”，上传 PDF、Markdown 或 UTF-8 TXT。单文件上限 25 MB，导入状态会自动刷新；失败后可重试。扫描版 PDF 需要先做 OCR。每本书最多切分 4000 段，超出时请拆分文件。书籍文件保存在 `backend/data/books`，仅所有者的书籍会进入检索；聊天回答展示引用书名、PDF 页码或文本段号。删除书籍会同时删除对应的 Milvus 向量和本地文件。项目不附带书籍，请只导入你有权使用的文件。

第一次验证流程时可上传 `backend/evals/demo_corpus.md`；它是项目自编的演示资料，不计入书籍评测。真实评测所用 PDF 的哈希和页数记录在 `backend/evals/corpus_manifest.json`，原文文件不随代码分发。

导入流程按层划分：`rag/parser.py` 解析和切分，`rag/embeddings.py` 调用 Embedding 模型，`rag/milvus_store.py` 存取向量，`services/knowledge_service.py` 协调导入。所有聊天问题均通过 `agents/chat_graph.py` 的书籍检索路径；`agents/tools.py` 提供受控检索工具。没有候选片段、回答缺少有效引用，或独立的证据核对判定结论不受引用片段支持时，系统统一说明知识库依据不足，不使用模型自身知识直接回答。用户长期记忆不会作为书籍证据注入回答。更换 Embedding 模型或维度时，需使用新的 `MILVUS_COLLECTION` 名称并重新导入书籍。模型核对仍可能出错，因此不能把这一机制视为绝对无幻觉保证。

纯向量基线会过滤相似度低于 `RAG_MIN_SCORE`（默认 0.55）的片段。默认混合重排模式使用独立的 `RERANK_MIN_SCORE`，不同阶段的分数不能共用阈值；更换书籍或模型后需要重新评估。

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
