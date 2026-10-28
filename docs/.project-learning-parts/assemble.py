"""将已核对的模块片段合并为学习页面数据；不修改业务源码或既有 Markdown。"""
import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs'
PARTS = Path(__file__).parent
SKILL = Path('C:/Users/19344/.codex/skills/project-learning')


def text(value):
    return {'type': 'text', 'text': value}


def table(headers, rows):
    return {'type': 'table', 'table': {'headers': headers, 'rows': rows}}


def code(language, content, caption):
    return {'type': 'code', 'code': {'language': language, 'content': content, 'caption': caption}}


def diagram(content, caption):
    return {'type': 'diagram', 'diagram': {'language': 'mermaid', 'content': content, 'caption': caption}}


def section(title, anchor, *blocks):
    return {'title': title, 'anchor': anchor, 'blocks': list(blocks)}


def evidence(rel, needle, count=10):
    lines = (ROOT / rel).read_text(encoding='utf-8-sig').splitlines()
    start = next(i for i, line in enumerate(lines) if needle in line)
    return f'{(ROOT / rel).as_posix()}:{start + 1}', '\n'.join(lines[start:start + count])


def step(layer, rel, function, needle, explanation, key_point, count=10):
    path, source = evidence(rel, needle, count)
    return {'layer': layer, 'filePath': path, 'functionName': function,
            'explanation': explanation, 'code': source, 'language': 'python', 'keyPoint': key_point}


def chain(label, *steps):
    return {'type': 'callchain', 'callchain': {'endpoint': label, 'steps': list(steps)}}


def routes():
    result = []
    for file in [ROOT / 'backend/app/main.py', *sorted((ROOT / 'backend/app/api/v1').glob('*.py'))]:
        tree = ast.parse(file.read_text(encoding='utf-8-sig'))
        prefix = ''
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'APIRouter':
                prefix = next((ast.literal_eval(k.value) for k in node.keywords if k.arg == 'prefix'), '')
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in node.decorator_list:
                if (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)
                        and deco.func.attr in ('get', 'post', 'put', 'patch', 'delete')):
                    suffix = ast.literal_eval(deco.args[0]) if deco.args else ''
                    path = ('' if file.name == 'main.py' else '/api/v1') + prefix + suffix
                    result.append([deco.func.attr.upper(), path, file.relative_to(ROOT).as_posix(), node.name + '()'])
    return result


def base_chapters():
    prep = [
        section('项目简介与当前边界', 'prep-project', text(
            '本项目是面向大模型工程知识的书籍 RAG 学习助手。RAG 的意思是先从指定资料找证据，再把问题和证据交给模型生成答案；本项目的资料来自用户自行导入的 LLM 相关书籍。默认使用 Milvus 原生 BM25 与稠密向量召回、RRF 融合和硅基流动 rerank。用户可以注册、管理会话、导入 PDF/Markdown/TXT、选择模型，并展开书名、位置和摘录核对来源。\n\n'
            '这份页面依据本地源码生成，范围为全部模块，粒度为 standard。重点代码是摘录，旁边标明文件与行号；不要把摘录当成可单独运行的完整程序。项目使用固定 LangGraph 工作流调用检索工具，不是任意工具自主选择的通用 Agent；没有内置 OCR、持久任务队列或零幻觉保证。'),
            table(['学习顺序', '先理解什么', '读完怎样自测'], [
                ['前置准备、架构设计', '请求、数据库、图状态分别由谁管理', '三分钟介绍项目，并说明 PostgreSQL 和 Milvus 的分工'],
                ['账户与会话 → 书籍知识库', '身份、归属、文件到向量的完整过程', '说清 user_id/book_id/conversation_id/thread_id 的区别'],
                ['对话与上下文 → 模型管理', '摘要、检索、流式核对、模型运行参数', '用一次长会话追问说明参数和返回结果怎样传递'],
                ['前端 → 运行与评测 → 面试', 'HTTP/SSE 边界、故障与验证', '按现象找到对应日志、状态和代码，不背未核实分数'],
            ])),
        section('技术栈概览', 'prep-stack', text(
            '先分清每种组件解决的问题，再记名称。前端负责操作和展示，FastAPI 负责 HTTP 接口，LangGraph 负责节点运行与状态持久化；模型推理和 Embedding 则由配置的外部服务提供。项目没有使用 Redis 或消息中间件，不需要为学习环境额外安装这些组件。'),
            table(['组件', '当前技术', '为何使用 / 代码依据'], [
                ['后端与验证', 'Python、FastAPI、Pydantic', '请求解析、Depends 认证依赖、参数验证；backend/app/api/v1 与 schemas'],
                ['关系数据', 'PostgreSQL、SQLModel、AsyncSession、asyncpg', '用户、会话、书籍元信息、模型配置、生成文件元数据；models 与 repositories'],
                ['图状态与长期记忆', 'LangGraph、AsyncPostgresSaver、AsyncPostgresStore', '按 thread_id 恢复会话；用户 namespace 隔离长期记忆；main.py/agents'],
                ['混合检索与重排', 'Milvus BM25、RRFRanker、BAAI/bge-m3、bge-reranker-v2-m3', '语义和关键词两路召回、排名融合、相关性重排；reranker.py 与 milvus_store.py'],
                ['文本处理', 'pypdf、RecursiveCharacterTextSplitter', '提取可复制的 PDF 文本及递归切分；不执行 OCR'],
                ['模型接入', 'LangChain、langchain-openai、HTTPX、OpenAI SDK', '兼容接口、结构化输出、模型流式调用；llm 与 model_service.py'],
                ['生成工具', '原生 tool_calls、python-docx、MarkdownIt、Pillow', '模型选择图片或 Word 工具，可信正文与身份由后端注入；artifact_tools.py 与 artifacts'],
                ['认证及凭据', 'Argon2、PyJWT、Fernet', '密码哈希、JWT 签名、可恢复的模型 Key 加密，各有不同用途'],
                ['前端与验证', 'React、TypeScript、Vite、普通 CSS、pytest', '两页应用与 SSE 读取；测试使用模拟模型或 HTTP transport，不能替代真实效果评测'],
            ])),
        section('环境搭建与运行', 'prep-run', text(
            '先启动依赖服务，再启动后端，最后启动前端。下面命令针对这个已有的本地目录，不包含虚构的 Git 仓库地址；请使用已安装项目依赖的 Python 环境。PostgreSQL 必须先启动，并创建 DATABASE_URL 指向的数据库；create_all 创建表，不会帮你创建数据库本身。\n\n'
            'backend/.env.example 是配置示例。仅在 backend/.env 不存在时复制，已有配置不要覆盖；填写数据库连接、JWT_SECRET_KEY、硅基流动 Key/地址，官方 Key 可选。所有示例凭据都只是占位符，不是实际密钥。默认回答模型名称和服务可用性应按实际账号核对。'),
            code('powershell', '''Set-Location 'E:\\work\\cache\\project\\VsCode\\llm-staudy'
if (-not (Test-Path -LiteralPath 'backend/.env')) {
    Copy-Item -LiteralPath 'backend/.env.example' -Destination 'backend/.env'
}
# 编辑 backend/.env，确保 PostgreSQL 已运行，目标数据库已存在。
Set-Location deploy
.\\milvus-standalone.bat start
docker ps --filter name=milvus-standalone''', '启动项目实际使用的本地 Milvus 脚本'),
            code('powershell', '''# 终端一：使用正确的 Python 环境
Set-Location 'E:\\work\\cache\\project\\VsCode\\llm-staudy\\backend'
python -m pip install -r requirements.txt
python -m app.infrastructure.database.init_db
python run.py''', '后端入口 run.py 也处理 Windows 的 psycopg 事件循环兼容'),
            code('powershell', '''# 终端二
Set-Location 'E:\\work\\cache\\project\\VsCode\\llm-staudy\\frontend'
npm install
npm run dev
# 浏览器访问 http://localhost:5173/login''', '前端默认 5173；/api 代理到 127.0.0.1:8000'),
            code('powershell', '''# 后端已启动后，在另一个终端逐项检查
Invoke-RestMethod 'http://127.0.0.1:8000/health'
Invoke-RestMethod 'http://127.0.0.1:8000/health/db'
Invoke-RestMethod 'http://127.0.0.1:8000/health/milvus' ''', '第一组 API 验证；HTTP 存活不等于数据库、Milvus 或模型都可用'),
            text('已有旧向量集合的环境，须在启动后端前暂停全部写入并执行 python -m app.rag.migrate_index --report evals/hybrid_index_migration.json；新环境没有旧数据时跳过。迁移不重新调用 Embedding，校验后发布新别名，旧集合保留。\n\n浏览器注册后导入 backend/evals/demo_corpus.md，等待 ready 再提问并展开来源，是最小验证路径。演示资料不是用户真实书籍评测集。阅读本文不需要启动真实模型、上传私人资料或支付接口调用费用；只有亲自运行问答、连接测试或真实评测时才会请求模型服务。')),
    ]
    arch = [
        section('系统架构图', 'arch-system', text(
            '一次浏览器操作会跨越多层。API 把 HTTP 数据转换成 schema 并取得可信身份，Service 协调业务，Repository 负责关系库读写；检索模块操作 Milvus，Graph 则把上下文准备、检索、生成和核对串起来。图中的箭头表示数据交互方向，源码中的实际调用在实战章节逐步展开。'),
            diagram('''flowchart TD
  UI["React：登录 / 聊天 / 知识库 / 模型选择"] --> API["FastAPI：路由、验证、JWT 身份"]
  API --> SERVICE["Service：会话、书籍、模型配置"]
  SERVICE --> REPO["Repository：用户隔离的关系库操作"]
  REPO --> PG[("PostgreSQL：用户、会话、书籍状态、模型配置")]
  API --> GRAPH["LangGraph：按需摘要与规划 → 书籍问答或生成工具"]
  GRAPH --> CP[("PostgreSQL：checkpoint / Store")]
  GRAPH --> TOOL["search_book_chunks：受控工具"]
  TOOL --> KNOW["BM25＋向量 → RRF → rerank"]
  SERVICE --> INGEST["解析 → 切分 → Embedding"]
  INGEST --> VEC[("Milvus：向量、正文与来源")]
  KNOW --> VEC
  GRAPH --> MODEL["所选模型：生成、摘要与核对"]
  GRAPH --> EXPORT["generate_image / generate_word：模型选择生成工具"]
  EXPORT --> IMAGE["硅基流动图片模型"]
  EXPORT --> FILES[("本地生成文件")]
  EXPORT --> META[("PostgreSQL：generated_artifacts")]
  GRAPH --> SSE["custom / updates → SSE"]
  SSE --> UI''', '前端是操作层，知识证据和对话状态分开保存')),
        section('模块划分与源码阅读入口', 'arch-modules', text(
            '按业务能力划分六个学习模块，而不是把每个目录当成独立功能。一个功能常同时涉及 API、schema、Service 和 Repository，因此阅读时跟随调用链，不要只按文件夹顺序翻页。纯粹的增删查步骤会合并说明，值得深入的是授权、资源生命周期、失败补偿及模型输入的变化。'),
            table(['模块', '主要入口', '重点'], [
                ['账户认证与会话', 'auth.py / users.py / conversations.py / deps.py', '密码哈希、JWT、用户资源归属、thread_id'],
                ['书籍知识库与检索', 'knowledge.py / knowledge_service.py / rag', '后台导入、来源保留、Milvus 过滤与检索策略'],
                ['对话、上下文与记忆', 'chat.py / chat_graph.py / context.py / memories.py', '状态节点、增量摘要、真正流式、依据核对、Store'],
                ['模型配置与运行', 'models.py / model_service.py', '预设与用户配置、加密、授权解析、请求级图与预算'],
                ['前端交互', 'main.tsx / App.tsx / api.ts / ModelControls.tsx / MessageContent.tsx', '页面、请求、SSE 分帧、Markdown、来源详情'],
                ['运行保障与评测', 'main.py / database / deploy / tests / rag/evaluate*', '共享资源、健康检查、状态恢复、指标的真实含义'],
            ])),
        section('模块依赖关系与四种存储', 'arch-dependencies', text(
            '身份校验提供 current_user.id，之后会话、书籍和模型配置仍分别检查归属。会话关系记录提供 thread_id，checkpoint 用它恢复完整消息、摘要和游标；这与 Store 保存跨会话偏好不是同一件事。模型配置决定回答客户端和输入预算，Embedding 配置决定向量空间，两者不能混用。'),
            table(['上游 → 下游', '传递的数据', '不能混淆的边界'], [
                ['认证 → 各业务接口', '可信 user.id、数据库 session', '登录不等于拥有任意资源，get_owned 仍要检查'],
                ['会话记录 → Graph/checkpoint', 'conversation.thread_id', 'conversation_id 是公开业务编号，thread_id 是状态键'],
                ['模型配置 → Graph/context', 'ModelSpec、ContextLimits', '明文 Key 不进入 ChatState；切换不更换 Embedding'],
                ['书籍记录 → Milvus', 'user_id、ready book_ids、query_vector', 'PostgreSQL 决定可检索范围，Milvus 再做过滤'],
                ['磁盘 → 解析 → Milvus', '原文件 → PageText → BookChunk → vectors', '原文件用于重试；元数据记录进度；向量和正文用于检索'],
                ['Graph → checkpoint / Store', '完整会话状态 / 用户稳定信息', '会话摘要不是用户长期记忆，也不是书籍证据'],
            ])),
        section('代码中的设计方式与取舍', 'arch-patterns', text(
            '本项目可以从分层、工厂、依赖注入和状态图四个角度理解，不必给每个 if 分支套一个设计模式名称。create_chat_model 及模型服务集中创建客户端，FastAPI Depends 提供身份和 session，Repository 隔离关系库操作；这些做法减少调用处重复配置。\n\n'
            '关键取舍也要说明：BackgroundTasks 简单但不是持久任务队列；编译请求级图可以沿用同一节点结构但有装配开销；摘要减少输入量但有额外调用和信息损失；先流式展示再核对改善等待体验，但草稿可能最终被弃答替换。'),
            chain('启动装配 → 图节点注册 → 运行时使用',
                step('应用装配', 'backend/app/main.py', 'lifespan()', '    await init_db()',
                     'FastAPI 启动时先创建关系表并标记被中断的书籍导入，再创建共享 checkpoint 和 Store。默认模型及图在启动过程中装配，接口通过 app.state 引用它们，因此接口不需要每次重新建立默认资源。',
                     '共享存储由应用生命周期管理，用户请求仍使用独立身份和会话。', 8),
                step('图编排', 'backend/app/agents/chat_graph.py', 'build_chat_graph()', '    builder.add_node("manage_context"',
                     'build_chat_graph 将内部函数注册成有名称的节点，并通过边限定执行顺序。节点返回字典更新状态，messages 使用消息合并语义，其余字段通常覆盖；这也是摘要节点不返回 messages 就不会删除原历史的原因。',
                     '函数注册到节点，再由图的边决定执行，而不是接口逐个手写调用。', 12))),
        section('全部接口入口核对表', 'arch-routes', text(
            '下表由源码路由声明提取，覆盖 API_V1_PREFIX 为默认 /api/v1 时的业务接口和三个健康检查。它用于确认范围完整，不是参数文档；实战章节会说明非平凡路径怎样经过服务层、仓储、图或外部服务。若未来改动路由或前缀，需重新生成。'),
            table(['方法', '路径', '源文件', '接口函数'], routes())),
    ]
    interview_data = [
        ('架构与数据', '为什么需要 PostgreSQL 和 Milvus，不能只用一个数据库吗？',
         'PostgreSQL 保存用户、会话、书籍导入状态和 model_profiles，并由 LangGraph 的 Postgres 组件保存 checkpoint/Store；Milvus 保存 BookChunk 的向量、正文和来源字段。KnowledgeService.search 先通过 KnowledgeBookRepository.list_ready_ids 确定可查书籍，再调用 MilvusBookStore.hybrid_search 做两路过滤召回、RRF 融合，随后重排。这里的分工对应不同访问方式，并不意味着只有这种架构可行；它也引入跨存储失败补偿，不能声称存在一个共同事务。'),
        ('导入与一致性', '上传返回 202 为什么不能立即认为导入成功？失败后怎么办？',
         'knowledge.upload_book 先调用 KnowledgeService.upload 保存原件并创建 processing 记录，再安排 BackgroundTasks 执行 process。process 使用独立 AsyncSession 解析、切分、向量化和写 Milvus，结束后更新 ready/failed；失败会尝试清理向量，mark_interrupted_imports 在启动时处理遗留 processing。BackgroundTasks 是进程内执行，不具备持久队列的自动恢复保证，重试依赖保存的文件和状态。'),
        ('RAG 与依据', '向量检索已经命中，为什么还可能弃答？',
         '相似度只表示语义接近。retrieve 在预算内组织证据后，grounded_answer 要求引用编号，evidence_supports_answer 先用 citations_valid 检查编号范围，再让结构化核对模型判断候选结论是否被所引片段支持；失败或异常就固定弃答并清空来源。这个机制降低风险，但核对模型也会犯错，不能承诺零幻觉。'),
        ('上下文管理', '增量摘要怎样保留历史，为什么成功后才推进游标？',
         'manage_context 把完整 messages、conversation_summary 和 summarized_count 传给 ContextManager.prepare。prepare 从游标后选择未摘要旧消息，recent_start 保留近期原文，_batches 限制每批大小；对应批次成功后才更新 summary/cursor，失败保留已完成进度。返回字典不包含 messages，因此完整历史仍在 checkpoint；下游只用摘要和 context_messages。UTF-8 字节估算不是精确 tokenizer，摘要也可能遗漏信息。'),
        ('流式与核对', '为什么既有 model.astream，又有 graph.astream？最终答案覆盖草稿是什么意思？',
         'grounded_answer 中 model.astream 从供应商读取回答文字增量，writer 将文字作为 custom token/status 发给图；chat.event_stream 使用 graph.astream 接收 custom 和节点 updates，再序列化为 SSE。生成结束后独立核对，最终 AIMessage 通过 answer/sources 事件确认或替换草稿，前端 api.streamChat 和 App.send 的回调负责更新最后一条消息。历史保存最终结果，实时草稿可能尚未得到证据支持，不能把 SSE 等同于已核对 token。'),
        ('模型与安全', '切换模型为什么不丢历史？用户 Key 保存在哪里？',
         '浏览器只发 model_id，ModelService.resolve 读取平台预设或按 user_id 查询自定义记录，授权后解密成内部 ModelSpec。runtime_graph 为非默认模型创建独立客户端和 ContextLimits，但复用 app.state.checkpointer/Store，chat.py 仍传原 conversation.thread_id，所以状态连续。model_profiles.encrypted_api_key 使用 credential_cipher 的 Fernet 加密，public_profile 不返回密钥；若轮换加密所用服务端秘密，旧配置需重新添加。切换回答模型不改变 Embedding，因此不需重导书籍。'),
        ('认证与权限', '有 JWT 认证为什么还要 get_owned？工具的 user_id 为什么不能让模型决定？',
         'get_current_user 只证明当前身份，不证明对某个资源有权限。ConversationService.get_owned、KnowledgeBookRepository.get_owned、ModelProfileRepository.get_owned 继续约束资源归属，Milvus 查询也过滤用户和书籍。retrieve 从 runtime.context.user_id 传入 search_book_chunks，身份由认证接口设置，不由语言模型生成；否则攻击者可能通过请求或提示越权检索。'),
        ('混合检索与评测', 'RRF 与 rerank 分别解决什么问题，怎样证明改进有效？',
         'Milvus RRFRanker 按各路名次对 1/(k+rank) 求和，避免直接混加 BM25 和余弦原始分数；Reranker.rerank 则同时读取问题和候选原文重新评分。evaluate.py 用 --strategy 比较同一语料下的 dense、hybrid、hybrid_rerank，前 K 统计与额外短书上下文分开。2026-10-22 的 15 题中，默认完整链路 MRR@5 为 1.0，dense 为 0.6533；检索层弃答分别为 1.0 和 0.60，P50 从约 271 ms 增至 1004 ms。该小型既有数据集不是独立留出集，不能代表所有问题效果；最终事实核对和人工审阅仍然必要。'),
    ]
    interview = [section('用实际实现回答面试追问', 'interview-questions',
                        text('回答先讲场景和数据流，再解释具体函数及取舍。下面八题的参考答案依据当前源码，不包含未实现的持久任务队列、OCR 或工具自主规划；不要背成产品能力保证。'),
                        *[{'type': 'question', 'question': {'category': c, 'question': q, 'answer': a}} for c, q, a in interview_data]),
                 section('闭卷自测与故障定位', 'interview-self-check',
                         text('能看懂不等于能在面试时讲清。先关掉页面复述两条主链，再打开对应代码核对；忘记名字时说明职责和数据，不要编造实现。'),
                         table(['自测', '必须给出的具体证据'], [
                             ['一本书上传后查不到', '检查 ready/error、解析文字、chunk_count、向量维度、用户过滤及分数'],
                             ['长会话失败', '检查窗口配置、摘要游标、context_messages、ensure_input_budget 和模型错误'],
                             ['模型切换失败', '检查 model_id/归属、地址/模型名/Key、流式及结构化输出方式'],
                             ['草稿被弃答替换', '区分 token 草稿与最终依据核对，查看 final answer/sources'],
                             ['可以开始面试的标准', '三分钟介绍、两条主链、关键代码骨架、失败排查、如实说明局限'],
                         ]))]
    return prep, arch, interview


def assemble():
    prep, arch, interview = base_chapters()
    practice = []
    for name in ['identity', 'knowledge', 'dialogue', 'models', 'frontend', 'operations']:
        module = json.loads((PARTS / f'{name}.json').read_text(encoding='utf-8-sig'))
        for idx, sec in enumerate(module['sections']):
            sec['groupTitle'] = module['moduleName']
            if idx == 0:
                sec['blocks'][0]['text'] = module['moduleDescription'] + '\n\n' + sec['blocks'][0]['text']
            practice.append(sec)
    doc = {
        'meta': {'projectName': 'LLM Learning Assistant', 'primaryLanguage': 'Python',
                 'framework': 'FastAPI / LangGraph', 'description': '从源码学习书籍 RAG、增量上下文与模型接入；范围 all，含面试准备。',
                 'granularity': 'standard', 'generatedAt': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()},
        'chapters': [
            {'id': 'preparation', 'title': '前置准备', 'sections': prep},
            {'id': 'architecture', 'title': '架构设计', 'sections': arch},
            {'id': 'practice', 'title': '实战演练', 'sections': practice},
            {'id': 'interview', 'title': '面试准备', 'optional': True, 'sections': interview},
        ],
    }
    (OUT / 'project-learning.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
    page = (SKILL / 'assets/template.html').read_text(encoding='utf-8-sig')
    # 保留 JSON 作为数据源；文件方式打开或 HTTP 数据加载失败时用同版内嵌数据回退。
    embedded = json.dumps(doc, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    page = page.replace('const EMBEDDED_DATA = null;', 'const EMBEDDED_DATA = ' + embedded + ';')
    page = page.replace('catch (e) { docData = null; }', 'catch (e) { docData = null; }\n  if (!docData) docData = EMBEDDED_DATA;')
    # 本项目专用阅读进度，避免与其他项目的同模板页面共用记录。
    page = page.replace("'pl-", "'llm-staudy-learning-")
    # 模板底部四个控件在窄屏需收紧间距，否则按钮会被裁掉。
    page = page.replace('</style>', '''
@media (max-width: 768px) {
  .section-nav { gap: 6px; padding: 0 8px; height: 64px; }
  .section-nav-btn { gap: 4px; padding: 9px 8px; font-size: .72rem; }
  .section-nav-info { min-width: 44px; font-size: .72rem; }
}
.diagram-view-btn { margin: 14px auto 0; }
.diagram-dialog { width: min(96vw, 1500px); max-width: 96vw; max-height: 92vh;
  padding: 18px; border: 1px solid var(--border-card); border-radius: 12px;
  color: var(--text-primary); background: var(--bg-surface); }
.diagram-dialog::backdrop { background: rgba(0,0,0,.55); }
.diagram-dialog-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.diagram-dialog-content { max-height: 75vh; overflow: auto; margin-top: 16px; }
</style>''')
    page = page.replace('</body>', '''
<dialog id="diagramDialog" class="diagram-dialog" aria-label="图表原尺寸查看">
  <div class="diagram-dialog-heading"><strong>图表原尺寸查看（可横向滚动）</strong>
    <button type="button" class="section-nav-btn" onclick="closeLearningDiagram()">关闭</button></div>
  <div id="diagramDialogContent" class="diagram-dialog-content"></div>
</dialog>
<script>
const originalDiagramRenderer = renderDiagramBlock;
renderDiagramBlock = function(data) {
  const html = originalDiagramRenderer(data);
  return html.replace(/<\\/div>$/, '<button type="button" class="section-nav-btn diagram-view-btn" onclick="openLearningDiagram(this)">原尺寸查看图表</button></div>');
};
function openLearningDiagram(button) {
  const source = button.closest('.diagram-wrapper').querySelector('svg');
  if (!source) return;
  const clone = source.cloneNode(true);
  clone.style.maxWidth = 'none';
  clone.style.width = Math.ceil(source.viewBox.baseVal.width) + 'px';
  clone.style.height = 'auto';
  document.getElementById('diagramDialogContent').replaceChildren(clone);
  document.getElementById('diagramDialog').showModal();
}
function closeLearningDiagram() {
  document.getElementById('diagramDialog').close();
  document.getElementById('diagramDialogContent').replaceChildren();
}
</script>
</body>''')
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    print('Assembled', len(practice), 'practice sections;', len(routes()), 'routes')


if __name__ == '__main__':
    assemble()
