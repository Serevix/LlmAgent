"""同步本次混合检索变更到已生成的学习页面片段，不改其他项目的资料。"""
import hashlib
import json
import re
from pathlib import Path

from assemble import ROOT, PARTS, section, text, table, step, chain


def read(name):
    """读取一个已有学习模块，保留不受本次检索调整影响的章节。"""
    return json.loads((PARTS / f'{name}.json').read_text(encoding='utf-8'))


def save(name, data):
    """按原片段格式写回，供 assemble 统一生成 JSON 和页面。"""
    (PARTS / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


k = read('knowledge')
old = {s['anchor']: s for s in k['sections']}
k['moduleDescription'] = '书籍从上传到向量化后，同时建立稠密向量与 BM25 稀疏索引。查询经过两路召回、RRF 融合及独立重排，再由图组织可引用的原文；关系表继续决定用户归属与 ready 状态。'
k['sections'] = [
    section('背景知识：召回、融合和重排分别解决什么', 'practice-knowledge-concept',
        text('背景知识：向量检索擅长找意思接近的文本，BM25 通过分词及词项统计寻找关键词匹配；两者都有盲点。项目先各取候选，Milvus 原生 RRFRanker 按排名而不是直接相加原始分数融合，公式是对每一路命中的 1/(k+rank) 求和，k 默认 60、rank 从 1 开始。随后专门的重排模型同时读取问题与原文，再计算相关性。\n\nRRF 合并的是不同召回信号，rerank 重新判断问题与候选之间的关系，二者不是重复步骤。最后的回答依据核对仍保留：相关片段不等于能够支持所有结论。'),
        table(['字段', '含义', '用途与限制'], [
            ['dense_score', '查询向量与片段向量的余弦相似度', 'dense 基线过滤；判断是否强命中短书'],
            ['fusion_score', 'Milvus 原生 RRF 分数', '候选排名，不使用原余弦 0.55 阈值'],
            ['rerank_score', '硅基流动重排相关性分数', '默认模式排序并以 0.30 初始阈值筛选，需重新校准'],
            ['score / score_type', '当前排序分数及其类型', 'dense、rrf、rerank、expansion 不同量纲，不当作正确概率'],
            ['is_expansion', '是否来自短书上下文扩展', '低分上下文不冒充独立合格命中，仍受图的证据预算限制'],
        ]),
        chain('检索结果数据结构', step('数据结构', 'backend/app/rag/types.py', 'RetrievedChunk', '    score: float',
            'search 返回带来源的 RetrievedChunk，工具再转成字典交给图。新增字段分别保留不同阶段分数，避免把 RRF 小数误用余弦阈值过滤；is_expansion 告诉后续这是同书上下文。',
            '先问分数是哪一阶段产生的，再讨论阈值。', 8))),
    section('集合结构：原文如何自动形成 BM25 索引', 'practice-knowledge-storage',
        text('PostgreSQL 的 KnowledgeBook 仍保存书籍归属、文件路径、哈希与 processing/ready/failed 状态，没有新增关系表。新的 Milvus 混合集合保留原来的 id/vector/user_id/book_id/title/page/text，text 启用 jieba＋lowercase 分词，再添加 sparse 字段与 BM25 Function。原有向量集合不就地改造，通过迁移脚本建立新集合并发布别名。'),
        chain('书籍导入 → 创建混合索引', step('Milvus', 'backend/app/rag/milvus_store.py', 'MilvusBookStore._ensure_collection()', '        schema.add_field("text",',
            '调用者仍只写原文与稠密向量，Milvus 的 BM25 Function 自动从 text 生成 sparse。两个向量字段分别建立 COSINE 和 BM25 索引，不需要在 Python 维护一份额外的关键词索引。',
            'BM25 稀疏字段由数据库函数生成，应用不上传手算 BM25 向量。', 11),
            step('验证', 'backend/app/rag/milvus_store.py', '_validate_hybrid_schema()', '        description = client.describe_collection(name)',
            '查询或写入已有混合集合时先检查稠密维度、sparse 字段及 BM25 函数。若连接到了旧集合或错误向量空间就拒绝继续，而不是返回看似正常的错误结果。',
            '字段存在与向量维度匹配都是索引可用的前提。', 8))),
    old['practice-knowledge-upload'],
    old['practice-knowledge-indexing'],
    section('默认检索调用链：问题到可引用证据', 'practice-knowledge-retrieval',
        text('调用顺序是图 retrieve → search_book_chunks → KnowledgeService.search。Service 先从关系库确认当前用户的 ready 书籍，再对问题向量化，调用 Milvus 原生混合检索；融合结果及可选补取候选整体交给 reranker，最终片段回到 retrieve。浏览器选择的聊天模型不改变 Embedding 和 rerank 服务。'),
        chain('chat_graph.retrieve → tool → KnowledgeService → Milvus → rerank → tool → retrieve',
            step('工具', 'backend/app/agents/tools.py', 'search_book_chunks()', '    async with AsyncSession(engine)',
                'retrieve 从可信 runtime.context 传入 user_id，不由模型决定检索身份。工具打开自己的关系库会话调用 KnowledgeService.search，再把结果转换成含原文、来源和分阶段分数的字典返回。',
                '工具返回证据及分数，不直接返回最终答案。', 12),
            step('Service', 'backend/app/services/knowledge_service.py', 'KnowledgeService.search()', '        book_ids = await self.books.list_ready_ids',
                '关系库先限定 ready book_ids；没有书籍立即返回，不发模型请求。hybrid_rerank 模式先融合及补取，再重排并筛选，关键词命中即使余弦较低也有机会进入重排。',
                '不能在混合重排之前用旧余弦阈值丢掉关键词候选。', 13),
            step('Milvus', 'backend/app/rag/milvus_store.py', '_hybrid_search()', '            requests = [',
                '两个 AnnSearchRequest 分别查询 vector 和 sparse，并携带相同用户及书籍过滤表达式。hybrid_search 使用 RRFRanker(k) 合并排名，输出当前融合分数；结果还按 ID 去重并核对归属。',
                '权限过滤同时应用于两路，而不是检索全库后才过滤。', 10),
            step('外部重排', 'backend/app/rag/reranker.py', 'Reranker.rerank()', '                response = await client.post',
                '重排接口收到 query 和候选原文，top_n 请求所有候选的分数。返回的 index 对应提交时 documents 的位置，应用必须按 index 找回原片段，不能假设结果与输入顺序一致。',
                '重新评分后仍保留原片段 ID、来源和融合分数。', 13),
            step('验证与排序', 'backend/app/rag/reranker.py', 'Reranker.rerank()', '            for result in results:',
                '逐项验证 index 唯一且不越界，分数必须有限且位于接口约定范围。再使用 replace 保留不可变片段内容并写入 rerank_score，异常响应明确报错，不静默冒充完整链路。',
                '重排失败与没有相关证据是不同情况。', 10))),
    section('数量问题：补取后重排，同时保留有界的同书上下文', 'practice-knowledge-expansion',
        text('“八种框架分别是什么”可能需要同一短文档中的多个页面，最终只取五个片段又会漏项。项目对明确数量问题，用独立 dense_score 判断是否强语义命中；书籍 ready 且总片段数不超过 24 时补取全文片段。合并、去重后整体重排，若最终合格结果包含这本书，就附带该书的扩展上下文；低分项标记 is_expansion，仍要接受回答依据核对和窗口预算。'),
        chain('KnowledgeService.search → _expand_candidates → book_chunks → rerank → 最终选择',
            step('扩展条件', 'backend/app/services/knowledge_service.py', '_expand_candidates()', '        if not hits or not asks_for_numbered_list(question):',
                '普通问题直接使用候选，数量问题才寻找 dense_score 最强的命中并检查短书条件。RRF 分数不具备余弦量纲，不能与 MIN_EXPANSION_SCORE 直接比较。',
                '强命中条件使用单独的余弦信号。', 11),
            step('补取与去重', 'backend/app/services/knowledge_service.py', '_expand_candidates()', '        expanded = await self.vectors.book_chunks',
                '读取短书片段后核对实际数量与关系表 chunk_count，并按 chunk.id 合并。与原候选重叠时保留融合分数，补取标记跟随候选进入重排，避免重复引用同一片段。',
                '补取发生在重排之前，不在最终排序后偷偷拼接未评分候选。', 9),
            step('最终证据', 'backend/app/services/knowledge_service.py', 'search()', '            selected = qualified[:limit]',
                '先取通过阈值的前 K 项，再附带同一已命中书籍的扩展上下文。扩展上下文可能低于阈值但有明确标记，返回数量可能超过 K；评测不能将所有额外片段算进 Hit@K。',
                '前 K 相关证据与同书邻接上下文需要分别解释。', 10))),
    old['practice-knowledge-recovery'],
    section('迁移、失败与模块总结', 'practice-knowledge-summary',
        text('旧索引只包含稠密向量，因此新增字段不能靠修改 Python schema 自动迁移已有集合。停止所有写入后执行 python -m app.rag.migrate_index，它复制已有片段和向量，逐行计算摘要并校验目标及来源没有变化，烟测混合检索后才发布目标别名。失败不发布，旧集合和暂存集合都保留；发布后应用不双写旧集合，旧集合只是迁移时快照。'),
        chain('python -m app.rag.migrate_index → migrate()',
            step('迁移校验', 'backend/app/rag/migrate_index.py', 'migrate()', '        client.flush(stage)',
                '向量和原文复制后，重新读取目标内容摘要，并与复制时的来源摘要逐 ID 比较，同时再次读取来源检测写入变化。校验后用保存的向量做混合检索烟测，不调用 Embedding 或 rerank 服务。',
                '数量一致不等于内容一致，ID、元数据和 float32 向量都要校验。', 11),
            step('发布', 'backend/app/rag/migrate_index.py', 'migrate()', '        client.create_alias',
                '创建目标别名是最后一步，应用在此之前不会通过该入口看到半成品集合。目标已存在时拒绝覆盖；本机端口检查不是分布式锁，其他写入进程必须由操作者暂停。',
                '不删除旧数据，不把端口检查说成完整的并发迁移方案。', 6)),
        text('纯向量模式保留作基线；仅混合模式不套用余弦阈值，所以通常不会主动弃答；默认 hybrid_rerank 用独立重排阈值。重排服务错误明确失败，不是“无依据”，也不自动退回没有重排的结果。测试见 backend/tests/test_hybrid_retrieval.py；评测报告记录模式、分数类型、返回总数与前 K 指标。')),
]
save('knowledge', k)

ops = read('operations')
for i, sec in enumerate(ops['sections']):
    if sec['anchor'] == 'practice-operations-retrieval-evaluation':
        ops['sections'][i] = section('检索消融：纯向量、混合融合与重排怎样比较', sec['anchor'],
            text('消融实验是在相同语料和标注问题上逐步启用组件，而不是换一批问题比较分数。evaluate.py 新增 --strategy dense/hybrid/hybrid_rerank，Service 使用请求级模式而不改全局配置。Hit@K 和 MRR@K 只统计前 K 项，短书上下文的实际返回总数另记 returned_count，避免指标虚高。'),
            chain('evaluate.main → evaluate → KnowledgeService.search → score_case',
                step('评测', 'backend/app/rag/evaluate.py', 'evaluate()', '            hits = await service.search',
                    '同一用户与书籍范围内计时检索，保留总返回数量后截取前 K 用于评测。负例的正确弃答表示检索结果为空，不等于最终回答模型的弃答指标。',
                    '阶段、范围、阈值和 K 的定义必须随报告一起保存。', 8),
                step('排名指标', 'backend/app/rag/evaluate.py', 'score_case()', '    rank = next(',
                    '寻找第一个正确来源的排名，命中记为 1，倒数排名为 1/rank。多个片段可能属于同一书页或书籍，所以报告保留标注粒度，不能把 MRR 当作事实准确率。',
                    'MRR 衡量正确来源排在多靠前。', 3)),
            table(['2026-10-22 / 15 题', 'Hit@5', 'MRR@5', '检索弃答（5负例）', 'P50/P95 ms'], [
                ['dense', '1.00', '0.6533', '0.60', '271 / 805'],
                ['hybrid', '0.90', '0.6750', '0.00', '286 / 1015'],
                ['hybrid_rerank', '1.00', '1.0000', '1.00', '1004 / 1420'],
            ]),
            text('数据来自 backend/evals/hybrid_ablation_20261022.json 及三份逐题报告。本机 5 本 ready 资料、10 正例和 5 负例，初始重排阈值为 0.30；这不是独立留出集，样本较小，不能承诺泛化结果或零幻觉。融合本身扩大候选而没有使用余弦阈值，重排与最终依据核对是另外两个阶段。'))
save('operations', ops)

# 把旧知识库片段中的摘录更新到同一实际函数，并重新定位其他未变的摘录。
for p in PARTS.glob('*.json'):
    data = json.loads(p.read_text(encoding='utf-8'))
    for sec in data.get('sections', []):
        for block in sec['blocks']:
            if block['type'] != 'callchain':
                continue
            for item in block['callchain']['steps']:
                ref = item['filePath']
                match = re.match(r'^(.*?):(\d+)(?:-\d+)?$', ref)
                if not match:
                    continue
                path = Path(match.group(1))
                source = path.read_text(encoding='utf-8-sig').splitlines()
                item['code'] = item.get('code', '').replace('client.insert(settings.MILVUS_COLLECTION, data=rows)', 'client.insert(self.collection, data=rows)')
                if 'delete_book()' in item.get('functionName', '') and path.name == 'milvus_store.py':
                    new = step('Milvus', 'backend/app/rag/milvus_store.py', '_delete_book()', '            if client.has_collection(self.collection):',
                        '删除当前混合集合中的用户书籍片段，两路索引随实体一起更新。旧集合保留作迁移快照，不继续双写，业务可见性仍由关系表的 ready 书籍决定。',
                        '同一实体同时承载稠密与 BM25 索引，不维护两份正文。', 5)
                    item.update(new)
                snippet = [line.strip() for line in item.get('code', '').splitlines()]
                if snippet:
                    starts = [n for n in range(len(source)-len(snippet)+1)
                              if [line.strip() for line in source[n:n+len(snippet)]] == snippet]
                    if starts:
                        nearest = min(starts, key=lambda n: abs(n+1-int(match.group(2))))
                        item['filePath'] = f'{path.as_posix()}:{nearest+1}'
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

print('Updated hybrid retrieval learning modules.')
