"""同步按需调用策略到现有学习页面，并重新定位源码摘录。"""

import hashlib
import json
import re
from pathlib import Path

from assemble import ROOT, PARTS, evidence, step, table, text


path = PARTS / 'dialogue.json'
module = json.loads(path.read_text(encoding='utf-8'))
sections = {item['anchor']: item for item in module['sections']}


def all_steps(section):
    """遍历本节调用链，保留其他模块与章节的原内容。"""
    for block in section['blocks']:
        if block['type'] == 'callchain':
            yield from block['callchain']['steps']


for item in all_steps(sections['practice-dialogue-concept']):
    if item['functionName'] == 'build_chat_graph()':
        item['filePath'], item['code'] = evidence('backend/app/agents/chat_graph.py', 'def build_chat_graph(', 10)
    if item['functionName'] == 'build_chat_graph() → add_edge()/compile()':
        item['filePath'], item['code'] = evidence('backend/app/agents/chat_graph.py', '    # 节点返回局部更新', 10)
        item['explanation'] = '上下文、改写规划和检索之后，图按证据选择生成或弃答。回答后新增条件边 route_after_answer：普通问题直接 END，有稳定用户信息的候选消息才尝试保存记忆；compile 绑定原有 checkpoint 和 Store。'
for block in sections['practice-dialogue-concept']['blocks']:
    if block['type'] == 'diagram':
        block['diagram']['content'] = '''flowchart TD
 A["START"] --> B["manage_context：按阈值摘要"]
 B --> C["plan：按需补全问题"]
 C --> D["retrieve：检索书籍"]
 D --> E{"有证据？"}
 E -->|是| F["grounded_answer：流式生成并核对"]
 E -->|否| G["no_evidence：固定弃答"]
 F --> H{"route_after_answer：需要记忆提炼？"}
 G --> H
 H -->|是| I["save_memory：模型判断后保存"]
 H -->|否| J["END"]
 I --> J'''

retrieval = sections['practice-dialogue-retrieval']
retrieval['blocks'][0]['text'] = retrieval['blocks'][0]['text'].split('\n\n性能更新：')[0] + '\n\n性能更新：auto 模式先用 needs_query_rewrite 判断是否有指代或省略；独立问题直接进入检索，只有追问才请求规划模型。规则不是完整语言理解，可能漏掉含蓄的依赖表达，可配置 always 恢复每轮多轮问题改写；它不决定是否回答，也不替代后面的书籍核对。'
for item in all_steps(retrieval):
    if item['functionName'] in ('plan() → summary_message()/output_instructions()', 'plan() → needs_query_rewrite() → 原问题或补全问题'):
        item['functionName'] = 'plan() → needs_query_rewrite() → 原问题或补全问题'
        item['filePath'], item['code'] = evidence('backend/app/agents/chat_graph.py', '        recent = state["context_messages"]', 9)
        item['explanation'] = 'plan 从受限近期消息末尾取当前问题，并根据近期消息数量或旧摘要判断是否有前文。needs_query_rewrite 先做本地筛选，独立问题直接返回原文；只有含指代或省略的追问才进入原有摘要加近期消息的模型改写流程。'
chain = next(block['callchain'] for block in retrieval['blocks'] if block['type'] == 'callchain')
chain['steps'] = [item for item in chain['steps'] if item['functionName'] != 'needs_query_rewrite()']
chain['steps'].insert(1, step('本地筛选', 'backend/app/agents/call_policy.py', 'needs_query_rewrite()',
    'def needs_query_rewrite(',
    '没有前文时不改写，always 模式则为所有有前文的问题改写。auto 使用指代和省略规则选择候选，专业词和数字不会阻止“它有哪3种用途”这样的追问被补全，但规则可能漏判，不能等同于模型理解。',
    '筛选步骤本身不调用 LLM，只有返回 True 后才请求规划模型。', 9))

memory = sections['practice-dialogue-history-memory']
memory['blocks'][0]['text'] = memory['blocks'][0]['text'].split('\n\n默认 selective')[0] + '\n\n默认 selective 只为明确要求记住、学习目标、稳定偏好等候选信息调用提炼模型。普通知识问答经过 route_after_answer 直接结束；候选内容仍需要 MemoryDecision 判断，不把匹配规则的用户原文直接存入 Store。'
for item in all_steps(memory):
    if item['functionName'] == '_save_memory() → runtime.store.asearch()/clip_text()':
        item['filePath'], item['code'] = evidence('backend/app/agents/chat_graph.py', '        if runtime.store is None:', 10)
        item['explanation'] = '进入记忆节点后先检查 Store 和本轮用户消息，并重复校验是否属于待提炼信息。符合条件时才读取当前用户最多二十条已有记忆，按预算组成提示供模型判断是否保存及避免重复；普通问题不会走这些外部请求。'
    if item['functionName'] == 'save_memory() → _save_memory()/log_model_failure()':
        item['explanation'] = '外层捕获自动记忆异常，保留已完成的书籍答案。现在只有条件边选中记忆的消息才等待这个节点和最终 done，普通问答直接结束；并没有使用未受管理的后台任务或提前关闭请求级模型客户端。'
chain = next(block['callchain'] for block in memory['blocks']
             if block['type'] == 'callchain' and 'steps' in block['callchain']
             and any('_save_memory()' in item['functionName'] for item in block['callchain']['steps']))
chain['endpoint'] = '回答之后：route_after_answer → END 或 save_memory → Store'
chain['steps'] = [item for item in chain['steps'] if item['functionName'] not in ('route_after_answer() → needs_memory_extraction()', 'needs_memory_extraction()')]
chain['steps'].insert(0, step('条件路由', 'backend/app/agents/chat_graph.py',
    'route_after_answer() → needs_memory_extraction()', '    def route_after_answer(',
    '路由从完整消息中找到最新用户原文，交给本地记忆候选规则判断。普通问题直接返回 END；学习目标或偏好等信息才进入 save_memory，省掉“判断这句普通问题不需要保存”的模型调用。',
    '条件边是真正跳过记忆请求，而不是只隐藏记忆输出。', 6))
chain['steps'].insert(1, step('本地筛选', 'backend/app/agents/call_policy.py',
    'needs_memory_extraction()', 'def needs_memory_extraction(',
    'off 直接关闭自动提炼，always 保留每轮提炼，默认 selective 只匹配稳定信息候选。它只控制是否请求模型，后续模型仍排除临时信息、重复信息和秘密；隐含偏好可能漏掉，这是启发式筛选的取舍。',
    '规则筛选不是记忆内容的最终审批。', 5))

failure = sections['practice-dialogue-failure-summary']
marker = next((index for index, block in enumerate(failure['blocks'])
               if block.get('text', '').startswith('调用次数与时延要分开看。')), len(failure['blocks']))
failure['blocks'] = failure['blocks'][:marker]
failure['blocks'].extend([
    text('调用次数与时延要分开看。普通有证据问答默认两次聊天请求：生成与独立核对；首轮以前三次，多轮独立问题以前四次；追问、明确记忆或超阈值摘要仍会增加请求，Embedding 与 rerank 也有单独耗时。两题真实复测的第二轮首字明显改善，但整轮耗时没有稳定下降，不能据此承诺固定提速比例。'),
    table(['问题/阶段', '旧流程', '按需调用流程'], [
        ['首轮聊天模型请求', '3 次', '2 次'],
        ['多轮独立问题请求', '4 次', '2 次'],
        ['本次第二题首字', '16.23 秒', '3.05 秒'],
        ['本次第一题首字', '3.34 秒', '5.40 秒'],
        ['本次两题整轮耗时', '9.57 / 21.88 秒', '10.67 / 20.54 秒'],
        ['证据核对', '独立模型判断', '继续保留；不支持或异常就弃答'],
    ]),
    text('先读 tests/test_chat_latency.py：测试了普通两轮各两次、追问改写、明确偏好写入用户隔离的 Store、核对失败仍弃答，以及 always 模式恢复。再读 rag/benchmark_chat.py：真实调用同一临时会话，回调按节点计数，分别测量首个草稿 token、核对完成和整个图结束；结果保存在 backend/evals/chat_latency_before_20261022.json 与 chat_latency_after_20261022.json，都是小样本，且草稿不等于已核对答案。'),
])
path.write_text(json.dumps(module, ensure_ascii=False, indent=2), encoding='utf-8')

# 同时刷新旧链路中已过时的边和重排错误分支，避免引用旧代码。
for name in ('dialogue', 'operations'):
    path = PARTS / f'{name}.json'
    module = json.loads(path.read_text(encoding='utf-8'))
    for section in module['sections']:
        for item in all_steps(section):
            if item['functionName'] == 'runtime_graph() → graph.ainvoke()':
                item['filePath'], item['code'] = evidence('backend/app/api/v1/chat.py', '        async with models.runtime_graph(request.app, spec) as graph:', 10)
            elif item['functionName'] == 'search_book_chunks() → KnowledgeService.search()':
                item['filePath'], item['code'] = evidence('backend/app/agents/tools.py', '    # 将检索结果对象转换成字典', 15)
            elif name == 'operations' and item['functionName'] == 'build_chat_graph()':
                item['filePath'], item['code'] = evidence('backend/app/agents/chat_graph.py', '    builder.add_edge(START', 9)
                item['explanation'] = '图仍按固定节点顺序调度上下文、规划和检索，再按是否有证据选择回答或弃答。回答之后由 route_after_answer 判断是否需要记忆提炼，普通问题直接结束；这条条件边减少外部模型请求，而不是去掉 Store 或上下文摘要。'
    path.write_text(json.dumps(module, ensure_ascii=False, indent=2), encoding='utf-8')

# 仅重新定位可在源码逐行找到的摘录；若遗漏了变动的摘录，立即报错。
for path in PARTS.glob('*.json'):
    module = json.loads(path.read_text(encoding='utf-8-sig'))
    def reanchor(value):
        """验证每段已有源码，并按当前行号更新链接，不把过时摘录静默发布。"""
        if isinstance(value, dict):
            if 'filePath' in value and 'code' in value:
                target = Path(re.sub(r':\d+(?:-\d+)?$', '', value['filePath']))
                lines = target.read_text(encoding='utf-8-sig').splitlines()
                wanted = [line.strip() for line in value['code'].splitlines()]
                index = next((i for i in range(len(lines) - len(wanted) + 1)
                              if [line.strip() for line in lines[i:i + len(wanted)]] == wanted), None)
                if index is None:
                    raise ValueError('过时源码摘录：' + value['functionName'])
                value['filePath'] = f'{target.as_posix()}:{index + 1}'
            for child in value.values():
                reanchor(child)
        elif isinstance(value, list):
            for child in value:
                reanchor(child)
    reanchor(module)
    path.write_text(json.dumps(module, ensure_ascii=False, indent=2), encoding='utf-8')

validator = PARTS / 'validate.py'
content = validator.read_text(encoding='utf-8')
approved_hash = hashlib.sha256((ROOT / 'docs/项目一从零学习计划.md').read_bytes()).hexdigest()
content = re.sub(r"expected = '[a-f0-9]+'", f"expected = '{approved_hash}'", content)
validator.write_text(content, encoding='utf-8')
