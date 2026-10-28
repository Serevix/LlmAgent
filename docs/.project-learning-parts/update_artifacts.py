"""把图片与 Word 工具同步到既有学习页面，并验证每段源码仍能准确定位。"""

import hashlib
import json
import re
from pathlib import Path

from assemble import PARTS, ROOT, chain, evidence, step, table, text


modules = {name: json.loads((PARTS / f'{name}.json').read_text(encoding='utf-8-sig'))
           for name in ('identity', 'knowledge', 'dialogue', 'models', 'frontend', 'operations')}


def sections(name):
    """按稳定锚点读取已有章节，避免创建重复的学习文档。"""
    return {section['anchor']: section for section in modules[name]['sections']}


def steps(section):
    """遍历本节全部调用步骤，保留其他说明与章节顺序。"""
    for block in section['blocks']:
        if block['type'] == 'callchain':
            yield from block['callchain']['steps']


def append_once(section, marker, blocks):
    """以说明文字作为边界替换本次更新，重复运行不会累计重复内容。"""
    start = next((i for i, block in enumerate(section['blocks'])
                  if block.get('text', '').startswith(marker)), len(section['blocks']))
    section['blocks'] = section['blocks'][:start] + blocks


dialogue = sections('dialogue')
modules['dialogue']['moduleDescription'] = ('本模块把认证后的聊天请求送入 LangGraph，管理上下文与书籍问答，'
    '并在明确的生成需求出现时由模型选择图片或 Word 工具。checkpoint 保留完整历史与附件，Store 保存独立用户记忆；'
    'SSE 传递草稿、核对结果、生成进度和可下载附件。')
for item in steps(dialogue['practice-dialogue-concept']):
    if item['functionName'] == 'build_chat_graph() → add_edge()/compile()':
        item['filePath'], item['code'] = evidence('backend/app/agents/chat_graph.py', '    # 节点返回局部更新', 12)
        item['explanation'] = ('图先执行上下文准备和 plan，再由 route_from_plan 决定书籍检索或直接执行生成工具。'
            '新知识文档先完成检索与依据核对，再进入 run_artifact_tools；工具完成后按需提炼记忆并结束，同一节点不会自动反复调用生成工具。')
for block in dialogue['practice-dialogue-concept']['blocks']:
    if block['type'] == 'diagram':
        block['diagram']['content'] = '''flowchart TD
 A["START"] --> B["manage_context：按阈值摘要"]
 B --> C["plan：普通问题按需改写，生成需求由模型选工具"]
 C -->|书籍问答或新知识 Word| D["retrieve：只读检索，最多尝试两次"]
 C -->|图片或旧答案导出| T["run_artifact_tools：生成图片或 Word"]
 D -->|有证据| F["grounded_answer：流式正文与依据核对"]
 D -->|无证据| G["no_evidence：固定弃答"]
 F -->|有工具计划| T
 G -->|有工具计划，Word 不导出无依据内容| T
 F -->|无工具且需要记忆| M["save_memory：提炼后写 Store"]
 G -->|无工具且需要记忆| M
 F -->|无工具且无需记忆| Z["END"]
 G -->|无工具且无需记忆| Z
 T -->|需要记忆| M
 T -->|无需记忆| Z
 M --> Z'''

for item in steps(dialogue['practice-dialogue-retrieval']):
    if item['functionName'].startswith('plan() → needs_query_rewrite()'):
        item['explanation'] = ('plan 先清空上轮工具计划，生成或导出候选交给模型选择工具，普通问题继续按原规则判断是否改写。'
            '图片和已有答案导出直接进入生成节点，新知识 Word 使用模型提供的独立问题进入书籍检索；这些分支共享当前会话状态。')

for item in steps(dialogue['practice-dialogue-history-memory']):
    if item['functionName'] == 'route_after_answer() → needs_memory_extraction()':
        item['functionName'] = 'route_after_tools() → needs_memory_extraction()'
        item['filePath'], item['code'] = evidence('backend/app/agents/chat_graph.py', '    def route_after_tools(', 6)
        item['explanation'] = ('回答后若有工具计划先进入 run_artifact_tools，完成工具后由这个函数决定是否提炼记忆。'
            '没有工具的普通回答也使用相同的记忆条件，因此新增文件功能没有恢复每轮都请求记忆模型的旧行为。')

append_once(dialogue['practice-dialogue-failure-summary'], '生成工具更新：', [
    text('生成工具更新：项目现在有书籍检索、图片生成和 Word 生成三个业务工具。检索按原工作流固定调用，两个生成工具在明确的生成候选出现时交给模型选择；身份、会话、文件路径与可信正文仍由后端掌控。'),
    chain('自然语言生成需求 → 原生 tool_calls → 可信数据注入 → 文件与附件',
        step('候选筛选', 'backend/app/agents/artifact_planner.py', 'may_need_artifact()', 'def may_need_artifact(',
             '本地规则先判断当前问题是否像图片或文件生成需求，避免普通问答每轮都额外规划工具。命中只是候选，模型还会判断用户是在要求生成，还是仅咨询实现方法。',
             '普通 RAG 路径不必为工具选择增加一次固定调用。', 3),
        step('模型选工具', 'backend/app/agents/artifact_planner.py', 'plan_artifacts() → bind_tools().ainvoke()',
             '        response = await model.bind_tools(',
             '模型读取摘要和近期消息，返回工具名及公开参数，程序再校验数量、名称和参数字段。每轮最多各调用一次图片与 Word 工具，模型输出的隐藏身份参数会被拒绝，选择结果也不会直接执行任意函数。',
             '模型做选择，后端负责执行边界。', 6),
        step('工具定义', 'backend/app/agents/artifact_tools.py', 'generate_word() 与 InjectedToolArg', 'async def generate_word(',
             '模型可以选择标题、latest_answer 或 knowledge 模式，以及新文档的知识检索问题。user_id、thread_id、request_key、content、sources 被标记为注入参数，不出现在提供给模型的工具 schema 中。',
             '模型不能提供服务器路径或替换已核对正文。', 11),
        step('可信执行', 'backend/app/agents/chat_graph.py', 'run_artifact_tools() → 工具.ainvoke()',
             '            injected = {"user_id": runtime.context.user_id,',
             '执行前从运行时身份和会话配置构造可信参数，再把已核对正文及来源注入 Word 工具。后端参数在字典合并的后面，覆盖同名输入；节点不配置自动重试，避免重复收费或重复写入。',
             '工具调用参数来自模型和后端两部分，信任边界要清楚。', 10),
        step('文件业务', 'backend/app/services/artifact_service.py', 'ArtifactService.create() → 仓储与格式服务',
             '    async def create(',
             '生成前先按用户、会话和请求键查已有结果，随后调用图片接口或本地 Word 排版，保存文件与元数据。文件路径由服务器生成，公开结果只返回文件编号、名称、类型和下载地址，生成失败时执行本次文件清理。',
             '文件与关系记录分开保存，同一请求优先复用已有结果。', 14),
        step('受保护下载', 'backend/app/api/v1/files.py', 'file_content() → get_owned()/FileResponse',
             '    assert current_user.id is not None',
             '下载接口使用认证用户查询文件所有者，再验证实际路径属于生成目录并确认文件仍存在。浏览器取得附件编号后仍需携带 Bearer 认证，不能仅凭一个文件 ID 下载其他用户的结果。',
             '附件元数据不是公共下载授权。', 13)),
    table(['分支', '正文或提示词来源', '失败时如何处理'], [
        ['生成图片', '模型根据当前用户需求填写 prompt；独立硅基流动图片模型执行', '生成或下载失败返回安全错误，不自动重复收费请求'],
        ['导出旧答案', '最近一条 AIMessage 中的已核对正文与来源', '没有可导出答案则明确提示，不回头寻找更早的不相关回答'],
        ['新知识 Word', '当前检索结果 → 生成正文 → 独立书籍依据核对', '无依据或核对失败继续弃答，不生成知识文档'],
        ['历史与前端', 'AIMessage.additional_kwargs.artifacts → 历史响应或 SSE artifacts', '下载失败就地显示错误；不把 token 写入 URL'],
    ]),
    text('实际接口烟测已生成图片、导出旧答案，并为自然语言数据预处理主题生成了带五个来源的 Word；同时保留了知识依据不足而未生成文档的样例。报告在 backend/evals/artifact_tools_smoke_20261022.json，浏览器图片预览、认证下载、SSE 附件及历史恢复验证在 artifact_tools_ui_20261022.json；这些是功能烟测，不能代替文档事实准确率评测。'),
])

frontend = sections('frontend')
append_once(frontend['practice-frontend-render'], '生成附件更新：', [
    text('生成附件更新：MessageContent 现在还遍历 message.artifacts 渲染 ArtifactCard。图片通过认证接口读取为 Blob 后预览，Word 点击按钮下载；刷新后从历史消息恢复同样的附件描述，不依赖临时的服务商图片 URL。'),
    chain('SSE artifacts / 历史附件 → ArtifactCard → Bearer 请求 → Blob',
        step('认证下载', 'frontend/src/api.ts', 'artifactBlob()', 'export async function artifactBlob(',
             '浏览器按附件 ID 请求后端文件内容，并在 Authorization 中携带现有登录凭据。下载地址不拼接 token，也不直接访问模型返回的外部地址，错误统一显示中文提示。',
             '认证文件通过 fetch 读取，不能直接把私有地址当作公开链接。', 10),
        step('消息附件', 'frontend/src/ArtifactCard.tsx', 'ArtifactCard() → useEffect()', 'export function ArtifactCard(',
             '图片卡片挂载后读取文件并创建 Blob URL，组件卸载时取消尚未完成的请求并释放地址。Word 按钮调用 download 读取文件，浏览器再根据附件文件名触发保存，前端不参与正文生成。',
             'Blob URL 属于页面资源，需要在卸载后释放。', 13),
        step('点击保存', 'frontend/src/ArtifactCard.tsx', 'download()', '  async function download(',
             '下载时可复用已加载的图片预览，Word 则在点击后取得文件并创建临时链接。浏览器开始下载后释放临时地址，失败只影响该附件提示，不清空会话历史。',
             '界面只负责展示与下载，不保存服务端密钥。', 14)),
])

operations = sections('operations')
append_once(operations['practice-operations-foundations'], '生成文件存储更新：', [
    text('生成文件存储更新：新增 generated_artifacts 业务表，当前共有用户、会话、书籍、模型配置、生成文件五张业务表。checkpoint 与 Store 由 LangGraph 管理，图片和 Word 的实际字节保存在 backend/data/generated，下载接口按记录里的 user_id 校验归属。'),
    table(['数据', '存放位置', '用途'], [
        ['id、user_id、thread_id、request_key', 'generated_artifacts', '关联用户和会话，同一次请求复用文件'],
        ['filename、media_type、size_bytes、storage_path', 'generated_artifacts', '展示附件、返回正确下载类型、定位文件'],
        ['PNG/JPEG/WEBP 或 docx 字节', 'backend/data/generated/<user_id>', '只经认证接口访问，生成目录不提交 Git'],
        ['artifacts 公开描述', 'AIMessage.additional_kwargs 与 checkpoint', '聊天流与历史恢复附件，不包含服务器路径'],
    ]),
])
models = sections('models')
append_once(models['practice-models-boundaries'], '生成工具模型边界：', [
    text('生成工具模型边界：选择工具仍使用当前聊天模型，因此该模型必须支持原生工具调用，JSON 输出能力不能替代 tool_calls。真正的图片生成由 IMAGE_MODEL 与独立服务端图片凭据配置决定，切换聊天模型不会切换图片账号；Word 格式转换使用 python-docx，不额外请求一个写文件模型。'),
])

for module in modules.values():
    for section in module['sections']:
        for block in section['blocks']:
            if 'text' in block:
                block['text'] = block['text'].replace('PostgreSQL 的四张业务表', 'PostgreSQL 的五张业务表')
        for item in steps(section):
            if item['functionName'] == 'event_stream() → answer/sources/done 事件':
                item['functionName'] = 'event_stream() → answer/sources/artifacts/done 事件'
            if item['functionName'] == 'streamChat() → onToken()/onAnswer()/onSources()/onStatus()':
                item['functionName'] = 'streamChat() → onToken()/onAnswer()/onSources()/onStatus()/onArtifacts()'


def reanchor(value):
    """定位已有摘录；源码变动时从对应函数或语句开头更新相同长度的摘录。"""
    if isinstance(value, dict):
        if 'filePath' in value and 'code' in value:
            target = Path(re.sub(r':\d+(?:-\d+)?$', '', value['filePath']))
            lines = target.read_text(encoding='utf-8-sig').splitlines()
            old = [line.strip() for line in value['code'].splitlines()]
            start = next((i for i in range(len(lines) - len(old) + 1)
                          if [line.strip() for line in lines[i:i + len(old)]] == old), None)
            if start is None:
                first = old[0]
                if first.startswith('for node in ('):
                    first = 'for node in ('
                    start = next(i for i, line in enumerate(lines) if line.strip().startswith(first))
                else:
                    start = next(i for i, line in enumerate(lines) if line.strip() == first)
                value['code'] = '\n'.join(lines[start:start + len(old)])
            value['filePath'] = f'{target.as_posix()}:{start + 1}'
        for child in value.values():
            reanchor(child)
    elif isinstance(value, list):
        for child in value:
            reanchor(child)


for name, module in modules.items():
    reanchor(module)
    (PARTS / f'{name}.json').write_text(json.dumps(module, ensure_ascii=False, indent=2), encoding='utf-8')

validator = PARTS / 'validate.py'
approved = hashlib.sha256((ROOT / 'docs/项目一从零学习计划.md').read_bytes()).hexdigest()
content = validator.read_text(encoding='utf-8')
validator.write_text(re.sub(r"expected = '[a-f0-9]+'", f"expected = '{approved}'", content), encoding='utf-8')
print('Generation tools and source references synchronized.')
