"""验证本次生成的数据结构、源码引用、覆盖范围与旧文件不变。"""
import hashlib
import json
import re
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
SKILL = Path('C:/Users/19344/.codex/skills/project-learning')
data = json.loads((DOCS / 'project-learning.json').read_text(encoding='utf-8'))
schema = json.loads((SKILL / 'schema/project-learning-schema.json').read_text(encoding='utf-8'))
jsonschema.Draft202012Validator(schema).validate(data)

anchors = []
steps = []
diagrams = []
references = []
groups = {}
errors = []


def inspect_step(item):
    steps.append(item)
    refs = item['filePath']
    match = re.match(r'^(.*?):(\d+)(?:-\d+)?$', refs)
    target = Path(match.group(1)) if match else Path(refs)
    if not target.is_absolute():
        target = ROOT / target
    if not target.is_file():
        errors.append('Missing source: ' + refs)
    elif match and int(match.group(2)) > len(target.read_text(encoding='utf-8-sig').splitlines()):
        errors.append('Invalid line: ' + refs)
    else:
        references.append(refs)
    sentences = re.findall(r'[。！？]', item['explanation'])
    if len(sentences) < 2:
        errors.append('Explanation too short: ' + item.get('functionName', refs))
    if not item.get('keyPoint'):
        errors.append('Missing keyPoint: ' + item.get('functionName', refs))
    for child in item.get('subSteps', []):
        inspect_step(child)


for chapter in data['chapters']:
    for sec in chapter['sections']:
        anchor = sec.get('anchor', '')
        if not anchor or anchor in anchors:
            errors.append('Missing/duplicate anchor: ' + anchor)
        anchors.append(anchor)
        if sec['blocks'][0]['type'] != 'text':
            errors.append('No concept opening: ' + sec['title'])
        if chapter['id'] == 'practice':
            group = sec['groupTitle']
            groups[group] = groups.get(group, 0) + 1
        for block in sec['blocks']:
            if block['type'] == 'callchain':
                for item in block['callchain']['steps']:
                    inspect_step(item)
            if block['type'] == 'endpoint' and ('request' in block['endpoint'] or 'response' in block['endpoint']):
                errors.append('Unwanted API parameter table: ' + sec['title'])
            if block['type'] == 'diagram':
                diagrams.append(block['diagram']['content'])

if any(count < 4 for count in groups.values()):
    errors.append('Too few sections: ' + str(groups))
if '拿到什么输入' in json.dumps(data, ensure_ascii=False):
    errors.append('Contains disallowed wording')
expected = '2acd8973b612d22d3bc2aba52a6bc8ca552590cc16f2b4df35b28764ca535374'
actual = hashlib.sha256((DOCS / '项目一从零学习计划.md').read_bytes()).hexdigest()
if actual != expected:
    errors.append('Learning Markdown differs from the approved current revision')
assert not errors, '\n'.join(errors)
summary = {'chapters': len(data['chapters']), 'sections': len(anchors), 'modules': groups,
           'callchainSteps': len(steps), 'sourceReferences': len(references),
           'diagrams': len(diagrams), 'learningMarkdownRevisionVerified': actual == expected}
print(json.dumps(summary, ensure_ascii=False, indent=2))
