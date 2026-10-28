import { useEffect, useState, type ChangeEvent, type FormEvent } from 'react'
import { createPortal } from 'react-dom'
import { api, type ModelOption } from './api'

type ModelForm = {
  name: string; model: string; base_url: string; api_key: string
  context_window: number; max_output_tokens: number; temperature: string; top_p: string
  timeout: number; structured_output_method: string
}

/** 生成一份独立的空表单；初始化或关闭/保存后使用，清除界面中的明文密钥。 */
const emptyForm = (): ModelForm => ({
  name: '', model: '', base_url: '', api_key: '', context_window: 32768,
  max_output_tokens: 2048, temperature: '', top_p: '', timeout: 60,
  structured_output_method: 'function_calling',
})

/** 验证 JSON 对象及字段，补齐默认值并返回表单；不执行文件，也不自动保存配置。 */
export function parseModelConfig(text: string): ModelForm {
  // 导入的是 API 参数 JSON，不执行文件内容，也不接收模型权重。
  const value: unknown = JSON.parse(text)
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('模型参数文件必须是 JSON 对象。')
  const data = value as Record<string, unknown>
  const allowed = Object.keys(emptyForm())
  if (Object.keys(data).some((key) => !allowed.includes(key))) throw new Error('参数包含未知字段，请参考示例格式。')
  if (typeof data.model !== 'string' || typeof data.base_url !== 'string') throw new Error('参数文件必须包含 model 和 base_url。')
  if (data.api_key !== undefined && typeof data.api_key !== 'string') throw new Error('api_key 必须是字符串。')
  // 从导入对象读取数值，缺失时使用默认值；非有限数字立即拒绝。
  const numeric = (key: string, fallback: number): number => {
    const result = data[key] === undefined ? fallback : Number(data[key])
    if (!Number.isFinite(result)) throw new Error(`${key} 必须是有效数字。`)
    return result
  }
  return {
    name: typeof data.name === 'string' ? data.name : data.model,
    model: data.model, base_url: data.base_url, api_key: data.api_key as string ?? '',
    context_window: numeric('context_window', 32768), max_output_tokens: numeric('max_output_tokens', 2048),
    temperature: data.temperature == null ? '' : String(numeric('temperature', 0)),
    top_p: data.top_p == null ? '' : String(numeric('top_p', 1)), timeout: numeric('timeout', 60),
    structured_output_method: typeof data.structured_output_method === 'string' ? data.structured_output_method : 'function_calling',
  }
}

/** 展示模型选择与参数管理面板；onSelect 仅修改父组件模型编号，不修改会话历史。 */
export function ModelControls({ selected, onSelect, disabled }: {
  selected: string; onSelect: (id: string) => void; disabled: boolean
}) {
  const [models, setModels] = useState<ModelOption[]>([])
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState<ModelForm>(emptyForm)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  // 首次挂载获取内置及当前用户配置；卸载后不再把异步结果写入组件状态。
  useEffect(() => {
    let alive = true
    api<ModelOption[]>('/models').then((items) => { if (alive) setModels(items) })
      .catch((cause) => { if (alive) setError(cause instanceof Error ? cause.message : '加载模型列表失败') })
    return () => { alive = false }
  }, [])

  /** 更新指定表单字段，并清除上次测试/导入提示；不发送请求。 */
  function update<K extends keyof ModelForm>(key: K, value: ModelForm[K]) {
    setForm((current) => ({ ...current, [key]: value }))
    setNotice('')
  }

  /** 空闲时关闭面板并清空表单凭据；请求进行中保持面板，防止中途丢失操作状态。 */
  function close() {
    if (busy) return
    setOpen(false)
    setForm(emptyForm()) // 关闭后从界面状态中清除密钥，不写入浏览器持久存储。
    setError(''); setNotice('')
  }

  /** 把界面字符串转换为 API 参数：裁剪名称/地址空白，可选采样值为空则传 null。 */
  function parameters() {
    return { ...form, name: form.name.trim(), model: form.model.trim(), base_url: form.base_url.trim(),
      temperature: form.temperature === '' ? null : Number(form.temperature),
      top_p: form.top_p === '' ? null : Number(form.top_p) }
  }

  /** 阻止表单默认动作及事件冒泡，保存配置后加入列表、选中它并清空表单。 */
  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); event.stopPropagation()
    setBusy(true); setError(''); setNotice('')
    try {
      const created = await api<ModelOption>('/models', { method: 'POST', body: JSON.stringify(parameters()) })
      setModels((items) => [...items, created])
      onSelect(created.id)
      setForm(emptyForm()); setOpen(false)
    } catch (cause) { setError(cause instanceof Error ? cause.message : '保存模型失败') }
    finally { setBusy(false) }
  }

  /** 将当前参数交给 /models/test 检查流式与结构化输出；不保存配置或聊天记录。 */
  async function test() {
    setBusy(true); setError(''); setNotice('')
    try {
      const result = await api<{ message: string }>('/models/test', { method: 'POST', body: JSON.stringify(parameters()) })
      setNotice(result.message)
    } catch (cause) { setError(cause instanceof Error ? cause.message : '连接测试失败') }
    finally { setBusy(false) }
  }

  /** 读取至多 64 KB 的 JSON 到表单；重置文件控件允许重选，不自动发送密钥。 */
  async function importFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setError(''); setNotice('')
    try {
      if (file.size > 64 * 1024) throw new Error('参数文件不能超过 64 KB。')
      setForm(parseModelConfig(await file.text()))
      setNotice('参数已导入，请确认后保存。')
    } catch (cause) { setError(cause instanceof SyntaxError ? 'JSON 格式不正确。' : cause instanceof Error ? cause.message : '导入参数失败') }
  }

  /** 删除指定自定义配置并移出列表；删除当前选项时回到默认模型，历史不受影响。 */
  async function remove(model: ModelOption) {
    setBusy(true); setError(''); setNotice('')
    try {
      await api(`/models/${model.id}`, { method: 'DELETE' })
      setModels((items) => items.filter((item) => item.id !== model.id))
      if (selected === model.id) onSelect('default')
    } catch (cause) { setError(cause instanceof Error ? cause.message : '删除模型失败') }
    finally { setBusy(false) }
  }

  return <>
    <div className="model-controls">
      <label>模型<select aria-label="选择回答模型" disabled={disabled || busy} value={selected} onChange={(event) => onSelect(event.target.value)}>
        {models.length ? models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>) : <option value="default">默认模型</option>}
      </select></label>
      <button type="button" className="model-manage" disabled={disabled || busy} onClick={() => { setOpen(true); setNotice('') }}>添加 / 管理模型</button>
    </div>
    {open && createPortal(<div className="modal-backdrop" onClick={(event) => { event.stopPropagation(); close() }}>
      <section className="knowledge-panel model-panel" role="dialog" aria-modal="true" aria-label="模型配置" onClick={(event) => event.stopPropagation()}>
        <div className="panel-heading"><h2>模型配置</h2><button type="button" className="logout" disabled={busy} onClick={close}>关闭</button></div>
        <p className="muted">填写兼容 OpenAI 的模型服务参数，或导入 JSON 配置。支持云端及已部署的本地模型。</p>
        <form className="model-form" onSubmit={save}>
          <label>显示名称<input value={form.name} onChange={(event) => update('name', event.target.value)} maxLength={100} required placeholder="例如：我的学习模型" disabled={busy} /></label>
          <label>模型标识<input value={form.model} onChange={(event) => update('model', event.target.value)} maxLength={200} required placeholder="填写服务商提供的 model 名称" disabled={busy} /></label>
          <label className="model-wide">接口地址<input type="url" value={form.base_url} onChange={(event) => update('base_url', event.target.value)} required placeholder="例如：http://localhost:11434/v1" disabled={busy} /></label>
          <label className="model-wide">API Key<input type="password" value={form.api_key} onChange={(event) => update('api_key', event.target.value)} autoComplete="off" placeholder="无需鉴权的本地服务可留空" disabled={busy} /></label>
          <label>上下文窗口（token）<input type="number" min={4096} max={2000000} value={form.context_window} onChange={(event) => update('context_window', Number(event.target.value))} required disabled={busy} /></label>
          <label>最大输出（token）<input type="number" min={128} max={32768} value={form.max_output_tokens} onChange={(event) => update('max_output_tokens', Number(event.target.value))} required disabled={busy} /></label>
          <label>温度（可选）<input type="number" min={0} max={2} step="0.1" value={form.temperature} onChange={(event) => update('temperature', event.target.value)} placeholder="留空使用模型默认值" disabled={busy} /></label>
          <label>Top P（可选）<input type="number" min={0.01} max={1} step="0.01" value={form.top_p} onChange={(event) => update('top_p', event.target.value)} placeholder="留空使用模型默认值" disabled={busy} /></label>
          <label>超时（秒）<input type="number" min={10} max={120} value={form.timeout} onChange={(event) => update('timeout', Number(event.target.value))} required disabled={busy} /></label>
          <label>结构化输出方式<select value={form.structured_output_method} onChange={(event) => update('structured_output_method', event.target.value)} disabled={busy}>
            <option value="function_calling">工具调用（推荐）</option><option value="json_schema">JSON Schema</option><option value="json_mode">JSON 模式</option>
          </select></label>
          <label className="model-wide">导入参数文件<input type="file" accept=".json,application/json" onChange={(event) => void importFile(event)} disabled={busy} /></label>
          <details className="model-wide model-example"><summary>JSON 格式示例</summary><pre>{JSON.stringify({ name: '本地模型', model: '填写模型名称', base_url: 'http://localhost:11434/v1', api_key: '', context_window: 32768, max_output_tokens: 2048, structured_output_method: 'function_calling' }, null, 2)}</pre></details>
          <p className="muted model-wide">窗口大小请按模型实际规格填写。测试连接会调用模型，可能产生少量费用。</p>
          {error && <p className="error model-wide" role="alert">{error}</p>}
          {notice && <p className="model-notice model-wide" role="status">{notice}</p>}
          <div className="model-wide model-actions"><button type="button" className="model-secondary" disabled={busy} onClick={() => void test()}>测试连接</button><button className="primary" type="submit" disabled={busy}>{busy ? '处理中…' : '保存并选择'}</button></div>
        </form>
        <div className="model-saved"><h3>已保存的自定义模型</h3>{models.filter((model) => model.kind === 'custom').map((model) => <div className="book-item" key={model.id}>
          <div><span>{model.name}</span><p>{model.model}</p></div><button type="button" className="logout" disabled={busy} onClick={() => void remove(model)}>删除</button>
        </div>)}{!models.some((model) => model.kind === 'custom') && <p className="muted">尚未添加自定义模型。</p>}</div>
      </section>
    </div>, document.body)}
  </>
}
