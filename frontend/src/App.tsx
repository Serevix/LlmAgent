import { useEffect, useRef, useState, type FormEvent } from 'react'
import { api, clearToken, getToken, setToken, streamChat, uploadBook, type Book, type ChatMessage, type Conversation } from './api'
import { MessageContent } from './MessageContent'
import { ModelControls } from './ModelControls'

/** 返回当前路径和 navigate；用 History API 管理 /login、/chat，并响应浏览器前进后退。 */
function usePath() {
  // 两页应用使用浏览器历史切换路径；监听前进/后退使 React 与地址栏一致。
  const [path, setPath] = useState(window.location.pathname)
  useEffect(() => {
    // popstate 回调把地址栏的新路径同步到 React，不发起后端请求。
    const update = () => setPath(window.location.pathname)
    window.addEventListener('popstate', update)
    return () => window.removeEventListener('popstate', update)
  }, [])
  // 应用内跳转更新地址和路径状态；pushState 本身不会触发 popstate。
  const navigate = (next: string) => {
    window.history.pushState(null, '', next)
    setPath(next)
  }
  return { path, navigate }
}

/** 渲染共用的登录/注册页；认证成功通过 onSuccess 通知 App 跳转。 */
function Login({ onSuccess }: { onSuccess: () => void }) {
  // 登录与注册共用表单，成功后保存 token 并进入聊天页。
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  /** 按 Tab 提交认证 JSON，保存响应 JWT；失败显示提示，finally 恢复提交按钮。 */
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      const result = await api<{ access_token: string }>(mode === 'login' ? '/auth/login' : '/auth/register', {
        method: 'POST',
        body: JSON.stringify(mode === 'login' ? { username: username.trim(), password } : {
          username: username.trim(), email: email.trim(), password,
        }),
      })
      setToken(result.access_token)
      onSuccess()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '操作失败')
    } finally {
      setBusy(false)
    }
  }

  return <main className="login-page">
    <section className="login-card">
      <h1>LLM Learning Assistant</h1>
      <p className="subtitle">你的 LangChain / LangGraph 学习助手</p>
      <div className="tabs" role="tablist" aria-label="登录或注册">
        <button type="button" role="tab" aria-selected={mode === 'login'} className={mode === 'login' ? 'selected' : ''} onClick={() => { setMode('login'); setError('') }}>登录</button>
        <button type="button" role="tab" aria-selected={mode === 'register'} className={mode === 'register' ? 'selected' : ''} onClick={() => { setMode('register'); setError('') }}>注册</button>
      </div>
      <form onSubmit={submit}>
        <label>{mode === 'login' ? '用户名 / 邮箱' : '用户名'}
          <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete={mode === 'login' ? 'username' : 'username'} minLength={2} required placeholder={mode === 'login' ? '输入用户名或邮箱' : '至少 2 个字符'} />
        </label>
        {mode === 'register' && <label>邮箱
          <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" required placeholder="name@example.com" />
        </label>}
        <label>密码
          <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} minLength={8} required placeholder="至少 8 个字符" />
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="primary full" disabled={busy} type="submit">{busy ? '请稍候…' : mode === 'login' ? '登录' : '注册并进入'}</button>
      </form>
    </section>
  </main>
}

/** 管理会话列表、历史、输入和流式回复；模型编号独立于会话选择，退出由 App 处理。 */
function Chat({ onLogout }: { onLogout: () => void }) {
  // 页面只保存展示状态；后端根据会话 thread_id 恢复历史，前端不重复上传全部消息。
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [streamStatus, setStreamStatus] = useState('')
  const [showBooks, setShowBooks] = useState(false)
  // 模型选择只影响下一次请求；不修改 selected 会话或 messages，切换不会刷新历史。
  const [selectedModel, setSelectedModel] = useState('default')
  const skipHistory = useRef<number | null>(null)
  const activeRequest = useRef<AbortController | null>(null)
  const bottom = useRef<HTMLDivElement | null>(null)

  // 聊天页挂载后加载会话列表；卸载时忽略晚到的结果，并取消仍在进行的聊天流。
  useEffect(() => {
    let alive = true
    api<Conversation[]>('/conversations').then((items) => {
      if (alive) setConversations(items)
    }).catch((cause) => {
      if (alive) setError(cause instanceof Error ? cause.message : '加载会话失败')
    }).finally(() => { if (alive) setLoading(false) })
    return () => { alive = false; activeRequest.current?.abort() }
  }, [])

  // 切换会话后读取历史；新建会话的首次发送跳过加载，避免覆盖正在显示的流式占位。
  useEffect(() => {
    if (selected === null) { setMessages([]); return }
    if (skipHistory.current === selected) { skipHistory.current = null; return }
    let alive = true
    setMessages([])
    api<{ messages: ChatMessage[] }>(`/chat/${selected}/history`).then((result) => {
      if (alive) setMessages(result.messages.filter((item) => item.role === 'user' || item.role === 'assistant'))
    }).catch((cause) => { if (alive) setError(cause instanceof Error ? cause.message : '加载历史失败') })
    return () => { alive = false }
  }, [selected])

  // 消息更新时滚动到末尾，让新增内容保持可见。
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  /**
   * 读取输入；首次发问先创建会话，再添加用户消息和空 AI 消息作为流式占位。
   * 把会话/问题/选定模型交给 streamChat，各回调更新最后一条 AI 消息。
   * 最终答案覆盖草稿；失败展示错误，取消不当作业务失败，结束后恢复操作状态。
   */
  async function send(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const question = draft.trim()
    if (!question || busy) return
    setDraft('')
    setError('')
    setBusy(true)
    setStreamStatus('正在检索书籍依据…')
    const controller = new AbortController()
    activeRequest.current = controller
    let conversationId = selected
    let answerFinalized = false
    try {
      if (conversationId === null) {
        const created = await api<Conversation>('/conversations', {
          method: 'POST',
          body: JSON.stringify({ title: question.slice(0, 40) }),
        })
        conversationId = created.id
        skipHistory.current = created.id
        setSelected(created.id)
        setConversations((items) => [created, ...items])
      }
      setMessages((items) => [...items, { role: 'user', content: question }, { role: 'assistant', content: '' }])
      await streamChat(conversationId, question, (token) => {
        setMessages((items) => items.map((item, index) => index === items.length - 1 ? { ...item, content: item.content + token } : item))
      }, (sources, pending) => {
        setMessages((items) => items.map((item, index) => index === items.length - 1 ? { ...item, sources, sourcesPending: pending } : item))
      }, controller.signal, (answer) => {
        // 用核对后的最终答案覆盖流式草稿，避免重复拼接或保留被否决的内容。
        answerFinalized = true
        setStreamStatus('')
        setMessages((items) => items.map((item, index) => index === items.length - 1 ? { ...item, content: answer } : item))
      }, setStreamStatus, selectedModel, (artifacts) => {
        setMessages((items) => items.map((item, index) => index === items.length - 1 ? { ...item, artifacts } : item))
      })
      const refreshed = await api<Conversation[]>('/conversations')
      setConversations(refreshed)
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : '发送失败')
        if (!answerFinalized) {
          setMessages((items) => items.at(-1)?.role === 'assistant'
            ? items.map((item, index) => index === items.length - 1
              ? { ...item, content: '本次回答未完成，生成或核对失败，请重试。', sources: [], sourcesPending: false }
              : item)
            : items)
        }
      }
    } finally {
      activeRequest.current = null
      setBusy(false)
      setStreamStatus('')
    }
  }

  return <main className="chat-page">
    <aside className="sidebar">
      <button className="new-chat" disabled={busy} onClick={() => { setSelected(null); setMessages([]); setError('') }}>＋ 新建对话</button>
      <div className="conversation-list" aria-label="会话列表">
        {loading && <p className="muted">加载会话中…</p>}
        {!loading && conversations.length === 0 && <p className="muted">还没有对话</p>}
        {conversations.map((item) => <button key={item.id} disabled={busy} className={`conversation ${selected === item.id ? 'active' : ''}`} onClick={() => { setSelected(item.id); setError('') }} title={item.title}>{item.title}</button>)}
      </div>
    </aside>
    <section className="chat-main">
      <header className="chat-header"><h1>LLM Learning Assistant</h1><div className="header-actions"><button className="logout" onClick={() => setShowBooks(true)}>知识库</button><button className="logout" onClick={onLogout}>退出登录</button></div></header>
      <div className="messages" aria-live="polite">
        {messages.length === 0 && <div className="empty-state"><h2>开始学习</h2><p>提出关于 LangChain、LangGraph、RAG 或 Agent 的问题。</p></div>}
        {messages.map((item, index) => <div key={`${selected}-${index}`} className={`message ${item.role === 'user' ? 'user' : 'assistant'}`}>
          <span className="role">{item.role === 'user' ? '你' : 'AI'}</span>
          <div className="message-body">
            <MessageContent message={item} />
            {busy && item.role === 'assistant' && index === messages.length - 1 && streamStatus && <p className="muted" role="status">{streamStatus}</p>}
          </div>
        </div>)}
        <div ref={bottom} />
      </div>
      <div className="composer-wrap">
        {error && <p className="error" role="alert">{error}</p>}
        <form className="composer" onSubmit={send}>
          <ModelControls selected={selectedModel} onSelect={setSelectedModel} disabled={busy} />
          <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder="输入问题…（Shift + Enter 换行）" rows={2} disabled={busy} aria-label="输入问题" />
          <button className="primary" type="submit" disabled={busy || !draft.trim()}>{busy ? '回复中…' : '发送'}</button>
        </form>
      </div>
      {showBooks && <KnowledgePanel onClose={() => setShowBooks(false)} />}
    </section>
  </main>
}

/** 展示当前用户书籍，支持导入、失败重试和删除；轮询后台任务状态，关闭由父组件处理。 */
function KnowledgePanel({ onClose }: { onClose: () => void }) {
  // 上传接口返回 processing，定时查询状态；202 不代表解析和向量化已经完成。
  const [books, setBooks] = useState<Book[]>([])
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  /** GET 书籍列表并替换 books；处理中的导入会在后续轮询显示 ready/failed。 */
  async function refresh() {
    try { setBooks(await api<Book[]>('/knowledge/books')) }
    catch (cause) { setError(cause instanceof Error ? cause.message : '加载书籍失败') }
  }

  // 挂载即加载，之后每三秒刷新；卸载清理定时器，不继续轮询。
  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => { void refresh() }, 3000)
    return () => window.clearInterval(timer)
  }, [])

  /** 将选定文件和标题交给 uploadBook，清空界面选择并刷新；上传成功不等于索引完成。 */
  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!file) return
    setBusy(true); setError('')
    try { await uploadBook(file, title); setFile(null); setTitle(''); await refresh() }
    catch (cause) { setError(cause instanceof Error ? cause.message : '上传失败') }
    finally { setBusy(false) }
  }

  /** 用 POST 重试或 DELETE 删除指定书籍，成功后刷新列表；共用忙碌状态和错误展示。 */
  async function act(path: string, method: 'POST' | 'DELETE') {
    setBusy(true); setError('')
    try { await api(path, { method }); await refresh() }
    catch (cause) { setError(cause instanceof Error ? cause.message : '操作失败') }
    finally { setBusy(false) }
  }

  return <div className="modal-backdrop" onClick={onClose}>
    <section className="knowledge-panel" role="dialog" aria-modal="true" aria-label="书籍知识库" onClick={(event) => event.stopPropagation()}>
      <div className="panel-heading"><h2>书籍知识库</h2><button className="logout" onClick={onClose}>关闭</button></div>
      <p className="muted">支持 PDF、Markdown、TXT，单文件最多 25 MB。扫描版 PDF 需先做 OCR。</p>
      <form className="book-upload" onSubmit={upload}>
        <input type="file" accept=".pdf,.md,.txt" onChange={(event) => setFile(event.target.files?.[0] ?? null)} aria-label="选择书籍" />
        <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="书名（留空则使用文件名）" maxLength={200} />
        <button className="primary" type="submit" disabled={!file || busy}>导入书籍</button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="book-list">{books.length === 0 && <p className="muted">尚未导入书籍</p>}{books.map((book) => <div className="book-item" key={book.id}><div><strong>{book.title}</strong><p>{book.filename} · {book.status === 'ready' ? `${book.chunk_count} 段已索引` : book.status === 'processing' ? '正在解析与向量化…' : `导入失败：${book.error ?? '未知错误'}`}</p></div><div className="book-actions">{book.status === 'failed' && <button disabled={busy} onClick={() => void act(`/knowledge/books/${book.id}/retry`, 'POST')}>重试</button>}{book.status !== 'processing' && <button disabled={busy} onClick={() => void act(`/knowledge/books/${book.id}`, 'DELETE')}>删除</button>}</div></div>)}</div>
    </section>
  </div>
}

/** 应用入口：根据 JWT 是否存在显示 Login 或 Chat，并同步页面路径；真正授权由后端检查。 */
export default function App() {
  const { path, navigate } = usePath()
  const authenticated = !!getToken()
  // 令地址与登录状态一致；只是页面跳转，不能替代服务端 JWT 验证。
  useEffect(() => {
    if (!authenticated && path !== '/login') navigate('/login')
    if (authenticated && path !== '/chat') navigate('/chat')
  }, [authenticated, path])
  if (!authenticated) return <Login onSuccess={() => navigate('/chat')} />
  return <Chat onLogout={() => { clearToken(); navigate('/login') }} />
}
