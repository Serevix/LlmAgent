import { useEffect, useRef, useState, type FormEvent } from 'react'
import { api, clearToken, getToken, setToken, streamChat, uploadBook, type Book, type ChatMessage, type Conversation } from './api'

function usePath() {
  const [path, setPath] = useState(window.location.pathname)
  useEffect(() => {
    const update = () => setPath(window.location.pathname)
    window.addEventListener('popstate', update)
    return () => window.removeEventListener('popstate', update)
  }, [])
  const navigate = (next: string) => {
    window.history.pushState(null, '', next)
    setPath(next)
  }
  return { path, navigate }
}

function Login({ onSuccess }: { onSuccess: () => void }) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

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

function Chat({ onLogout }: { onLogout: () => void }) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [showBooks, setShowBooks] = useState(false)
  const skipHistory = useRef<number | null>(null)
  const activeRequest = useRef<AbortController | null>(null)
  const bottom = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let alive = true
    api<Conversation[]>('/conversations').then((items) => {
      if (alive) setConversations(items)
    }).catch((cause) => {
      if (alive) setError(cause instanceof Error ? cause.message : '加载会话失败')
    }).finally(() => { if (alive) setLoading(false) })
    return () => { alive = false; activeRequest.current?.abort() }
  }, [])

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

  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function send(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const question = draft.trim()
    if (!question || busy) return
    setDraft('')
    setError('')
    setBusy(true)
    const controller = new AbortController()
    activeRequest.current = controller
    let conversationId = selected
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
      }, (sources) => {
        setMessages((items) => items.map((item, index) => index === items.length - 1 ? { ...item, sources } : item))
      }, controller.signal)
      const refreshed = await api<Conversation[]>('/conversations')
      setConversations(refreshed)
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : '发送失败')
        setMessages((items) => items.at(-1)?.role === 'assistant' && !items.at(-1)?.content ? items.slice(0, -1) : items)
      }
    } finally {
      activeRequest.current = null
      setBusy(false)
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
        {messages.map((item, index) => <div key={index} className={`message ${item.role === 'user' ? 'user' : 'assistant'}`}><span className="role">{item.role === 'user' ? '你' : 'AI'}</span><div className="message-body"><div className="bubble">{item.content || (busy ? '检索并核对书籍依据中…' : '')}</div>{!!item.sources?.length && <div className="sources"><strong>参考书籍</strong>{item.sources.map((source, sourceIndex) => <div key={source.chunk_id} className="source"><b>[{sourceIndex + 1}] {source.title}</b> · {source.page > 0 ? `第 ${source.page} 页` : `第 ${-source.page} 段`}<p>{source.excerpt}</p></div>)}</div>}</div></div>)}
        <div ref={bottom} />
      </div>
      <div className="composer-wrap">
        {error && <p className="error" role="alert">{error}</p>}
        <form className="composer" onSubmit={send}>
          <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder="输入问题…（Shift + Enter 换行）" rows={2} disabled={busy} aria-label="输入问题" />
          <button className="primary" type="submit" disabled={busy || !draft.trim()}>{busy ? '回复中…' : '发送'}</button>
        </form>
      </div>
      {showBooks && <KnowledgePanel onClose={() => setShowBooks(false)} />}
    </section>
  </main>
}

function KnowledgePanel({ onClose }: { onClose: () => void }) {
  const [books, setBooks] = useState<Book[]>([])
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function refresh() {
    try { setBooks(await api<Book[]>('/knowledge/books')) }
    catch (cause) { setError(cause instanceof Error ? cause.message : '加载书籍失败') }
  }

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => { void refresh() }, 3000)
    return () => window.clearInterval(timer)
  }, [])

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!file) return
    setBusy(true); setError('')
    try { await uploadBook(file, title); setFile(null); setTitle(''); await refresh() }
    catch (cause) { setError(cause instanceof Error ? cause.message : '上传失败') }
    finally { setBusy(false) }
  }

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

export default function App() {
  const { path, navigate } = usePath()
  const authenticated = !!getToken()
  useEffect(() => {
    if (!authenticated && path !== '/login') navigate('/login')
    if (authenticated && path !== '/chat') navigate('/chat')
  }, [authenticated, path])
  if (!authenticated) return <Login onSuccess={() => navigate('/chat')} />
  return <Chat onLogout={() => { clearToken(); navigate('/login') }} />
}
