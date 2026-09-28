export type Conversation = {
  id: number
  title: string
  updated_at: string
}

export type Source = { book_id: string; title: string; page: number; chunk_id: string; score: number; excerpt: string }
export type ChatMessage = { role: string; content: string; sources?: Source[] }
export type Book = { id: string; title: string; filename: string; status: string; chunk_count: number; error: string | null; created_at: string }

const base = import.meta.env.VITE_API_BASE_URL ?? ''
const tokenKey = 'llm_learning_token'

export const getToken = () => localStorage.getItem(tokenKey)
export const setToken = (token: string) => localStorage.setItem(tokenKey, token)
export const clearToken = () => localStorage.removeItem(tokenKey)

function errorMessage(status: number, data: unknown): string {
  if (data && typeof data === 'object' && 'detail' in data) {
    const detail = (data as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) return detail.map((item) => item.msg ?? '输入无效').join('；')
  }
  return `请求失败（${status}）`
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers)
  headers.set('Content-Type', 'application/json')
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(`${base}/api/v1${path}`, { ...options, headers })
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    if (response.status === 401 && path !== '/auth/login') clearToken()
    throw new Error(errorMessage(response.status, data))
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export async function uploadBook(file: File, title: string): Promise<Book> {
  const body = new FormData()
  body.append('file', file)
  if (title.trim()) body.append('title', title.trim())
  const response = await fetch(`${base}/api/v1/knowledge/books`, {
    method: 'POST', body,
    headers: { Authorization: `Bearer ${getToken() ?? ''}` },
  })
  if (!response.ok) throw new Error(errorMessage(response.status, await response.json().catch(() => null)))
  return response.json() as Promise<Book>
}

export async function streamChat(
  conversationId: number,
  message: string,
  onToken: (token: string) => void,
  onSources: (sources: Source[]) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`${base}/api/v1/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${getToken() ?? ''}`,
    },
    body: JSON.stringify({ conversation_id: conversationId, message }),
    signal,
  })
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    if (response.status === 401) clearToken()
    throw new Error(errorMessage(response.status, data))
  }
  if (!response.body) throw new Error('浏览器不支持流式响应')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let done = false
  while (!done) {
    const chunk = await reader.read()
    buffer += decoder.decode(chunk.value, { stream: !chunk.done }).replace(/\r\n/g, '\n')
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const event = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const data = event.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n')
      if (data) {
        const parsed = JSON.parse(data) as { type: string; content?: string; sources?: Source[] }
        if (parsed.type === 'token') onToken(parsed.content ?? '')
        if (parsed.type === 'sources') onSources(parsed.sources ?? [])
        if (parsed.type === 'error') throw new Error(parsed.content ?? '生成回复失败')
        if (parsed.type === 'done') done = true
      }
      boundary = buffer.indexOf('\n\n')
    }
    if (chunk.done) break
  }
  if (!done) throw new Error('连接中断，回复可能不完整')
}
