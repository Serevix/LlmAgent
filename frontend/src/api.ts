export type Conversation = {
  id: number
  title: string
  updated_at: string
}

export type Source = { book_id: string; title: string; page: number; chunk_id: string; score: number; excerpt: string
  score_type?: string; dense_score?: number | null; fusion_score?: number | null
  rerank_score?: number | null; is_expansion?: boolean }
export type Artifact = { id: string; kind: 'image' | 'word'; title: string; filename: string
  media_type: string; size_bytes: number; download_url: string }
export type ChatMessage = { role: string; content: string; sources?: Source[]; sourcesPending?: boolean; artifacts?: Artifact[] }
export type Book = { id: string; title: string; filename: string; status: string; chunk_count: number; error: string | null; created_at: string }
export type ModelOption = {
  id: string; name: string; model: string; kind: 'builtin' | 'custom'; base_url?: string
  context_window: number; max_output_tokens: number; temperature?: number | null
  top_p?: number | null; timeout: number; structured_output_method: string; has_api_key: boolean
}

const base = import.meta.env.VITE_API_BASE_URL ?? ''
const tokenKey = 'llm_learning_token'

/** 读取登录凭据；App 用它选择页面，请求函数用它生成 Bearer 认证头。 */
export const getToken = () => localStorage.getItem(tokenKey)
/** 登录/注册成功后保存 JWT；模型 API Key 不使用这个存储入口。 */
export const setToken = (token: string) => localStorage.setItem(tokenKey, token)
/** 退出登录或接口返回无效身份时清除 JWT，不删除服务端会话历史。 */
export const clearToken = () => localStorage.removeItem(tokenKey)

/** 执行 HTTP 请求并返回原始响应；连接失败转中文提示，主动取消保留原异常。 */
async function fetchResponse(url: string, options: RequestInit): Promise<Response> {
  try {
    return await fetch(url, options)
  } catch (cause) {
    // 浏览器的网络异常未经过后端，统一转换成中文；主动取消请求保留原异常。
    if (!options.signal?.aborted && cause instanceof TypeError) {
      throw new Error('无法连接服务，请确认前后端已启动并检查网络。')
    }
    throw cause
  }
}

/** 从后端 detail 字符串或验证错误列表提取提示；没有可读信息则显示状态码。 */
function errorMessage(status: number, data: unknown): string {
  if (data && typeof data === 'object' && 'detail' in data) {
    const detail = (data as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) return detail.map((item) => item.msg ?? '输入无效').join('；')
  }
  return `请求失败（${status}）`
}

/** 调用 /api/v1 下的 JSON 接口，返回 T；失败抛 Error，204 返回 undefined。 */
export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  // 普通接口统一携带登录 token；解析后端业务错误与浏览器连接错误。
  const headers = new Headers(options.headers)
  headers.set('Content-Type', 'application/json')
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetchResponse(`${base}/api/v1${path}`, { ...options, headers })
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    if (response.status === 401 && path !== '/auth/login') clearToken()
    throw new Error(errorMessage(response.status, data))
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

/** 将文件和可选标题作为 multipart 上传，返回书籍记录；后台索引状态由列表查询。 */
export async function uploadBook(file: File, title: string): Promise<Book> {
  // 上传使用 FormData，浏览器自动添加 multipart 边界，不能手写 JSON Content-Type。
  const body = new FormData()
  body.append('file', file)
  if (title.trim()) body.append('title', title.trim())
  const response = await fetchResponse(`${base}/api/v1/knowledge/books`, {
    method: 'POST', body,
    headers: { Authorization: `Bearer ${getToken() ?? ''}` },
  })
  if (!response.ok) throw new Error(errorMessage(response.status, await response.json().catch(() => null)))
  return response.json() as Promise<Book>
}

/** 文件与图片都使用 Bearer 下载；不把 token 放进 URL，也不直接打开第三方地址。 */
export async function artifactBlob(id: string, signal?: AbortSignal): Promise<Blob> {
  const response = await fetchResponse(`${base}/api/v1/files/${encodeURIComponent(id)}/content`, {
    headers: { Authorization: `Bearer ${getToken() ?? ''}` }, signal,
  })
  if (!response.ok) {
    if (response.status === 401) clearToken()
    throw new Error(errorMessage(response.status, await response.json().catch(() => null)))
  }
  return response.blob()
}

/**
 * 把会话编号、问题和模型编号发送到流式接口；signal 用于取消请求。
 * 解析 SSE 后分别调用 onToken（追加草稿）、onAnswer（覆盖最终答案）、
 * onSources（候选/最终来源）和 onStatus（处理进度），由 Chat 更新页面状态。
 * 以 done 判断流程完整结束；后端 error 或连接提前中断会抛 Error。
 */
export async function streamChat(
  conversationId: number,
  message: string,
  onToken: (token: string) => void,
  onSources: (sources: Source[], pending: boolean) => void,
  signal: AbortSignal,
  onAnswer: (answer: string) => void,
  onStatus: (status: string) => void,
  modelId: string = 'default',
  onArtifacts: (artifacts: Artifact[]) => void = () => {},
): Promise<void> {
  const response = await fetchResponse(`${base}/api/v1/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${getToken() ?? ''}`,
    },
    body: JSON.stringify({ conversation_id: conversationId, message, model_id: modelId }),
    signal,
  })
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    if (response.status === 401) clearToken()
    throw new Error(errorMessage(response.status, data))
  }
  if (!response.body) throw new Error('浏览器不支持流式响应')

  const reader = response.body.getReader()
  // 一个网络读取块可能包含半个或多个 SSE 事件；缓存到双换行后才解析。
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
        const parsed = JSON.parse(data) as { type: string; content?: string; sources?: Source[]; pending?: boolean; artifacts?: Artifact[] }
        if (parsed.type === 'token') onToken(parsed.content ?? '')
        if (parsed.type === 'answer') onAnswer(parsed.content ?? '')
        if (parsed.type === 'status') onStatus(parsed.content ?? '')
        if (parsed.type === 'sources') onSources(parsed.sources ?? [], parsed.pending === true)
        if (parsed.type === 'artifacts') onArtifacts(parsed.artifacts ?? [])
        if (parsed.type === 'error') throw new Error(parsed.content ?? '生成回复失败')
        if (parsed.type === 'done') done = true
      }
      boundary = buffer.indexOf('\n\n')
    }
    if (chunk.done) break
  }
  if (!done) throw new Error('连接中断，回复可能不完整')
}
