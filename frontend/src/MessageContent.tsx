import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage } from './api'
import { ArtifactCard } from './ArtifactCard'

/** 输入来源摘录，返回清理后的 text 和 isCode；仅影响展示，不修改存储或检索证据。 */
export function formatSourceExcerpt(excerpt: string) {
  // 只清理展示文本，兼容 PDF 控制字符及其字面转义，不修改知识库原文。
  const text = excerpt.normalize('NFKC')
    .replace(/\\u00(?:0[0-8bcef]|1[0-9a-f])|\\u007f/gi, ' ')
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, ' ')
    .replace(/\r\n?/g, '\n')
    .trim()
  const lines = text.split('\n')
  const codeLines = lines.filter((line) => /^\s*(?:from\s+\w|import\s+\w|async\s*def\b|def\s+|class\s*\w|await\s+|return\b|if\s+|for\s+|#|["'][\w-]+["']\s*:|[\w.]+\s*=)/.test(line))
  const isCode = codeLines.length >= 2
  if (isCode) return { text, isCode }
  // PDF 正文常在行宽处换行；合并软换行，保留空行与独立的小标题。
  const paragraphs = text
    .replace(/\n(?=\s*(?:【|[一二三四五六七八九十]+、))/g, '\n\n')
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
  return { text: paragraphs.join('\n\n'), isCode }
}

/**
 * 渲染单条消息：AI 内容按 Markdown 展示并跳过原始 HTML，用户文字按纯文本显示。
 * 候选来源不展示详情，最终来源默认折叠；展开后按引用编号、书名、位置展示摘录。
 */
export function MessageContent({ message }: {
  message: ChatMessage
}) {
  return <>
    {!!message.content && <div className={`bubble ${message.role === 'assistant' ? 'markdown' : ''}`}>
      {message.role === 'assistant'
        ? <Markdown remarkPlugins={[remarkGfm]} skipHtml>{message.content}</Markdown>
        : message.content}
    </div>}
    {message.role === 'assistant' && message.artifacts?.map((artifact) => <ArtifactCard key={artifact.id} artifact={artifact} />)}
    {message.role === 'assistant' && !!message.content && !!message.sources?.length && !message.sourcesPending && <details className="reference-details">
      <summary><span className="details-show">展示详情</span><span className="details-hide">收起详情</span></summary>
      <div className="sources sources-expanded">
      <div className="sources-heading"><strong>参考文献</strong><span>{message.sources.length} 个片段</span></div>
      <p className="sources-note">以下为检索到的原文摘录，不是完整页面。</p>
      <div className="source-list">
        {message.sources.map((source, index) => {
          const excerpt = formatSourceExcerpt(source.excerpt)
          return <div key={source.chunk_id} className="source">
            <div className="source-heading">
              <span className="source-number">[{index + 1}]</span>
              <span className="source-title">{source.title}</span>
              <span className="source-location">{source.page > 0 ? `第 ${source.page} 页` : `第 ${-source.page} 段`}</span>
            </div>
            <div className="source-detail">
              <span className="source-label">原文摘录</span>
              {excerpt.isCode
                ? <pre className="source-code"><code>{excerpt.text}</code></pre>
                : <div className="source-prose">{excerpt.text || '暂无可显示的摘录。'}</div>}
            </div>
          </div>
        })}
      </div>
      </div>
    </details>}
  </>
}
