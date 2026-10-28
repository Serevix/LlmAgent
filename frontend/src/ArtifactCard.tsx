import { useEffect, useState } from 'react'
import { artifactBlob, type Artifact } from './api'

/** 通过认证读取图片预览或下载 Word，组件卸载时释放浏览器临时文件地址。 */
export function ArtifactCard({ artifact }: { artifact: Artifact }) {
  const [preview, setPreview] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (artifact.kind !== 'image') return
    const controller = new AbortController()
    let url = ''
    artifactBlob(artifact.id, controller.signal).then((blob) => {
      if (!controller.signal.aborted) {
        url = URL.createObjectURL(blob)
        setPreview(url)
      }
    }).catch((cause) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : '图片加载失败')
    })
    return () => { controller.abort(); if (url) URL.revokeObjectURL(url) }
  }, [artifact.id, artifact.kind])

  /** 点击后读取私有附件并触发保存；图片可以复用已经取得的预览数据。 */
  async function download() {
    setBusy(true)
    setError('')
    let url = ''
    try {
      url = preview || URL.createObjectURL(await artifactBlob(artifact.id))
      const link = document.createElement('a')
      link.href = url
      link.download = artifact.filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      // 延迟释放，让浏览器有机会开始下载；预览地址由 effect 管理。
      if (!preview) { const downloadUrl = url; window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000) }
    } catch (cause) {
      if (url && !preview) URL.revokeObjectURL(url)
      setError(cause instanceof Error ? cause.message : '下载失败')
    } finally {
      setBusy(false)
    }
  }

  return <section className="artifact-card" aria-label={artifact.kind === 'image' ? '生成图片' : 'Word 附件'}>
    {artifact.kind === 'image' && (preview
      ? <img className="artifact-preview" src={preview} alt={artifact.title} />
      : !error && <p className="muted">正在加载图片…</p>)}
    <div className="artifact-info">
      <div><strong>{artifact.title}</strong><p className="muted">{artifact.kind === 'image' ? 'AI 生成图片' : 'Word 文档'} · {Math.max(1, Math.round(artifact.size_bytes / 1024))} KB</p></div>
      <button type="button" className="artifact-download" disabled={busy} onClick={() => void download()}>{busy ? '下载中…' : artifact.kind === 'image' ? '下载图片' : '下载 Word'}</button>
    </div>
    {error && <p className="error" role="alert">{error}</p>}
  </section>
}
