import { useState } from 'react'

function App() {
  const [prompt, setPrompt] = useState('Red apple on a white background')
  const [steps, setSteps] = useState(20)
  const [seed, setSeed] = useState(42)
  const [imageUrl, setImageUrl] = useState(null)
  const [statusText, setStatusText] = useState('')
  const [isRendering, setIsRendering] = useState(false)

  const handleRender = async () => {
    setIsRendering(true)
    setStatusText('Rendering...')
    const response = await fetch('/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, steps, seed })
    })
    const data = await response.json()
    setImageUrl(data.image_url)
    setStatusText(`Image rendered: ${data.image_path}`)
    setIsRendering(false)
  }

  return (
    <div className="app">
      <div className="image-panel">
        {imageUrl && <img src={imageUrl} alt="Generated" />}
      </div>
      <div className="controls">
        <label>
          Prompt
          <textarea value={prompt} onChange={e => setPrompt(e.target.value)} />
        </label>
        <label>
          Steps
          <input type="number" value={steps} onChange={e => setSteps(Number(e.target.value))} />
        </label>
        <label>
          Seed
          <input type="number" value={seed} onChange={e => setSeed(Number(e.target.value))} />
        </label>
        <button onClick={handleRender} disabled={isRendering}>
          Render
        </button>
        <div className="status-line">{statusText}</div>
      </div>
    </div>
  )
}

export default App
