import { useState } from 'react'

function App() {
  const [prompt, setPrompt] = useState('a beautiful sunset over mountains')
  const [steps, setSteps] = useState(20)
  const [seed, setSeed] = useState(42)
  const [imageUrl, setImageUrl] = useState(null)
  const [status, setStatus] = useState('idle')

  const handleRender = async () => {
    setStatus('rendering')
    const response = await fetch('/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, steps, seed })
    })
    const data = await response.json()
    setImageUrl(data.image_url)
    setStatus('done')
  }

  return (
    <div className="app">
      <div className="image-panel">
        {imageUrl && <img src={imageUrl} alt="Generated" />}
        {status === 'rendering' && <div className="spinner">Rendering...</div>}
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
        <button onClick={handleRender} disabled={status === 'rendering'}>
          {status === 'rendering' ? 'Rendering...' : 'Render'}
        </button>
      </div>
    </div>
  )
}

export default App
