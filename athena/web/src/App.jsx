import { useState } from 'react'

function App() {
  const [prompt, setPrompt] = useState('Photorealistic portrait of a young athletic man. Photorealistic skin, wrinkles, sss, hairs, aging. Big face with teeth, nice smile, mouth of smile. Rough basic denim trousers. Lether shoes.')
  const [steps, setSteps] = useState(4)
  const [seed, setSeed] = useState(0)
  const [cfg, setCfg] = useState(2.1)
  const [controlnetStrength, setControlnetStrength] = useState(0.8)
  const [imageUrl, setImageUrl] = useState(null)
  const [statusText, setStatusText] = useState('')
  const [isRendering, setIsRendering] = useState(false)

  const handleRender = async () => {
    setIsRendering(true)
    setStatusText('Rendering...')
    const response = await fetch('/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, steps, seed, cfg, controlnet_strength: controlnetStrength })
    })
    const data = await response.json()
    setImageUrl(data.image_url)
    setStatusText(`${data.render_time} >> ${data.image_path}`)
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
        <label>
          CFG
          <input type="number" step="0.1" value={cfg} onChange={e => setCfg(Number(e.target.value))} />
        </label>
        <label>
          ControlNet Strength
          <input type="number" step="0.1" value={controlnetStrength} onChange={e => setControlnetStrength(Number(e.target.value))} />
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
