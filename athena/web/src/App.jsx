import { useState } from 'react'

function App() {
  const [activeTab, setActiveTab] = useState('image')
  
  // Image tab state
  const [prompt, setPrompt] = useState('Red apple with a green leaf on a plain background')
  const [negativePrompt, setNegativePrompt] = useState('')
  const [seed, setSeed] = useState(0)
  const [steps, setSteps] = useState(4)
  const [cfg, setCfg] = useState(2.1)
  const [sampler, setSampler] = useState('DPM++ 2M')
  const [controlImagePath, setControlImagePath] = useState('C:/Users/kko8/OneDrive/projects/houdini_snippets/prod/3d/render/athena/ctr_images/05K_apple_canny.jpg')
  const [controlnetStrength, setControlnetStrength] = useState(0.8)
  const [imageUrl, setImageUrl] = useState(null)
  const [statusText, setStatusText] = useState('')
  const [isRendering, setIsRendering] = useState(false)
  
  // Video tab state
  const [videoPrompt, setVideoPrompt] = useState('')
  const [videoNegativePrompt, setVideoNegativePrompt] = useState('')
  const [videoSeed, setVideoSeed] = useState(0)
  const [videoSteps, setVideoSteps] = useState(20)
  const [videoCfg, setVideoCfg] = useState(7.5)
  const [videoFrames, setVideoFrames] = useState(24)
  const [videoFps, setVideoFps] = useState(8)
  const [videoStatusText, setVideoStatusText] = useState('')
  const [isVideoRendering, setIsVideoRendering] = useState(false)

  const handlePickFile = async () => {
    const response = await fetch('/pick-file')
    const data = await response.json()
    if (data.path) {
      setControlImagePath(data.path)
    }
  }

  const handleVideoRender = async () => {
    setIsVideoRendering(true)
    setVideoStatusText('Rendering video...')
    const response = await fetch('/render-video', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: videoPrompt,
        negative_prompt: videoNegativePrompt,
        steps: videoSteps,
        seed: videoSeed,
        cfg: videoCfg,
        frames: videoFrames,
        fps: videoFps
      })
    })
    const data = await response.json()
    setVideoStatusText(`${data.render_time} >> ${data.video_path}`)
    setIsVideoRendering(false)
  }

  const handleRender = async () => {
    setIsRendering(true)
    setStatusText('Rendering...')
    const response = await fetch('/render-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, negative_prompt: negativePrompt, steps, seed, cfg, sampler, control_image_path: controlImagePath, controlnet_strength: controlnetStrength })
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
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'image' ? 'active' : ''}`}
            onClick={() => setActiveTab('image')}
          >
            Image
          </button>
          <button 
            className={`tab ${activeTab === 'video' ? 'active' : ''}`}
            onClick={() => setActiveTab('video')}
          >
            Video
          </button>
        </div>
        
        {activeTab === 'image' && (
          <div className="tab-content">
            <label>
              Positive Prompt
              <textarea className="prompt-positive" value={prompt} onChange={e => setPrompt(e.target.value)} />
            </label>
            <label>
              Negative Prompt
              <textarea className="prompt-negative" value={negativePrompt} onChange={e => setNegativePrompt(e.target.value)} />
            </label>
            <label>
              Seed
              <input type="number" value={seed} onChange={e => setSeed(Number(e.target.value))} />
            </label>
            <label>
              Steps
              <input type="number" value={steps} onChange={e => setSteps(Number(e.target.value))} />
            </label>
            <label>
              CFG
              <input type="number" step="0.1" value={cfg} onChange={e => setCfg(Number(e.target.value))} />
            </label>
            <label>
              Sampler
              <select value={sampler} onChange={e => setSampler(e.target.value)}>
                <option value="DPM++ 2M">DPM++ 2M</option>
                <option value="DPM++ 2M SDE">DPM++ 2M SDE</option>
                <option value="DPM++ 2S a">DPM++ 2S a</option>
                <option value="Euler">Euler</option>
                <option value="Euler A">Euler A</option>
              </select>
            </label>
            <label>
              ControlNet Image
              <div className="input-with-button">
                <input type="text" value={controlImagePath} onChange={e => setControlImagePath(e.target.value)} />
                <button type="button" className="pick-button" onClick={handlePickFile}>Pick</button>
              </div>
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
        )}
        
        {activeTab === 'video' && (
          <div className="tab-content">
            <label>
              Positive Prompt
              <textarea className="prompt-positive" value={videoPrompt} onChange={e => setVideoPrompt(e.target.value)} />
            </label>
            <label>
              Negative Prompt
              <textarea className="prompt-negative" value={videoNegativePrompt} onChange={e => setVideoNegativePrompt(e.target.value)} />
            </label>
            <label>
              Seed
              <input type="number" value={videoSeed} onChange={e => setVideoSeed(Number(e.target.value))} />
            </label>
            <label>
              Steps
              <input type="number" value={videoSteps} onChange={e => setVideoSteps(Number(e.target.value))} />
            </label>
            <label>
              CFG
              <input type="number" step="0.1" value={videoCfg} onChange={e => setVideoCfg(Number(e.target.value))} />
            </label>
            <label>
              Frames
              <input type="number" value={videoFrames} onChange={e => setVideoFrames(Number(e.target.value))} />
            </label>
            <label>
              FPS
              <input type="number" value={videoFps} onChange={e => setVideoFps(Number(e.target.value))} />
            </label>
            <button onClick={handleVideoRender} disabled={isVideoRendering}>
              Render
            </button>
            <div className="status-line">{videoStatusText}</div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
