# Athena Pipeline: Research and Developmnet
Generative AI content creation pipeline, images and video.

![ControlNet Apple Example](images/control_net_apple.jpg)

Current state: the backbone. We have web interface for python renderer to generate images and video. 
Having limited harware we utilize old robust Stable Diffusion 1.5 model for images and CogVideoX 2b for video.
This does not produce any interesting visuals but the goal is to learn the generative pipeline.


# Google Pipeline: Nano Banana and Veo3.1

## Workflow Overview
**Pipeline Strategy:** "Bake & Feed"
Stop treating Veo as a *compositor*. Use the Image Model to build the world (Bake), and the Video Model only to move it (Feed).

### Phase 1: Asset Generation (Nano Banana)
Generate the raw ingredients. Do not rely on one image per asset.

* **Character Kit:**
    * **Identity Truth:** Medium shot, 3/4 view (Face detail).
    * **Anatomy Truth:** Full body T-pose/A-pose (Proportions & Costume).
    * **Texture Truth:** Close-up action shot (Lighting & Material).
* **Location Kit:**
    * **Master Wide:** Establishing shot of the empty environment.
    * **Reverse Angle:** 180° view for dialogue coverage.
    * **Canny Layout:** Simple 3D blockout renders (edges) to force perspective if needed.

### Phase 2: The "Bake" (Pre-Composition)
**Goal:** Create the perfect "Shot 0" using Nano Banana's 14-image context window.

* **Action:** Feed your **Character Kit** + **Location Kit** + **Props** into Nano Banana.
* **Prompt:** Define the exact composition (e.g., "Character A standing left, holding prop, inside Location B").
* **Output:** A high-fidelity **Master Keyframe** where all subjects are already correctly placed and lit.

### Phase 3: Shot Simulation (Veo 3.1)
**Goal:** Animate the Master Keyframe. Use "Ingredients to Video" mode.

* **Input 1 (The Anchor):** Your **Master Keyframe** (from Phase 2).
    * *Role:* Sets the scene, lighting, and starting position.
* **Input 2 (The Passport):** Your **Character Identity Truth** (Face crop).
    * *Role:* Forces the model to maintain facial likeness during movement.
* **Prompt:** Describe the **motion only** (e.g., "Camera pushes in, character turns head"). Do not describe objects or colors; they are already in the image.

### Phase 4: Audio (Decoupled)
* **Voice:** Generate consistent dialogue in Vertex AI / ElevenLabs.
* **Performance:** Generate video in Veo with neutral or generic talking motion.
* **Sync:** Use an external Lip Sync model (e.g., SyncLabs) to merge the audio and video in post.


