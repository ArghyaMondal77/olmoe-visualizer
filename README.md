\# OLMoE-1B-7B Architecture Visualizer 🧠✨



An interactive, cinematic WebGL and DOM-based visualizer for the \*\*OLMoE-1B-7B Mixture-of-Experts (MoE)\*\* neural network. 



This project goes beyond high-level flowcharts. It uses \*\*real tensor telemetry\*\* extracted directly from the PyTorch model to animate the exact mathematical operations—from Tokenization and Rotary Positional Embeddings (RoPE) to Causal Attention Masking and Top-8 SwiGLU Expert Routing.



🔗 \*\*\[Live Demo on Vercel]\*\* \*(https://olmoe-visualizer.vercel.app)\*



\---



\## 🚀 The Engineering Approach



Running a 7-Billion parameter LLM backend just to visualize its architecture is expensive and introduces high latency. To make this visualizer fast, free to host, and mathematically authentic, we used a \*\*Static Telemetry Extraction\*\* approach:



1\. \*\*Colab Intercept:\*\* We ran a custom PyTorch script on a T4 GPU using `allenai/OLMoE-1B-7B-0924-Instruct`. We fed it a 2-pass autoregressive prompt (`"The capital of France is" ➔ " Paris" ➔ "."`).

2\. \*\*Tensor Extraction:\*\* Using PyTorch hooks, we intercepted the 2048D hidden states, the $16 \\times 16$ Attention Heatmaps, the MoE Router logits, and the LM Head probabilities.

3\. \*\*JSON Serialization:\*\* We converted these BFloat16 tensors into standard floats and packaged them into a lightweight, static `telemetry.json` file.

4\. \*\*Cinematic Frontend:\*\* The vanilla JavaScript frontend fetches this JSON on boot. The UI dynamically routes the real mathematical data into the CSS/Three.js animations, proving a 1:1 representation of the model's inner workings with \*\*zero backend latency\*\*.



\---



\## 🤖 About the Model: OLMoE-1B-7B



\*\*OLMoE\*\* (Open Language Model - Mixture of Experts) is a state-of-the-art sparse model developed by AllenAI. 



\* \*\*The 1B-7B Naming:\*\* The model contains \*\*7 Billion total parameters\*\*, giving it a massive knowledge capacity. However, because of its sparse routing, it only uses \*\*1 Billion active parameters\*\* per token. 

\* \*\*Efficiency:\*\* It performs on par with dense 7B models (like LLaMA-2 7B) but runs nearly as fast as a tiny 1B model, making it incredibly efficient for edge devices and fast inference.



\### Core Architectural Features Visualized in this Project:

\* \*\*RMSNorm:\*\* Pre-computation and post-computation variance stabilization.

\* \*\*RoPE (Rotary Position Embeddings):\*\* 2D complex vector rotation to inject absolute sequence position without learning positional weights.

\* \*\*Causal Masked Multi-Head Attention:\*\* 16 independent 128D heads calculating $Softmax(Q \\cdot K^T / \\sqrt{d}) \\cdot V$ with a strict upper-triangle $-\\infty$ mask.

\* \*\*Top-8 Sparse Routing:\*\* A Gating Network that evaluates 64 experts but mathematically forces 56 of them to zero, routing the token only to the Top 8.

\* \*\*SwiGLU Experts:\*\* Advanced Feed-Forward Networks utilizing dual expansion matrices ($W\_1$, $W\_3$) and a non-linear SiLU (Swish) activation curve.



\---



\## 📂 Repository Structure



The project is split into a \*\*Macro Topology\*\* pipeline and 6 \*\*Micro-Math Drill-Downs\*\*.



\* `index.html` - The Master Pipeline. A horizontal/vertical responsive circuit board tracking the token's journey through the Tokenizer, Embedding, 16-Layer Rack, and LM Head.

\* `telemetry.json` - The extracted PyTorch tensor data that drives the visualizations.

\* `tokenizer-zoom.html` - \*(Pending/Optional Micro-Math)\*

\* `embedding-zoom.html` - \*(Pending/Optional Micro-Math)\*

\* `rmsnorm-zoom.html` - Micro-math for Layer Normalization.

\* `qkv-zoom.html` - Micro-math for $W\_Q, W\_K, W\_V$ projection and multi-head slicing.

\* `rope-zoom.html` - Micro-math for 2D Rotary Embeddings.

\* `attention-zoom.html` - Explorable Softmax heatmap matrix and Causal Look-Ahead Masking.

\* `moe-router-zoom.html` - The switchboard logic masking 56 experts and applying Softmax to the Top-8.

\* `swiglu-expert-zoom.html` - The isolated math inside a single expert (SiLU activation and Hadamard product).

\* `lm-head-zoom.html` - The final $W\_{unembed}$ projection across the 50,304 vocabulary space and Argmax selection.



\---








