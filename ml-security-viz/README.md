# ML Security Visualizer: SVM Poisoning

An interactive, browser-based visualization of gradient-based poisoning attacks against Support Vector Machines (SVMs).

Based on the landmark paper:
**"Poisoning Attacks against Support Vector Machines"** by Battista Biggio, Blaine Nelson, and Pavel Laskov (ICML 2012).
[Read the paper on arXiv](https://arxiv.org/abs/1206.6389).

## Features

- 🧠 **Real-time SVM Training**: Trains Linear and RBF kernel SVMs entirely in the browser using Web Workers.
- ☠️ **Interactive Poisoning**: Configure the attacker's step size ($\eta$), backtracking factor ($\beta$), and initialization strategy to launch adversarial poisoning attacks.
- 🗺️ **Attacker Objective Heatmaps**: Visualize the validation hinge loss objective landscape and gradient vector field (quiver plots) directly on the canvas.
- 🔍 **Mathematical Transparency**: Integrated KaTeX equation panel that dynamically updates with live weight norms and metrics. Tracks Support Vector dynamics (gained/lost SVs).
- 📊 **Performance Comparison**: Live table tracking Accuracy, Precision, Recall, and F1 Score for both clean and poisoned models.
- 💾 **Data Export**: Export the raw attack trajectory to JSON for reproducible research, or download high-resolution SVG snapshots of the decision boundary.
- 📂 **Custom Datasets**: Upload your own 2D datasets via CSV.

## Getting Started

First, install the dependencies:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Usage Guide

1. **Setup**: Use the left Control Panel to select a dataset (or upload your own CSV) and configure the SVM model (Linear or RBF).
2. **Train Clean Model**: Click "Train Clean Model" to establish the baseline decision boundary.
3. **Configure Attack**: Select the number of poison points and gradient descent hyperparameters.
4. **Launch Attack**: Click "Launch Attack". The Web Worker will iteratively maximize the attacker's objective function.
5. **Analyze**: Use the Timeline and Playback Bar to step through the attack iteration by iteration. Toggle the Math Panel on the right to see the changing support vectors and mathematical impact.

## Architecture

- **Framework**: Next.js 16 (App Router), React 19, TypeScript
- **Styling**: Tailwind CSS v4
- **Visualization**: D3.js (Canvas math & scaling), KaTeX (Math rendering)
- **ML Engine**: Custom SVM and gradient ascent implementation written in raw TypeScript, executing off-main-thread via Web Workers.
- **State Management**: Zustand

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for new attack implementations or performance improvements.
