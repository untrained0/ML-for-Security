'use client';
import { useState, useEffect } from 'react';
import useStore from '@/store/useStore';

export default function TutorialModal() {
  const { showTutorial, setShowTutorial } = useStore();
  const [step, setStep] = useState(0);

  useEffect(() => {
    const hasSeenTutorial = localStorage.getItem('hasSeenTutorial');
    if (!hasSeenTutorial) {
      setShowTutorial(true);
    }
  }, [setShowTutorial]);

  const completeTutorial = () => {
    localStorage.setItem('hasSeenTutorial', 'true');
    setShowTutorial(false);
  };

  const steps = [
    {
      title: "Welcome to ML Security Viz",
      content: "This tool lets you visualize adversarial machine learning attacks in real-time. We'll specifically look at Poisoning Attacks against Support Vector Machines (SVMs)."
    },
    {
      title: "1. The Control Panel (Left)",
      content: "Here you can select your dataset (or upload a CSV), configure the SVM model hyperparameters, and set the poisoning attack parameters like the number of poison points and the gradient ascent step size."
    },
    {
      title: "2. The Canvas (Center)",
      content: "The main canvas plots your dataset, the SVM decision boundary, and the adversarial poison points (☠). It supports infinite pan and zoom."
    },
    {
      title: "3. The Math Inspector (Right)",
      content: "See the math come alive! This panel shows the exact equations being optimized. During an attack, it dynamically tracks the test metrics (F1, Precision, Recall) so you can measure the attack's effectiveness."
    },
    {
      title: "4. The Timeline (Bottom)",
      content: "The poisoning attack is solved iteratively via gradient ascent. The timeline plots the attacker's objective function and weight delta over each iteration. Use the Playback Bar above it to scrub through history!"
    }
  ];

  if (!showTutorial) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[var(--bg-secondary)] border border-[var(--border-default)] rounded-xl shadow-2xl w-[500px] max-w-[90vw] overflow-hidden flex flex-col animate-[fadeIn_0.3s_ease-out]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-[var(--border-subtle)] bg-[var(--bg-tertiary)] flex items-center justify-between">
          <h2 className="text-lg font-bold text-[var(--text-primary)] m-0">🎓 Interactive Tutorial</h2>
          <button onClick={completeTutorial} className="text-[var(--text-tertiary)] hover:text-[var(--text-primary)] transition-colors cursor-pointer text-xl bg-transparent border-none">×</button>
        </div>
        
        {/* Body */}
        <div className="p-6 flex flex-col gap-4 min-h-[160px]">
          <h3 className="text-base font-semibold text-[var(--accent-primary)]">{steps[step].title}</h3>
          <p className="text-sm text-[var(--text-secondary)] leading-relaxed">{steps[step].content}</p>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[var(--border-subtle)] bg-[var(--bg-primary)] flex items-center justify-between">
          <div className="flex gap-1.5">
            {steps.map((_, i) => (
              <div key={i} className={`w-2 h-2 rounded-full transition-colors ${i === step ? 'bg-[var(--accent-primary)]' : 'bg-[var(--border-default)]'}`} />
            ))}
          </div>
          <div className="flex gap-3">
            {step > 0 && (
              <button 
                onClick={() => setStep(s => s - 1)}
                className="px-4 py-1.5 rounded text-sm font-medium border border-[var(--border-default)] text-[var(--text-primary)] bg-transparent hover:bg-[var(--bg-tertiary)] cursor-pointer"
              >
                Back
              </button>
            )}
            {step < steps.length - 1 ? (
              <button 
                onClick={() => setStep(s => s + 1)}
                className="px-4 py-1.5 rounded text-sm font-medium border-none text-white bg-[var(--accent-primary)] hover:bg-[var(--accent-hover)] cursor-pointer"
              >
                Next
              </button>
            ) : (
              <button 
                onClick={completeTutorial}
                className="px-4 py-1.5 rounded text-sm font-medium border-none text-white bg-[#059669] hover:bg-[#047857] cursor-pointer"
              >
                Get Started!
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
