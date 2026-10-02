'use client';
import { useState } from 'react';
import useStore from '@/store/useStore';
import { findAlgorithm } from '@/engine/architectures';

export default function GuidedTour() {
  const { activeAlgorithm } = useStore();
  const [isOpen, setIsOpen] = useState(true);
  const [step, setStep] = useState(0);
  // Switching algorithms starts its tour at card 1 (state adjusted during render, not in an effect)
  const [shownFor, setShownFor] = useState(activeAlgorithm);
  if (shownFor !== activeAlgorithm) {
    setShownFor(activeAlgorithm);
    setStep(0);
  }

  const svmSteps = [
    {
      title: "Welcome to ML Security",
      content: "In this simulator, we explore how adversaries can trick machine learning models by subtly altering the training data. Here we are using a Support Vector Machine (SVM) to find the best decision boundary separating two classes."
    },
    {
      title: "The Attack (Biggio 2012)",
      content: "The attacker injects 'poison points' (red points) into the training set. The goal is to maximize the classification error on a clean validation set. They achieve this using a technique called Gradient Ascent."
    },
    {
      title: "Gradient Ascent",
      content: "Watch how the poison points move when you click 'Launch Attack'. The arrows on the canvas show the gradient direction pulling the poison points to maximize the model's error at each iteration."
    },
    {
      title: "The Impact",
      content: "Notice how the clean accuracy drops as the poisoned boundary shifts away from the clean boundary. The attacker successfully subverted the model without controlling the algorithm itself!"
    }
  ];

  const regressionSteps = [
    {
      title: "Regression Poisoning",
      content: "We are exploring how attackers can manipulate regression models (like predicting loan rates or drug dosages). We use models like Ridge or LASSO to fit a line through the data."
    },
    {
      title: "Bilevel Optimization (Jagielski 2018)",
      content: "The attacker injects poison points to maximize the Mean Squared Error (MSE) on validation data. This is a bilevel problem: the inner level trains the model, the outer level maximizes the error."
    },
    {
      title: "Implicit Differentiation",
      content: "To compute the gradient for the poison points, the attacker uses the KKT conditions and implicit differentiation on the closed-form regression solution!"
    },
    {
      title: "The Impact",
      content: "As you step through the attack timeline, observe how the regression line tilts significantly. The attacker has successfully corrupted the predictions for normal inputs."
    }
  ];

  // The tours describe the geometric workspace; a view module (or an unknown key) has none
  const alg = findAlgorithm(activeAlgorithm);
  if (!alg) return null;
  const steps = alg.tourSteps ?? (activeAlgorithm === 'biggio2012' ? svmSteps : regressionSteps);

  if (!isOpen) {
    return (
      <button 
        onClick={() => setIsOpen(true)}
        className="fixed bottom-16 right-6 w-12 h-12 bg-primary text-primary-foreground rounded-full shadow-glow flex items-center justify-center text-2xl hover:scale-105 transition-transform z-50 border border-primary/50 cursor-pointer"
        title="Open Guide"
      >
        💡
      </button>
    );
  }

  return (
    <div className="fixed bottom-16 right-6 w-[360px] glass-panel z-50 flex flex-col font-sans overflow-hidden animate-[fadeIn_0.3s_ease-out]">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle bg-secondary">
        <h3 className="font-bold text-foreground text-[15px]">{steps[step].title}</h3>
        <button onClick={() => setIsOpen(false)} className="text-muted-foreground/70 hover:text-foreground cursor-pointer text-xl leading-none transition-colors duration-150">&times;</button>
      </div>
      <div className="p-4 text-sm text-muted-foreground leading-relaxed min-h-[110px]">
        {steps[step].content}
      </div>
      <div className="px-4 py-3 bg-secondary/50 flex items-center justify-between border-t border-border-subtle">
        <button 
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          className="w-8 h-8 rounded-full flex items-center justify-center bg-secondary border border-border text-muted-foreground disabled:opacity-30 disabled:cursor-not-allowed hover:bg-muted hover:text-foreground cursor-pointer transition-colors duration-150"
        >
          &lt;
        </button>
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            {steps.map((_, i) => (
              <div key={i} className={`h-1.5 rounded-full transition-all duration-300 ${i === step ? 'w-4 bg-primary' : 'w-1.5 bg-muted'}`} />
            ))}
          </div>
          <span className="text-xs font-semibold text-primary bg-primary/10 px-2 py-0.5 rounded-full">
            {step + 1} / {steps.length}
          </span>
        </div>
        <button 
          onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
          disabled={step === steps.length - 1}
          className="w-8 h-8 rounded-full flex items-center justify-center bg-secondary border border-border text-muted-foreground disabled:opacity-30 disabled:cursor-not-allowed hover:bg-muted hover:text-foreground cursor-pointer transition-colors duration-150"
        >
          &gt;
        </button>
      </div>
    </div>
  );
}
