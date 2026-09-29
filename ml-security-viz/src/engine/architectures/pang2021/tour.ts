/**
 * Pang et al. 2021 — the guided tour (the 💡 panel): what the online attack does and how to read
 * the screen, in the order the user meets it.
 */

import type { TourStep } from '../registry';

export const pangTourSteps: TourStep[] = [
  {
    title: 'Poisoning a Model That Never Stops Learning',
    content: 'Pang et al. 2021 attack a model that trains online: a small neural network (or logistic regression) learns from real MNIST or CIFAR-10 images one batch at a time, with SGD momentum like the paper. The attacker sits upstream of the stream and may change every pixel of a batch by at most ε = 16/255, keeping the true labels.',
  },
  {
    title: 'Burn-in, Then the Accumulative Phase',
    content: '"Train Clean Model" runs the burn-in and gives θ₀. "Launch Attack" then perturbs the next T batches so that each update still looks like honest training while quietly moving the model to where one later batch will do a lot of damage (Algorithm 1, Eq. 9; λ weights that second goal). If test accuracy sags more than Δacc, the attacker stops early rather than be noticed.',
  },
  {
    title: 'The Trigger',
    content: 'The last frame feeds one poisoned trigger batch — a single ordinary update. The paper measures the single-step drop: test accuracy just before the trigger minus just after. The same kind of trigger fed to the honest model (same stream, no perturbations) is the baseline the accumulative phase has to beat.',
  },
  {
    title: 'Reading the Screen',
    content: 'The ☠ points are the batch fed at this frame, drawn on a 2-D projection of the images; click one to compare the clean image, the perturbed one and δ. The timeline tracks accuracy and the Eq. 7 alignment, and the 📖 explainer shows each equation with the numbers of the frame you are scrubbed to.',
  },
  {
    title: 'How Big Is the Effect?',
    content: 'Online, the damage is a few points per trigger step: at the defaults about 6 points on two-class CIFAR-10 against under 1 for the same trigger on the honest model, and about 4 on MNIST. The paper\'s online Table 1 reports 3–11 points; its drops to 10–30% come from the federated setting, where the attacker submits gradients instead of images.',
  },
];
