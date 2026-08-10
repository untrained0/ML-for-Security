import { Inter, Inter_Tight, JetBrains_Mono } from "next/font/google";
import "./globals.css";

/* Variable fonts — one file per family, every weight available.
   The `variable` names below are what globals.css `@theme` reads. */
const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const interTight = Inter_Tight({
  variable: "--font-inter-tight",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata = {
  title: "ML Security Lab — Interactive Poisoning Attack Visualizer",
  description:
    "Step through adversarial poisoning attacks on SVMs, regression models, and online learners. Watch decision boundaries bend, inspect the gradients driving each iteration, and compare clean against poisoned models in real time.",
  keywords:
    "machine learning, security, adversarial attacks, data poisoning, SVM, regression, online learning, visualization",
};

/* Resolves the theme before first paint so the page never flashes the
   wrong background. Static string — no user input reaches this script.
   Order: saved choice → system preference → dark. */
const themeInit = `(function(){try{var s=localStorage.getItem('ml-sec-theme');var t=(s==='light'||s==='dark')?s:(window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark');document.documentElement.classList.toggle('dark',t==='dark');document.documentElement.style.colorScheme=t;}catch(e){document.documentElement.classList.add('dark');}})();`;

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${interTight.variable} ${jetbrainsMono.variable}`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body className="font-sans bg-background text-foreground antialiased">
        {children}
      </body>
    </html>
  );
}
