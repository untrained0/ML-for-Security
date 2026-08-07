import { Inter, Inter_Tight, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const interTight = Inter_Tight({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["600", "700"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

export const metadata = {
  title: "ML Security Attack Visualizer — Poisoning Attacks on SVMs",
  description:
    "Interactive visualization of adversarial poisoning attacks on Support Vector Machines. Watch how gradient-ascent attacks degrade model decision boundaries in real time.",
  keywords:
    "machine learning, security, adversarial attacks, SVM, poisoning, visualization",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${inter.variable} ${interTight.variable} ${jetbrainsMono.variable} dark`}>
      <body className="font-sans bg-background text-foreground antialiased selection:bg-primary selection:text-white">
        {children}
      </body>
    </html>
  );
}
