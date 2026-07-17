import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Local Transcriber",
  description: "GPU-accelerated local audio transcription with faster-whisper",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-background antialiased">{children}</body>
    </html>
  );
}
