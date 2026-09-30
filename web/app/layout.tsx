import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tarteel · A companion for your recitation",
  description: "A quiet space to practice Quran recitation with local Qaloon speech recognition.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
