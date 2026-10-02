import type { Metadata } from "next";
import "./globals.css";
import "./learn.css";
import { LanguageProvider } from "@/lib/i18n";

export const metadata: Metadata = {
  title: "Rattil · Learn to recite the Quran",
  description: "Learn to read and recite the Quran from the first letter, with a speech model that listens.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body suppressHydrationWarning><LanguageProvider>{children}</LanguageProvider></body></html>;
}
