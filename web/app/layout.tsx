import type { Metadata } from "next";
import "./globals.css";
import "./learn.css";
import "./challenges.css";
import "./reader.css";
import "./tajweed.css";
import "./search.css";
import "./mobile.css";
import { LanguageProvider } from "@/lib/i18n";
import { MobileNavigation } from "@/components/mobile-navigation";

export const metadata: Metadata = {
  title: "Rattil · Learn to recite the Quran",
  description: "Learn to read and recite the Quran from the first letter, with a speech model that listens.",
};

export const viewport = { width: "device-width", initialScale: 1, viewportFit: "cover" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body suppressHydrationWarning><LanguageProvider>{children}<MobileNavigation /></LanguageProvider></body></html>;
}
