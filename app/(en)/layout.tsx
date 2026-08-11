import "katex/dist/katex.min.css";
import "../globals.css";
import { englishMetadata } from "../metadata";

export const metadata = englishMetadata;

export default function EnglishRootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
