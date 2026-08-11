import "katex/dist/katex.min.css";
import "../globals.css";
import { russianMetadata } from "../metadata";

export const metadata = russianMetadata;

export default function RussianRootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
