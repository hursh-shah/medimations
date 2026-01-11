import "./globals.css";

export const metadata = {
  title: "Medical Diffusion",
  description: "Generate medically grounded animations with Veo + BiomedCLIP verification"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
