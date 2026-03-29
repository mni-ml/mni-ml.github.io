import { useState, useEffect, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback: ReactNode;
}

export default function WebGPUDetect({ children, fallback }: Props) {
  const [supported, setSupported] = useState<boolean | null>(null);

  useEffect(() => {
    setSupported('gpu' in navigator);
  }, []);

  if (supported === null) {
    return <div className="webgpu-loading">Checking WebGPU support...</div>;
  }

  return <>{supported ? children : fallback}</>;
}
