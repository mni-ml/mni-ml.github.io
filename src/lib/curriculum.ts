export type SubSection = {
  title: string;
  slug?: string; // anchor within the page (h2 id)
  soon?: boolean;
};

export type Module = {
  order: number;
  title: string;
  slug: string; // route slug
  subSections?: SubSection[];
};

export const curriculum: Module[] = [
  { order: 1, title: 'Intro', slug: 'introduction' },
  { order: 2, title: 'Gradient Descent', slug: 'gradient-descent' },
  { order: 3, title: 'Optimizations', slug: 'optimizations' },
  {
    order: 4,
    title: 'Neural Networks',
    slug: 'neural-networks',
    subSections: [
      { title: 'torch.nn.Module' },
      { title: 'Linear Neural Network' },
      { title: 'CNN' },
      { title: 'Transformer' },
      { title: 'FNN', soon: true },
      { title: 'RNN', soon: true },
      { title: 'LSTM', soon: true },
      { title: 'GAN', soon: true },
      { title: 'Autoencoder', soon: true },
    ],
  },
];

export function getModuleBySlug(slug: string): Module | undefined {
  return curriculum.find((m) => m.slug === slug);
}

export function getPrevNext(slug: string): { prev?: Module; next?: Module } {
  const idx = curriculum.findIndex((m) => m.slug === slug);
  return {
    prev: idx > 0 ? curriculum[idx - 1] : undefined,
    next: idx < curriculum.length - 1 ? curriculum[idx + 1] : undefined,
  };
}
