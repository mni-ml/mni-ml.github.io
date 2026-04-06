// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import remarkSmartypants from 'remark-smartypants';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://mni-ml.pages.dev',
  output: 'static',
  integrations: [
    mdx(),
    react({ include: ['**/tstorch/**'] }),
    sitemap(),
  ],
  markdown: {
    remarkPlugins: [remarkMath, remarkSmartypants],
    rehypePlugins: [[rehypeKatex, { output: 'html' }]],
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
