import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const articles = defineCollection({
  loader: glob({ pattern: '**/index.mdx', base: './src/content/articles' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    author: z.string(),
    tags: z.array(z.string()),
    draft: z.boolean().default(false),
    cover: z.string().optional(),
    ogImage: z.string().optional(),
    tstorchBundles: z.array(z.string()).default([]),
  }),
});

export const collections = { articles };
