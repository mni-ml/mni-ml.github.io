import { defineCollection, z } from 'astro:content';
import { notionLoader } from './lib/notion/loader';

const schema = z.object({
  title: z.string(),
  description: z.string(),
  pubDate: z.coerce.date(),
  updatedDate: z.coerce.date().optional(),
  author: z.string().default('mni-ml'),
  tags: z.array(z.string()).default([]),
  draft: z.boolean().default(false),
  cover: z.string().optional(),
  ogImage: z.string().optional(),
  order: z.number().default(0),
  tstorchBundles: z.array(z.string()).default([]),
});

const notionArticles = defineCollection({
  loader: notionLoader(),
  schema,
});

export const collections = { notionArticles };
