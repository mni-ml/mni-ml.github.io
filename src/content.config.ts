import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';
import { notionLoader } from './lib/notion/loader';

const railSectionSchema = z.object({
  label: z.string(),
  slug: z.string().optional(),
  soon: z.boolean().default(false),
});

const posterSchema = z.object({
  image: z.string().optional(),
  eyebrow: z.string().optional(),
  title: z.string().optional(),
  summary: z.string().optional(),
  chips: z.array(z.string()).default([]),
});

const presentationSchema = z.object({
  accent: z.string().optional(),
  railTitle: z.string().optional(),
  railSections: z.array(railSectionSchema).default([]),
  poster: posterSchema.optional(),
});

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
  order: z.number().optional(),
  tstorchBundles: z.array(z.string()).default([]),
  presentation: presentationSchema.optional(),
});

const localArticles = defineCollection({
  loader: glob({ pattern: '**/index.mdx', base: './src/content/articles' }),
  schema,
});

const notionArticles = defineCollection({
  loader: notionLoader(),
  schema,
});

export const collections = { localArticles, notionArticles };
