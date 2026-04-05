import rss from '@astrojs/rss';
import { getAllArticles } from '../lib/articles';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const articles = await getAllArticles();

  return rss({
    title: 'TSTorch Notebook',
    description:
      'Research notebook for long-form ML and WebGPU articles',
    site: context.site!,
    items: articles.map((article) => ({
      title: article.data.title,
      pubDate: article.data.pubDate,
      description: article.data.description,
      link: `/articles/${article.id}/`,
    })),
  });
}
