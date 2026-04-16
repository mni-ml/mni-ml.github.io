import rss from '@astrojs/rss';
import { getAllArticles } from '../lib/articles';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const articles = await getAllArticles();
  const base = import.meta.env.BASE_URL;

  return rss({
    title: 'mni-ml',
    description:
      'A curriculum for understanding machine learning from scratch',
    site: context.site!,
    items: articles.map((article) => ({
      title: article.data.title,
      pubDate: article.data.pubDate,
      description: article.data.description,
      link: `${base}articles/${article.id}/`,
    })),
  });
}
