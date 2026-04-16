import { getCollection } from 'astro:content';

export async function getAllArticles() {
  const articles = await getCollection('notionArticles');
  return articles
    .filter((a) => !a.data.draft)
    .sort((a, b) => a.data.order - b.data.order);
}

export async function getPrevNextArticle(slug: string) {
  const articles = await getAllArticles();
  const idx = articles.findIndex((a) => a.id === slug);
  return {
    prev: idx > 0 ? articles[idx - 1] : undefined,
    next: idx < articles.length - 1 ? articles[idx + 1] : undefined,
  };
}
