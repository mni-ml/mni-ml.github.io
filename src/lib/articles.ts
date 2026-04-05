import { getCollection } from 'astro:content';

export async function getAllArticles() {
  const [local, notion] = await Promise.all([
    getCollection('localArticles'),
    getCollection('notionArticles'),
  ]);
  return [...local, ...notion]
    .filter((a) => !a.data.draft)
    .sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}
