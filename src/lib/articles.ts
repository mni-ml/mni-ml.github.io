import { getCollection, type CollectionEntry } from 'astro:content';

type LocalArticleEntry = CollectionEntry<'localArticles'>;
type NotionArticleEntry = CollectionEntry<'notionArticles'>;

export type ArticleEntry = LocalArticleEntry | NotionArticleEntry;
export type ArticlePresentation = NonNullable<ArticleEntry['data']['presentation']>;
export type ArticleHeading = {
  depth: number;
  slug: string;
  text: string;
};

export type ArticleNavItem = {
  slug: string;
  title: string;
  description: string;
  pubDate: Date;
  tags: string[];
  sequence: number;
  presentation?: ArticlePresentation;
};

export type ArticleRailSection = {
  label: string;
  slug?: string;
  depth: number;
  soon?: boolean;
};

function compareArticles(a: ArticleEntry, b: ArticleEntry) {
  const aOrder = a.data.order;
  const bOrder = b.data.order;

  if (typeof aOrder === 'number' && typeof bOrder === 'number' && aOrder !== bOrder) {
    return aOrder - bOrder;
  }

  if (typeof aOrder === 'number') return -1;
  if (typeof bOrder === 'number') return 1;

  return b.data.pubDate.getTime() - a.data.pubDate.getTime() || a.data.title.localeCompare(b.data.title);
}

export async function getAllArticles(): Promise<ArticleEntry[]> {
  const [local, notion] = await Promise.all([
    getCollection('localArticles'),
    getCollection('notionArticles'),
  ]);

  return [...local, ...notion]
    .filter((a) => !a.data.draft)
    .sort(compareArticles);
}

export function getArticleNav(articles: ArticleEntry[]): ArticleNavItem[] {
  return articles.map((article, index) => ({
    slug: article.id,
    title: article.data.title,
    description: article.data.description,
    pubDate: article.data.pubDate,
    tags: article.data.tags,
    sequence: index + 1,
    presentation: article.data.presentation,
  }));
}

export function getAdjacentArticles(articleNav: ArticleNavItem[], slug: string) {
  const idx = articleNav.findIndex((article) => article.slug === slug);

  return {
    prev: idx > 0 ? articleNav[idx - 1] : undefined,
    next: idx >= 0 && idx < articleNav.length - 1 ? articleNav[idx + 1] : undefined,
    current: idx >= 0 ? articleNav[idx] : undefined,
  };
}

export function getArticleRailSections(
  headings: ArticleHeading[],
  presentation?: ArticlePresentation,
): ArticleRailSection[] {
  if (presentation?.railSections?.length) {
    return presentation.railSections.map((section, index) => ({
      label: section.label,
      slug: section.slug,
      depth: index === 0 ? 2 : 3,
      soon: section.soon,
    }));
  }

  return headings
    .filter((heading) => heading.depth >= 2 && heading.depth <= 3)
    .map((heading) => ({
      label: heading.text,
      slug: heading.slug,
      depth: heading.depth,
    }));
}
