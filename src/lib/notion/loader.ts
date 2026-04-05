import type { Loader, LoaderContext } from "astro/loaders";
import { hasNotionEnv, getNotionEnv } from "./env";
import { notionFetchJson } from "./http";
import {
  normalizeNotionRow,
  blocksToMarkdown,
  type NotionPage,
  type NotionBlock,
} from "./normalize";

// ---------------------------------------------------------------------------
// Notion API types
// ---------------------------------------------------------------------------

type NotionPaginatedResponse = {
  results: unknown[];
  next_cursor: string | null;
  has_more: boolean;
};

// ---------------------------------------------------------------------------
// Notion API helpers (ported from portfolio/lib/notion/fetchPosts.ts)
// ---------------------------------------------------------------------------

async function queryBlogDatabasePage(
  dbId: string,
  body: unknown,
): Promise<NotionPaginatedResponse> {
  return await notionFetchJson<NotionPaginatedResponse>(
    `/databases/${dbId}/query`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

async function queryAllBlogDatabase(
  dbId: string,
  baseBody: Record<string, unknown>,
): Promise<unknown[]> {
  const all: unknown[] = [];
  let cursor: string | undefined;
  for (let guard = 0; guard < 50; guard++) {
    const body = { ...baseBody, ...(cursor ? { start_cursor: cursor } : {}) };
    const resp = await queryBlogDatabasePage(dbId, body);
    all.push(...(resp.results ?? []));
    if (!resp.has_more) break;
    cursor = resp.next_cursor ?? undefined;
    if (!cursor) break;
  }
  return all;
}

async function listAllBlockChildren(blockId: string): Promise<unknown[]> {
  const all: unknown[] = [];
  let cursor: string | undefined;
  for (let guard = 0; guard < 50; guard++) {
    const qp = new URLSearchParams({ page_size: "100" });
    if (cursor) qp.set("start_cursor", cursor);
    const resp = await notionFetchJson<NotionPaginatedResponse>(
      `/blocks/${blockId}/children?${qp.toString()}`,
      { method: "GET" },
    );
    all.push(...(resp.results ?? []));
    if (!resp.has_more) break;
    cursor = resp.next_cursor ?? undefined;
    if (!cursor) break;
  }
  return all;
}

// ---------------------------------------------------------------------------
// Type guards
// ---------------------------------------------------------------------------

function isNotionPage(value: unknown): value is NotionPage {
  return (
    !!value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as any).id === "string" &&
    typeof (value as any).properties === "object" &&
    (value as any).properties != null
  );
}

function isNotionBlock(value: unknown): value is NotionBlock {
  return (
    !!value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as any).id === "string" &&
    typeof (value as any).type === "string"
  );
}

// ---------------------------------------------------------------------------
// Astro content loader
// ---------------------------------------------------------------------------

export function notionLoader(): Loader {
  return {
    name: "notion-loader",

    async load(context: LoaderContext) {
      if (!hasNotionEnv()) {
        context.logger.warn(
          "Notion env vars not set (NOTION_TOKEN, NOTION_BLOG_DB_ID). Skipping Notion articles.",
        );
        return;
      }

      const { blogDbId } = getNotionEnv();
      const includeDrafts = process.env.NOTION_INCLUDE_DRAFTS === "true";

      context.logger.info("Fetching articles from Notion...");

      // Query all published pages
      const body: Record<string, unknown> = { page_size: 100 };
      if (!includeDrafts) {
        body.filter = {
          property: "status",
          status: { equals: "Published" },
        };
      }

      const results = await queryAllBlogDatabase(blogDbId, body);
      const pages = results.filter(isNotionPage);

      context.logger.info(`Found ${pages.length} Notion article(s).`);

      // Check for duplicate slugs
      const slugCounts = new Map<string, number>();
      const metas = pages.map((p) => normalizeNotionRow(p));
      for (const m of metas) {
        if (m.slug) slugCounts.set(m.slug, (slugCounts.get(m.slug) ?? 0) + 1);
      }
      const dups = Array.from(slugCounts.entries()).filter(([, c]) => c > 1);
      if (dups.length > 0) {
        const msg = `Duplicate blog slugs in Notion: ${dups.map(([s]) => s).join(", ")}`;
        context.logger.error(msg);
        throw new Error(msg);
      }

      // Clear previous entries so deleted Notion pages are removed
      context.store.clear();

      for (const page of pages) {
        const meta = normalizeNotionRow(page);
        if (!meta.slug) {
          context.logger.warn(`Skipping Notion page ${page.id}: no slug`);
          continue;
        }

        // Fetch block content
        const blocksRaw = await listAllBlockChildren(page.id);
        const blocks = blocksRaw.filter(isNotionBlock);

        const unsupported: string[] = [];
        const markdown = blocksToMarkdown(blocks, unsupported).trim();

        if (unsupported.length > 0) {
          const uniq = Array.from(new Set(unsupported)).sort();
          context.logger.warn(
            `Unsupported blocks in "${meta.slug}": ${uniq.join(", ")}`,
          );
        }

        // Validate against schema
        const data = await context.parseData({
          id: meta.slug,
          data: {
            title: meta.title,
            description: meta.description,
            pubDate: meta.pubDate || new Date().toISOString().split("T")[0],
            author: meta.author,
            tags: meta.tags,
            draft: meta.draft,
            cover: meta.cover,
            ogImage: meta.cover,
          },
        });

        // Render markdown through Astro's pipeline (KaTeX + Shiki)
        const rendered = await context.renderMarkdown(markdown);

        context.store.set({
          id: meta.slug,
          data,
          body: markdown,
          rendered,
          digest: context.generateDigest(markdown),
        });

        context.logger.info(`  ✓ ${meta.slug}`);
      }

      context.logger.info("Notion sync complete.");
    },
  };
}
