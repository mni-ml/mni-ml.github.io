export type NotionPage = {
  id: string;
  properties: Record<string, unknown>;
};

export type NotionBlock = {
  id: string;
  type: string;
  [key: string]: unknown;
};

export type ArticleMeta = {
  title: string;
  slug: string;
  description: string;
  pubDate: string;
  author: string;
  tags: string[];
  draft: boolean;
  cover?: string;
};

export type ArticleFull = ArticleMeta & {
  content: string;
};

type NormalizeOptions = {
  strict?: boolean;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function pickProperty(
  properties: Record<string, unknown>,
  names: string[],
): unknown {
  for (const name of names) {
    const v = properties[name];
    if (v != null) return v;
  }
  return undefined;
}

function findPropertyRecord(
  properties: Record<string, unknown>,
  names: string[],
  fallbackTypes: string[] = [],
): Record<string, unknown> | undefined {
  const byName = pickProperty(properties, names);
  if (isRecord(byName)) return byName;

  if (fallbackTypes.length === 0) return undefined;

  const matches: Record<string, unknown>[] = [];
  for (const v of Object.values(properties)) {
    if (!isRecord(v)) continue;
    const t = typeof v.type === "string" ? v.type : "";
    if (fallbackTypes.includes(t)) matches.push(v);
  }

  if (matches.length === 1) return matches[0];
  return undefined;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

// ---------------------------------------------------------------------------
// Rich text → Markdown (enhanced from portfolio's plain-text-only version)
// ---------------------------------------------------------------------------

type RichTextItem = {
  type: string;
  plain_text?: string;
  href?: string | null;
  annotations?: {
    bold?: boolean;
    italic?: boolean;
    strikethrough?: boolean;
    code?: boolean;
    underline?: boolean;
    color?: string;
  };
  equation?: {
    expression: string;
  };
};

function richTextItemToMarkdown(item: RichTextItem): string {
  // Inline equation (Notion equation mention)
  if (item.type === "equation" && item.equation?.expression) {
    return `$${item.equation.expression}$`;
  }

  let text = item.plain_text ?? "";
  if (!text) return "";

  const a = item.annotations;
  if (a) {
    if (a.code) text = `\`${text}\``;
    if (a.bold) text = `**${text}**`;
    if (a.italic) text = `*${text}*`;
    if (a.strikethrough) text = `~~${text}~~`;
  }

  if (item.href) {
    text = `[${text}](${item.href})`;
  }

  return text;
}

function richTextToMarkdown(richText: unknown): string {
  const items = asArray(richText);
  return items
    .map((it) => (isRecord(it) ? richTextItemToMarkdown(it as unknown as RichTextItem) : ""))
    .join("");
}

/** Plain text only — for property extraction where markdown isn't wanted. */
function getPlainText(richText: unknown): string {
  const items = asArray(richText);
  return items
    .map((it) => (isRecord(it) ? (it.plain_text as string | undefined) : undefined))
    .filter((s): s is string => typeof s === "string" && s.length > 0)
    .join("");
}

// ---------------------------------------------------------------------------
// Property extraction
// ---------------------------------------------------------------------------

function getTitle(properties: Record<string, unknown>): string {
  const matches: Record<string, unknown>[] = [];
  for (const v of Object.values(properties)) {
    if (!isRecord(v)) continue;
    const t = typeof v.type === "string" ? v.type : "";
    if (t === "title") matches.push(v);
  }

  if (matches.length !== 1) return "";
  return getPlainText((matches[0] as any).title);
}

function getRichTextProperty(
  properties: Record<string, unknown>,
  names: string[],
): string {
  const prop = findPropertyRecord(properties, names, ["rich_text"]);
  if (!prop) return "";
  return getPlainText((prop as any).rich_text);
}

function getSelectName(
  properties: Record<string, unknown>,
  names: string[],
): string {
  const prop = findPropertyRecord(properties, names, ["status", "select"]);
  if (!prop) return "";

  const status = (prop as any).status;
  if (isRecord(status)) return typeof status.name === "string" ? status.name : "";

  const select = (prop as any).select;
  if (isRecord(select)) return typeof select.name === "string" ? select.name : "";

  return "";
}

function getMultiSelectNames(
  properties: Record<string, unknown>,
  names: string[],
): string[] {
  const prop = findPropertyRecord(properties, names, ["multi_select"]);
  if (!prop) return [];
  const ms = asArray((prop as any).multi_select);
  return ms
    .map((t) => (isRecord(t) ? (t.name as string | undefined) : undefined))
    .filter((s): s is string => typeof s === "string" && s.length > 0);
}

function getDateStart(
  properties: Record<string, unknown>,
  names: string[],
): string {
  const prop = findPropertyRecord(properties, names, ["date"]);
  if (!prop) return "";
  const date = (prop as any).date;
  if (!isRecord(date)) return "";
  return typeof date.start === "string" ? date.start : "";
}

function getUrlOrRichText(
  properties: Record<string, unknown>,
  names: string[],
): string | undefined {
  const prop = findPropertyRecord(properties, names, ["url", "rich_text"]);
  if (!prop) return undefined;
  const url = (prop as any).url;
  if (typeof url === "string" && url.length > 0) return url;
  const rt = getPlainText((prop as any).rich_text);
  return rt.length > 0 ? rt : undefined;
}

function requireField(value: string, name: string): string {
  if (!value) throw new Error(`Missing required Notion property: ${name}`);
  return value;
}

// ---------------------------------------------------------------------------
// Row normalization (metadata only)
// ---------------------------------------------------------------------------

export function normalizeNotionRow(page: NotionPage): ArticleMeta {
  const props = page.properties ?? {};
  if (!isRecord(props)) throw new Error("Notion page properties missing");

  const slug = getTitle(props);
  const title = getRichTextProperty(props, ["title", "Title"]) || slug;
  const description = getRichTextProperty(props, ["summary", "Summary", "description", "Description"]);
  const pubDate = getDateStart(props, ["publishedAt", "PublishedAt", "published date", "pubDate"]);
  const author = getRichTextProperty(props, ["author", "Author"]) || "TSTorch Team";
  const tags = getMultiSelectNames(props, ["tags", "Tags"]);
  const cover = getUrlOrRichText(props, ["imageUrl", "ImageUrl", "image", "cover"]);
  const status = getSelectName(props, ["status", "Status"]);

  return {
    title,
    slug,
    description,
    pubDate,
    author,
    tags,
    draft: status !== "Published",
    cover,
  };
}

// ---------------------------------------------------------------------------
// Block → Markdown
// ---------------------------------------------------------------------------

function blockRichText(block: NotionBlock, key: string): string {
  const typed = block[block.type];
  if (!isRecord(typed)) return "";
  return richTextToMarkdown(typed[key]);
}

function blockCode(block: NotionBlock): { language: string; code: string } {
  const typed = block[block.type];
  if (!isRecord(typed)) return { language: "", code: "" };
  const language = typeof typed.language === "string" ? typed.language : "";
  const code = getPlainText(typed.rich_text);
  return { language, code };
}

function blockImage(block: NotionBlock): { url: string; alt: string } | null {
  const typed = block[block.type];
  if (!isRecord(typed)) return null;
  const caption = getPlainText(typed.caption);

  const file = typed.file;
  if (isRecord(file) && typeof file.url === "string") {
    return { url: file.url, alt: caption };
  }

  const external = typed.external;
  if (isRecord(external) && typeof external.url === "string") {
    return { url: external.url, alt: caption };
  }

  return null;
}

function blockEquation(block: NotionBlock): string {
  const typed = block[block.type];
  if (!isRecord(typed)) return "";
  const expression = typeof typed.expression === "string" ? typed.expression : "";
  if (!expression) return "";
  return `$$\n${expression}\n$$`;
}

type SupportedBlockType =
  | "heading_1"
  | "heading_2"
  | "heading_3"
  | "paragraph"
  | "bulleted_list_item"
  | "numbered_list_item"
  | "quote"
  | "code"
  | "divider"
  | "image"
  | "equation";

function isSupportedBlockType(type: string): type is SupportedBlockType {
  return [
    "heading_1", "heading_2", "heading_3",
    "paragraph", "bulleted_list_item", "numbered_list_item",
    "quote", "code", "divider", "image", "equation",
  ].includes(type);
}

function blockToMarkdownChunk(block: NotionBlock, unsupported: string[]): string {
  const type = block.type;
  if (!isSupportedBlockType(type)) {
    unsupported.push(type);
    return "";
  }

  switch (type) {
    case "heading_1":
      return `# ${blockRichText(block, "rich_text")}`.trimEnd();
    case "heading_2":
      return `## ${blockRichText(block, "rich_text")}`.trimEnd();
    case "heading_3":
      return `### ${blockRichText(block, "rich_text")}`.trimEnd();
    case "paragraph": {
      const t = blockRichText(block, "rich_text");
      return t.trimEnd();
    }
    case "bulleted_list_item":
      return `- ${blockRichText(block, "rich_text")}`.trimEnd();
    case "numbered_list_item":
      return `1. ${blockRichText(block, "rich_text")}`.trimEnd();
    case "quote":
      return `> ${blockRichText(block, "rich_text")}`.trimEnd();
    case "code": {
      const { language, code } = blockCode(block);
      // TSTorch demo marker convention
      if (language.startsWith("tstorch:")) {
        const component = language.slice("tstorch:".length).trim();
        return `<div data-tstorch="${component}">${code}</div>`;
      }
      const fence = `\`\`\`${language}`.trimEnd();
      return [fence, code, "```"].join("\n");
    }
    case "divider":
      return "---";
    case "image": {
      const img = blockImage(block);
      if (!img) return "";
      const safeAlt = img.alt ?? "";
      return `![${safeAlt}](${img.url})`;
    }
    case "equation":
      return blockEquation(block);
  }
}

function isListType(type: string): boolean {
  return type === "bulleted_list_item" || type === "numbered_list_item";
}

export function blocksToMarkdown(blocks: NotionBlock[], unsupported: string[]): string {
  const out: string[] = [];
  let prevType: string | null = null;

  for (const b of blocks) {
    if (!b || typeof b.type !== "string") continue;
    const chunk = blockToMarkdownChunk(b, unsupported);
    if (!chunk) {
      prevType = b.type;
      continue;
    }

    const needsBlankLine =
      out.length > 0 &&
      !isListType(b.type) &&
      !isListType(prevType ?? "") &&
      out[out.length - 1] !== "";

    if (needsBlankLine) out.push("");
    out.push(...chunk.split("\n"));
    prevType = b.type;
  }

  while (out.length > 0 && !out[out.length - 1]!.trim()) out.pop();
  return out.join("\n");
}

// ---------------------------------------------------------------------------
// Full page normalization
// ---------------------------------------------------------------------------

export function normalizeNotionPage(
  page: NotionPage,
  blocks: NotionBlock[],
  options: NormalizeOptions = {},
): ArticleFull {
  const meta = normalizeNotionRow(page);

  if (!meta.draft) {
    requireField(meta.pubDate, "publishedAt");
  }
  requireField(meta.slug, "slug");

  const unsupported: string[] = [];
  const content = blocksToMarkdown(blocks, unsupported).trim();

  if (options.strict && unsupported.length > 0) {
    const uniq = Array.from(new Set(unsupported)).sort();
    throw new Error(
      `Unsupported Notion blocks found: ${uniq.join(", ")}\nEdit the Notion page to use only supported blocks.`,
    );
  }

  if (!options.strict && unsupported.length > 0) {
    const uniq = Array.from(new Set(unsupported)).sort();
    console.warn(
      `Unsupported Notion blocks stripped for "${meta.slug}": ${uniq.join(", ")}`,
    );
  }

  return { ...meta, content };
}
