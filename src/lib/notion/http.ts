import { getNotionEnv } from "./env";

const NOTION_API_BASE = "https://api.notion.com/v1";
const NOTION_VERSION = "2022-06-28";

export class NotionHttpError extends Error {
  status: number;
  body: unknown;
  requestId?: string;

  constructor(message: string, status: number, body: unknown, requestId?: string) {
    super(message);
    this.status = status;
    this.body = body;
    this.requestId = requestId;
  }
}

export async function notionFetchJson<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const { token } = getNotionEnv();

  const res = await fetch(`${NOTION_API_BASE}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      "Notion-Version": NOTION_VERSION,
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });

  const requestId = res.headers.get("x-notion-request-id") ?? undefined;
  const text = await res.text();
  let json: unknown = null;
  if (text) {
    try {
      json = JSON.parse(text) as unknown;
    } catch {
      json = { raw: text };
    }
  }

  if (!res.ok) {
    throw new NotionHttpError(
      `Notion API error ${res.status}${res.statusText ? ` ${res.statusText}` : ""} for ${path}${
        requestId ? ` (request id: ${requestId})` : ""
      }`,
      res.status,
      json,
      requestId,
    );
  }

  return json as T;
}
