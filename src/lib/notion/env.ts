export type NotionEnv = {
  token: string;
  blogDbId: string;
};

function getEnvVar(name: string): string | undefined {
  // Vite injects .env/.env.local vars into import.meta.env at build time.
  // process.env is a fallback for CI/Cloudflare where vars are set directly.
  return (import.meta.env?.[name] as string | undefined) ?? process.env[name];
}

export function getNotionEnv(): NotionEnv {
  const token = getEnvVar("NOTION_TOKEN");
  const blogDbId = getEnvVar("NOTION_BLOG_DB_ID");

  if (!token || !blogDbId) {
    const json = {
      missingVars: {
        NOTION_TOKEN: !token,
        NOTION_BLOG_DB_ID: !blogDbId,
      },
      message: "Make sure all required Notion env vars are set.",
    };
    console.log("NOTION_ERROR_BODY", JSON.stringify(json, null, 2));
    throw new Error(
      [
        "Missing Notion env vars.",
        "Set vars in `.env`:",
        "- NOTION_TOKEN=ntn_...",
        "- NOTION_BLOG_DB_ID=...",
      ].join("\n"),
    );
  }

  return { token, blogDbId };
}

/**
 * Returns true if Notion env vars are configured. Use this to gracefully
 * skip Notion fetching when credentials aren't available.
 */
export function hasNotionEnv(): boolean {
  return !!getEnvVar("NOTION_TOKEN") && !!getEnvVar("NOTION_BLOG_DB_ID");
}
