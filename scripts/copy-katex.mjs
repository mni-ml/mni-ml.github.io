import { cpSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const katexDir = dirname(require.resolve('katex/package.json'));
const publicKatex = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'katex');

mkdirSync(publicKatex, { recursive: true });
cpSync(join(katexDir, 'dist', 'katex.min.css'), join(publicKatex, 'katex.min.css'));
cpSync(join(katexDir, 'dist', 'fonts'), join(publicKatex, 'fonts'), { recursive: true });

console.log('Copied KaTeX CSS and fonts to public/katex/');
