/**
 * 把 marked 与 lucide 的 UMD 构建复制到 static/vendor/。
 * 部署时不需要 Node，所以生成物必须提交；源文件缺失时直接失败，不静默回退 CDN。
 */
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outDir = resolve(root, "static", "vendor");

function copyFirst(candidates, outName) {
  const src = candidates.map((p) => resolve(root, p)).find((p) => existsSync(p));
  if (!src) {
    throw new Error(
      `找不到 ${outName} 的源文件，尝试过：\n${candidates.join("\n")}\n` +
        `请先在 rag_chat/ 目录运行 npm install。`,
    );
  }
  const dest = resolve(outDir, outName);
  copyFileSync(src, dest);
  console.log(`vendor: ${src} -> ${dest}`);
}

mkdirSync(outDir, { recursive: true });

copyFirst(
  ["node_modules/marked/marked.min.js", "node_modules/marked/lib/marked.umd.js"],
  "marked.min.js",
);

copyFirst(
  ["node_modules/lucide/dist/umd/lucide.min.js", "node_modules/lucide/dist/umd/lucide.js"],
  "lucide.min.js",
);
