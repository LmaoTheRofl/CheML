import fs from "node:fs";
import { mkdir, rename, stat, unlink } from "node:fs/promises";
import path from "node:path";

const BASE_URL = "https://models.datalab.to";
const CHECKPOINTS = [
  "layout/2025_09_23",
  "text_detection/2025_05_07",
  "text_recognition/2025_09_23",
  "table_recognition/2025_02_18",
  "ocr_error_detection/2025_02_18",
];
const CHUNK_SIZE = 24 * 1024;
const PROGRESS_STEP = 100 * 1024 * 1024;

const cacheHome =
  process.env.XDG_CACHE_HOME ||
  path.resolve("runs", "tools", "cache");
const modelRoot = path.join(cacheHome, "datalab", "models");
const includeWeights = process.argv.includes("--include-weights");

async function fileSize(file) {
  try {
    return (await stat(file)).size;
  } catch {
    return null;
  }
}

async function expectedSize(url) {
  for (let attempt = 1; attempt <= 10; attempt += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15_000);
    try {
      const separator = url.includes("?") ? "&" : "?";
      const response = await fetch(`${url}${separator}head=${Date.now()}_${attempt}`, {
        method: "HEAD",
        headers: { "user-agent": "Mozilla/5.0", "cache-control": "no-cache" },
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`HEAD ${url}: ${response.status}`);
      }
      const length = response.headers.get("content-length");
      return length ? Number(length) : null;
    } catch (error) {
      if (attempt === 10) throw error;
    } finally {
      clearTimeout(timer);
    }
  }
}

async function download(url, destination, expected) {
  const current = await fileSize(destination);
  if (current !== null && (expected === null || current === expected)) {
    console.log(`skip ${path.relative(modelRoot, destination)}`);
    return;
  }

  await mkdir(path.dirname(destination), { recursive: true });
  const partial = `${destination}.part`;
  let offset = (await fileSize(partial)) || 0;

  if (expected === null) {
    throw new Error(`missing content-length for ${url}`);
  }

  let lastProgress = offset;
  while (offset < expected) {
    const end = Math.min(offset + CHUNK_SIZE - 1, expected - 1);
    let wrote = false;
    for (let attempt = 1; attempt <= 20; attempt += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 15_000);
      try {
        const separator = url.includes("?") ? "&" : "?";
        const response = await fetch(`${url}${separator}download=${Date.now()}_${attempt}`, {
          headers: {
            "user-agent": "Mozilla/5.0",
            "cache-control": "no-cache",
            range: `bytes=${offset}-${end}`,
          },
          signal: controller.signal,
        });
        if (response.status !== 206) {
          throw new Error(`GET ${url}: expected 206, got ${response.status}`);
        }
        const body = Buffer.from(await response.arrayBuffer());
        if (body.length !== end - offset + 1) {
          throw new Error(`short range ${body.length} != ${end - offset + 1}`);
        }
        fs.appendFileSync(partial, body);
        offset += body.length;
        if (offset - lastProgress >= PROGRESS_STEP || offset === expected) {
          console.log(
            `progress ${path.relative(modelRoot, destination)} `
            + `${offset}/${expected}`,
          );
          lastProgress = offset;
        }
        wrote = true;
        break;
      } catch (error) {
        if (attempt === 10) {
          throw error;
        }
        console.log(
          `retry ${attempt}/20 ${path.relative(modelRoot, destination)} `
          + `range=${offset}-${end}`,
        );
        await new Promise((resolve) => setTimeout(resolve, attempt * 2000));
      } finally {
        clearTimeout(timer);
      }
    }
    if (!wrote) {
      throw new Error(`failed to write range ${offset}-${end}`);
    }
  }

  const finalSize = await fileSize(partial);
  if (expected !== null && finalSize !== expected) {
    throw new Error(`${destination}: ${finalSize} != ${expected}`);
  }
  await rename(partial, destination);
  console.log(`downloaded ${path.relative(modelRoot, destination)} ${finalSize}`);
}

for (const checkpoint of CHECKPOINTS) {
  const manifestPath = path.join(modelRoot, checkpoint, "manifest.json");
  const manifestUrl = `${BASE_URL}/${checkpoint}/manifest.json`;
  await download(manifestUrl, manifestPath, await expectedSize(manifestUrl));
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  for (const fileName of manifest.files) {
    const destination = path.join(modelRoot, checkpoint, fileName);
    if ((await fileSize(destination)) !== null) {
      console.log(`skip ${path.relative(modelRoot, destination)}`);
      continue;
    }
    if (fileName === "model.safetensors" && !includeWeights) {
      throw new Error(`missing Marker model weight; rerun with --include-weights: ${destination}`);
    }
    const url = `${BASE_URL}/${checkpoint}/${fileName}`;
    const expected = await expectedSize(url);
    await download(url, destination, expected);
  }
}

console.log("datalab model cache complete");
