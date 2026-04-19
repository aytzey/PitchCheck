import { existsSync } from "node:fs";
import { mkdir, rename, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const apiDir = join(root, "app", "api");
const disabledApiDir = join(root, ".next-tauri-disabled-api", "api");

async function moveIfExists(from, to) {
  if (!existsSync(from)) return false;
  await mkdir(dirname(to), { recursive: true });
  await rm(to, { recursive: true, force: true });
  await rename(from, to);
  return true;
}

function runNextBuild() {
  return new Promise((resolve, reject) => {
    const child = spawn("npx", ["next", "build"], {
      cwd: root,
      stdio: "inherit",
      env: {
        ...process.env,
        TAURI_BUILD: "1",
      },
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`next build exited with code ${code}`));
    });
  });
}

const movedApi = await moveIfExists(apiDir, disabledApiDir);

try {
  await runNextBuild();
} finally {
  if (movedApi) {
    await moveIfExists(disabledApiDir, apiDir);
  }
  await rm(join(root, ".next-tauri-disabled-api"), {
    recursive: true,
    force: true,
  });
}
