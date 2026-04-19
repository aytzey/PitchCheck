import { spawn } from "node:child_process";
import process from "node:process";

const args =
  process.platform === "linux"
    ? ["tauri", "build", "--bundles", "deb,rpm"]
    : ["tauri", "build"];

const child = spawn("npx", args, {
  stdio: "inherit",
  shell: process.platform === "win32",
});

child.on("exit", (code) => {
  process.exit(code ?? 1);
});
