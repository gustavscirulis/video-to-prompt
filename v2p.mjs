#!/usr/bin/env node
import { spawn, spawnSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, extname, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { parseArgs } from "node:util";

const MODEL_DIR = join(homedir(), ".cache", "v2p");
const WHISPER_MODEL = {
  file: "ggml-large-v3-turbo-q5_0.bin",
  url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
};
const VAD_MODEL = {
  file: "ggml-silero-v5.1.2.bin",
  url: "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin",
};
const MAX_IMAGE_EDGE = 1568;

const USAGE = `Usage: v2p <video> [options]

Turns a narrated product walkthrough into a prompt for Claude Code.

Options:
  --out <dir>        Output folder (default: <video-name>.v2p next to the video)
  --model <name>     Claude model for extraction, e.g. opus or sonnet
  --lang <code>      Spoken language for transcription (default: en, or "auto")
  --context <text>   Extra notes for the extractor (product, repo, focus area)
  -h, --help         Show this help`;

class UserError extends Error {}

function run(cmd, args, { input, onStderr } = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(cmd, args);
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (d) => (stdout += d));
    child.stderr.setEncoding("utf8").on("data", (d) => {
      stderr += d;
      onStderr?.(d);
    });
    child.on("error", (err) => reject(new UserError(`Failed to run ${cmd}: ${err.message}`)));
    child.on("close", (code) => {
      if (code === 0) return resolvePromise(stdout);
      const tail = (stderr || stdout).trim().split("\n").slice(-15).join("\n");
      reject(new UserError(`${cmd} exited with code ${code}:\n${tail}`));
    });
    child.stdin.end(input);
  });
}

function has(cmd) {
  return spawnSync("which", [cmd], { encoding: "utf8" }).status === 0;
}

function log(msg) {
  process.stderr.write(`${msg}\n`);
}

const BAR_WIDTH = 24;
const SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";

async function step(label, fn) {
  const tty = process.stderr.isTTY;
  const start = Date.now();
  let fraction = null;
  let tick = 0;

  const renderBar = (width) => {
    if (fraction === null) {
      const block = Math.max(2, Math.round(width / 4));
      const pos = tick % (width + block);
      return Array.from({ length: width }, (_, i) => (i >= pos - block && i < pos ? "█" : "░")).join("");
    }
    const filled = Math.round(Math.min(Math.max(fraction, 0), 1) * width);
    return "█".repeat(filled) + "░".repeat(width - filled);
  };

  const draw = () => {
    tick++;
    const cols = process.stderr.columns || 80;
    const spin = SPINNER[tick % SPINNER.length];
    const pct = fraction === null ? "" : `${Math.floor(fraction * 100)}%`;
    const elapsed = fmtTime((Date.now() - start) / 1000);
    const head = `${spin} ${label} `;
    const tail = ` ${pct.padStart(4)}  ${elapsed}`;
    const barWidth = Math.min(BAR_WIDTH, cols - 1 - head.length - tail.length);
    let line = barWidth >= 8 ? head + renderBar(barWidth) + tail : `${spin} ${label} ${pct} ${elapsed}`;
    // A line wider than the terminal wraps, and \r can't reach the earlier rows, leaving stale copies behind.
    line = line.slice(0, cols - 1);
    process.stderr.write(`\r\x1b[2K${line}`);
  };
  const finish = (symbol) => {
    clearInterval(timer);
    const secs = ((Date.now() - start) / 1000).toFixed(1);
    if (tty) process.stderr.write(`\r\x1b[2K${symbol} ${label} (${secs}s)\n`);
    else log(`${symbol} ${label} (${secs}s)`);
  };

  const timer = tty ? setInterval(draw, 80) : null;
  if (tty) draw();
  try {
    const result = await fn({ update: (f) => (fraction = f) });
    finish("✓");
    return result;
  } catch (err) {
    finish("✗");
    throw err;
  }
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function fmtTimePrecise(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function preflight() {
  const deps = [
    ["ffmpeg", "brew install ffmpeg"],
    ["ffprobe", "brew install ffmpeg"],
    ["whisper-cli", "brew install whisper-cpp"],
    ["claude", "see https://docs.claude.com/en/docs/claude-code"],
  ];
  const missing = deps.filter(([cmd]) => !has(cmd));
  if (missing.length) {
    throw new UserError(`Missing dependencies:\n${missing.map(([c, how]) => `  ${c}  →  ${how}`).join("\n")}`);
  }
}

async function ensureModel({ file, url }) {
  const path = join(MODEL_DIR, file);
  if (existsSync(path)) return path;
  mkdirSync(MODEL_DIR, { recursive: true });
  await step(`Downloading ${file.startsWith("ggml-silero") ? "VAD model" : "whisper model"}`, async ({ update }) => {
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new UserError(`Download failed (${res.status}) for ${url}`);
    const total = Number(res.headers.get("content-length")) || 0;
    let received = 0;
    const body = Readable.fromWeb(res.body);
    body.on("data", (chunk) => {
      received += chunk.length;
      if (total) update(received / total);
    });
    const part = `${path}.part`;
    await pipeline(body, createWriteStream(part));
    renameSync(part, path);
  });
  return path;
}

async function probe(video) {
  const info = JSON.parse(await run("ffprobe", ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", video]));
  const vStream = info.streams.find((s) => s.codec_type === "video");
  const aStream = info.streams.find((s) => s.codec_type === "audio");
  if (!vStream) throw new UserError("No video stream found in the file.");
  if (!aStream) throw new UserError("The video has no audio track, so there is no narration to transcribe.");
  const duration = Number(info.format.duration || vStream.duration);
  if (!duration) throw new UserError("Could not read the video duration.");
  return { duration, width: vStream.width, height: vStream.height };
}

function cleanSegments(whisperJson) {
  return whisperJson.transcription
    .map((seg) => ({
      start: seg.offsets.from / 1000,
      end: seg.offsets.to / 1000,
      text: seg.text
        .replace(/\[[^\]]*\]|\([^)]*\)|\*[^*]*\*/g, "")
        .replace(/\s+/g, " ")
        .trim(),
    }))
    .filter((seg) => /[\p{L}\p{N}]/u.test(seg.text));
}

async function transcribe(video, outDir, lang) {
  const cached = join(outDir, "transcript.json");
  if (existsSync(cached)) {
    log("✓ Transcribing (cached, delete transcript.json to redo)");
    return JSON.parse(readFileSync(cached, "utf8"));
  }

  const modelPath = await ensureModel(WHISPER_MODEL);
  const vadPath = await ensureModel(VAD_MODEL);

  const wav = join(outDir, "audio.wav");
  await step("Extracting audio", () =>
    run("ffmpeg", ["-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav]),
  );

  const whisperBase = join(outDir, "whisper");
  await step("Transcribing", ({ update }) =>
    run(
      "whisper-cli",
      ["-m", modelPath, "-f", wav, "-l", lang, "--vad", "-vm", vadPath, "-sns", "-np", "-pp", "-oj", "-of", whisperBase],
      {
        onStderr: (chunk) => {
          const matches = [...chunk.matchAll(/progress =\s*(\d+)%/g)];
          if (matches.length) update(Number(matches.at(-1)[1]) / 100);
        },
      },
    ),
  );
  const segments = cleanSegments(JSON.parse(readFileSync(`${whisperBase}.json`, "utf8")));
  if (!segments.length) throw new UserError("No speech detected in the recording. Was the mic on?");
  writeFileSync(cached, JSON.stringify(segments, null, 2));
  return segments;
}

const ITEMS_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["summary", "items", "open_questions"],
  properties: {
    summary: { type: "string" },
    items: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["title", "type", "priority", "quote", "request", "frame_times"],
        properties: {
          title: { type: "string" },
          type: { type: "string", enum: ["bug", "ux", "visual", "copy", "feature", "other"] },
          priority: { type: "string", enum: ["high", "medium", "low"] },
          quote: { type: "string" },
          request: { type: "string" },
          frame_times: { type: "array", minItems: 1, maxItems: 2, items: { type: "number" } },
        },
      },
    },
    open_questions: { type: "array", items: { type: "string" } },
  },
};

const SYSTEM_PROMPT =
  "You turn transcripts of narrated product walkthroughs into precise, actionable feedback items for a coding agent. You output only the requested structured data.";

function extractionPrompt(segments, duration, context) {
  const transcript = segments
    .map((s) => `[${fmtTimePrecise(s.start)}–${fmtTimePrecise(s.end)}] ${s.text}`)
    .join("\n");
  return `Below is the transcript of a screen recording (${duration.toFixed(1)}s long). The speaker clicks through a product and talks about what should change.

Extract every distinct piece of feedback as an item:
- title: a short name for the change.
- type and priority: priority comes from how strongly the speaker feels about it, defaulting to medium.
- quote: the speaker's own words, copied verbatim from the transcript. Trim filler words.
- request: a clear, specific, imperative instruction a coding agent can implement. Keep every concrete detail the speaker gave (exact copy, colours, sizes, positions, behaviour).
- frame_times: 1–2 timestamps in seconds where the thing being discussed is on screen. Usually that is a moment inside the segment where the speaker refers to it ("this button", "here"). It can also be slightly after. Avoid the end of a segment where they may already be navigating away. Use a second timestamp only when two screens are needed, such as before and after or a flow across two pages.

Rules:
- Merge feedback that spans several segments into one item. Split unrelated requests into separate items.
- Ignore pure narration ("now I'm opening settings"), filler and praise that asks for nothing.
- Never invent feedback the speaker did not give.
- Put anything ambiguous that the coding agent should confirm into open_questions. Leave it empty if nothing is unclear.
- summary: 1–2 sentences on what was reviewed and the overall direction.
${context ? `\nContext from the user:\n${context}\n` : ""}
Transcript (timestamps are m:ss.s):
${transcript}`;
}

async function extractFeedback(segments, duration, { model, context }) {
  const args = [
    "-p",
    "--output-format", "json",
    "--json-schema", JSON.stringify(ITEMS_SCHEMA),
    "--system-prompt", SYSTEM_PROMPT,
    "--tools", "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--setting-sources", "",
    "--no-session-persistence",
  ];
  if (model) args.push("--model", model);
  const stdout = await step("Extracting feedback", () =>
    run("claude", args, { input: extractionPrompt(segments, duration, context) }),
  );
  const envelope = JSON.parse(stdout);
  if (envelope.is_error) throw new UserError(`Claude returned an error: ${envelope.result}`);
  const data = envelope.structured_output ?? JSON.parse(envelope.result);
  if (!Array.isArray(data.items)) throw new UserError("Claude returned an unexpected response shape.");
  return data;
}

async function extractFrames(video, outDir, data, { duration, width, height }) {
  const framesDir = join(outDir, "frames");
  rmSync(framesDir, { recursive: true, force: true });
  mkdirSync(framesDir, { recursive: true });

  const clamp = (t) => Math.min(Math.max(Number(t) || 0, 0), Math.max(duration - 0.1, 0));
  const unique = [];
  for (const t of data.items.flatMap((item) => item.frame_times.map(clamp)).sort((a, b) => a - b)) {
    if (!unique.length || t - unique.at(-1) >= 0.5) unique.push(t);
  }

  const scale =
    Math.max(width, height) > MAX_IMAGE_EDGE
      ? ["-vf", `scale=${MAX_IMAGE_EDGE}:${MAX_IMAGE_EDGE}:force_original_aspect_ratio=decrease`]
      : [];

  const frames = await step(`Extracting ${unique.length} frame(s)`, async ({ update }) => {
    const out = [];
    for (const [i, time] of unique.entries()) {
      const path = join(framesDir, `${String(i + 1).padStart(2, "0")}.png`);
      await run("ffmpeg", ["-y", "-v", "error", "-ss", time.toFixed(3), "-i", video, "-frames:v", "1", ...scale, path]);
      out.push({ time, path });
      update(out.length / unique.length);
    }
    return out;
  });

  const frameFor = (t) => frames.reduce((best, f) => (Math.abs(f.time - t) < Math.abs(best.time - t) ? f : best));
  for (const item of data.items) {
    item.frames = [...new Map(item.frame_times.map(clamp).map(frameFor).map((f) => [f.path, f])).values()];
  }
}

function renderPrompt(data, segments) {
  const lines = [
    "# Product feedback from a screen recording",
    "",
    "I recorded a walkthrough of the product and narrated feedback. Implement the changes below.",
    "Each item has screenshot(s) from the recording. Read them with the Read tool before working on that item, and find the relevant code yourself.",
    'If anything is unclear, check "Open questions" and ask me rather than guessing.',
    "",
    "## Summary",
    "",
    data.summary,
    "",
    "## Feedback items",
  ];

  data.items.forEach((item, i) => {
    lines.push("", `### ${i + 1}. ${item.title} (${item.type} · ${item.priority})`, "");
    for (const f of item.frames) lines.push(`- Screenshot: \`${f.path}\` (${fmtTime(f.time)} in the recording)`);
    lines.push(`- What I said: "${item.quote}"`, `- Change: ${item.request}`);
  });

  if (data.open_questions.length) {
    lines.push("", "## Open questions", "", ...data.open_questions.map((q) => `- ${q}`));
  }

  lines.push(
    "",
    "## Full transcript (for context)",
    "",
    ...segments.map((s) => `[${fmtTime(s.start)}] ${s.text}`),
    "",
  );
  return lines.join("\n");
}

function outputDirFor(video) {
  const name = basename(video, extname(video)).replace(/[^\w.-]+/g, "-").replace(/^-+|-+$/g, "") || "recording";
  return join(dirname(video), `${name}.v2p`);
}

async function main() {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      out: { type: "string" },
      model: { type: "string" },
      lang: { type: "string", default: "en" },
      context: { type: "string" },
      help: { type: "boolean", short: "h" },
    },
  });
  if (values.help || positionals.length !== 1) {
    console.log(USAGE);
    process.exit(values.help ? 0 : 1);
  }

  const video = resolve(positionals[0]);
  if (!existsSync(video)) throw new UserError(`File not found: ${video}`);
  preflight();

  const outDir = resolve(values.out ?? outputDirFor(video));
  mkdirSync(outDir, { recursive: true });

  const meta = await probe(video);
  const segments = await transcribe(video, outDir, values.lang);
  const data = await extractFeedback(segments, meta.duration, values);
  if (!data.items.length) throw new UserError("Claude found no actionable feedback in the transcript.");
  await extractFrames(video, outDir, data, meta);

  writeFileSync(join(outDir, "items.json"), JSON.stringify(data, null, 2));
  const promptPath = join(outDir, "prompt.md");
  const prompt = renderPrompt(data, segments);
  writeFileSync(promptPath, prompt);

  let copied = false;
  if (process.platform === "darwin") {
    copied = spawnSync("pbcopy", { input: prompt }).status === 0;
  }

  log(`\nDone: ${data.items.length} feedback item(s)${copied ? ". Prompt copied to clipboard, paste it into Claude Code." : ""}`);
  console.log(promptPath);
}

main().catch((err) => {
  log(err instanceof UserError ? `\nError: ${err.message}` : err.stack);
  process.exit(1);
});
