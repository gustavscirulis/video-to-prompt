# v2p: video to prompt

Record yourself clicking through a product and saying what should change. `v2p` turns that recording into a prompt you can paste into Claude Code.

```
v2p ~/Desktop/walkthrough.mov
```

It does four things:

1. Transcribes the narration locally with whisper.cpp.
2. Asks Claude (through your existing `claude` CLI login) to pull out each feedback item and the moment it refers to.
3. Grabs a screenshot from the video at each of those moments.
4. Writes `prompt.md`, which cites each screenshot by absolute path, and copies it to the clipboard.

Paste the prompt into Claude Code and it reads the screenshots itself.

## Setup

```
brew install ffmpeg whisper-cpp
npm link            # puts `v2p` on your PATH
```

You also need [Claude Code](https://docs.claude.com/en/docs/claude-code), logged in.

The first run downloads the whisper model (about 550 MB) and a small voice-detection model to `~/.cache/v2p/`.

## Usage

```
v2p <video> [--out DIR] [--model opus|sonnet] [--lang en|auto] [--context "notes"]
```

- `--out`: the output folder. The default is `<video-name>.v2p/` next to the video.
- `--model`: the Claude model used for extraction. The default is Claude Code's default.
- `--lang`: the spoken language. The default is `en`. Use `auto` to detect it.
- `--context`: extra hints for the extractor, for example `"this is the billing settings in our admin app"`.

## Output

```
walkthrough.v2p/
  prompt.md         # paste this into Claude Code (already on your clipboard)
  frames/01.png …   # screenshots cited by prompt.md
  items.json        # structured feedback items
  transcript.json   # cleaned transcript segments
  audio.wav, whisper.json
```

A re-run reuses `transcript.json`, so trying another `--model` or `--context` only repeats the Claude step. Delete `transcript.json` to transcribe again.

Keep the output folder until Claude Code has finished. The prompt points at the files inside it.

## Recording tips

- Say what you are looking at ("on the settings page, this save button…"). The screenshot is taken while you talk about something, so stay on that screen while you describe it.
- Give specifics out loud: the exact copy, the colour, where something should move.
- Pauses are fine, because silence is filtered out.
