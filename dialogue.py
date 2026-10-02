"""Have the local LLM write a 3-4 person script, voice it with ElevenLabs,
and produce one audio file plus an .srt subtitle file."""
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests
from elevenlabs.client import ElevenLabs

from main import get_api_key, get_llm_response

VOICE_POOL = {
    "Custom": "rCYFsCX2waxtHCgVD0e8",
    "Rachel": "dag0x2dW9i5XIWwJ5KlD",
    "Adam": "A48Y4xFQ2YpCgXhLXguK",
    "Antoni": "fgDJOgmENIR82PueQrVs",
    "Bella": "6aO1exAR9bDruq155LzQ",
    "Josh": "RIxlkEgVTlW2XuYWyYjL",
}
GAP_SECONDS = 0.35  # silence between lines
OUTPUT_DIR = Path("output")
SPEAKER_RANGE = (3, 4)  # random number of speakers per run, override with -n
LINES_PER_SPEAKER = 6

SCRIPT_PROMPT = """Write a short, natural-sounding dialogue between {n} people about: {topic}

Rules:
- Exactly {n} characters, each with a distinct first name.
- About {lines} lines in total. Everyone should speak several times and react to each other, not just take turns in a fixed order.
- Every line must be formatted exactly as: Name: what they say
- Only spoken words. No stage directions, no parentheses, no asterisks, no emojis, no title, no narration.
- Keep each line to one to three sentences."""


def generate_script(topic: str, n_speakers: int) -> list[tuple[str, str]]:
    prompt = SCRIPT_PROMPT.format(topic=topic, n=n_speakers, lines=n_speakers * LINES_PER_SPEAKER)
    raw = get_llm_response([{"role": "user", "content": prompt}], max_tokens=3000)
    return parse_script(raw, n_speakers)


def parse_script(raw: str, max_speakers: int) -> list[tuple[str, str]]:
    script = []
    for line in raw.splitlines():
        m = re.match(r"^\W*([A-Za-z][\w .'-]{0,30}?)\*{0,2}\s*:\s*\*{0,2}\s*(.+)$", line.strip())
        if not m:
            continue
        name = m.group(1).strip()
        text = re.sub(r"\([^)]*\)|\[[^\]]*\]|\*", "", m.group(2)).strip()
        if text:
            script.append((name, text))

    # Keep only the first N speakers seen, in case the model adds extras.
    speakers = list(dict.fromkeys(name for name, _ in script))[:max_speakers]
    return [(n, t) for n, t in script if n in speakers]


def synthesize_line(client: ElevenLabs, text: str, voice_id: str, path: Path) -> None:
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    with open(path, "wb") as f:
        for chunk in audio:
            f.write(chunk)


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def srt_time(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def build_audio_and_subs(parts: list[tuple[str, str, Path]], out_mp3: Path, out_srt: Path) -> None:
    """parts: (speaker, text, mp3 path). Concatenates with gaps, writes matching SRT."""
    gap = out_mp3.parent / "gap.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "anullsrc=r=44100:cl=mono", "-t", str(GAP_SECONDS), "-b:a", "128k", str(gap)],
        check=True,
    )

    concat_list = out_mp3.parent / "concat.txt"
    srt, t = [], 0.0
    with open(concat_list, "w") as f:
        for i, (speaker, text, path) in enumerate(parts, 1):
            d = duration(path)
            srt.append(f"{i}\n{srt_time(t)} --> {srt_time(t + d)}\n{speaker}: {text}\n")
            f.write(f"file '{path.resolve()}'\nfile '{gap.resolve()}'\n")
            t += d + GAP_SECONDS

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
        check=True,
    )
    out_srt.write_text("\n".join(srt), encoding="utf-8")


def main():
    args = sys.argv[1:]
    n_speakers = random.randint(*SPEAKER_RANGE)
    if len(args) >= 2 and args[0] == "-n":
        n_speakers = max(2, min(int(args[1]), len(VOICE_POOL)))
        args = args[2:]

    topic = " ".join(args).strip() or input("Topic for the dialogue: ").strip()
    if not topic:
        return

    print(f"Writing a {n_speakers}-person script with the local LLM...")
    try:
        script = generate_script(topic, n_speakers)
    except requests.RequestException as e:
        sys.exit(f"KoboldCPP error: {e}")
    names = list(dict.fromkeys(n for n, _ in script))
    if len(names) < 2:
        sys.exit("Couldn't find a multi-person script in the LLM output, try again.")

    voice_names = random.sample(list(VOICE_POOL), len(names))
    voices = {n: VOICE_POOL[v] for n, v in zip(names, voice_names)}
    for n, v in zip(names, voice_names):
        print(f"  {n} -> {v}")
    print()
    for n, t in script:
        print(f"{n}: {t}")

    run_dir = OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    (run_dir / "script.txt").write_text("\n".join(f"{n}: {t}" for n, t in script), encoding="utf-8")

    client = ElevenLabs(api_key=get_api_key())
    parts = []
    for i, (name, text) in enumerate(script, 1):
        print(f"Voicing line {i}/{len(script)}...")
        path = run_dir / f"line_{i:03}.mp3"
        synthesize_line(client, text, voices[name], path)
        parts.append((name, text, path))

    out_mp3, out_srt = run_dir / "dialogue.mp3", run_dir / "dialogue.srt"
    build_audio_and_subs(parts, out_mp3, out_srt)
    print(f"\nSaved {out_mp3} and {out_srt}")

    subprocess.run(["mpv", "--force-window=yes", f"--sub-file={out_srt}", str(out_mp3)])


if __name__ == "__main__":
    main()
