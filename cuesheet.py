"""MOSS SFX Cue Sheet node.

Reads a JSON cue sheet (what sound, when), generates every sound with MOSS-SoundEffect v2.0,
places them on a timeline and returns the mix as AUDIO.
It also writes, into one output folder:
  - one WAV per sound (already trimmed / faded / volume applied)
  - mix.wav (all sounds placed on the timeline)
  - timeline.xml (FCP 7 XML: video track + one audio track per sound) for FilmCraft, Premiere, Resolve

Cue sheet format (times in seconds):
{
  "duration": 5.06,                      optional, otherwise taken from the video input
  "sounds": [
    {"name": "room", "type": "continuous", "prompt": "...", "start": 0.0, "end": 5.06, "volume_db": -10},
    {"name": "heel", "type": "oneshot", "prompt": "...", "length": 1.0, "times": [1.31, 1.75], "volume_db": 0},
    {"name": "heel_far", "type": "oneshot", "use": "heel", "times": [3.88], "volume_db": -12}
  ]
}
Optional keys per sound: "negative" (replaces the node's negative prompt), "seed" (fixed seed),
"fade" (seconds, continuous sounds), "normalize" (true/false, default true).
"""

from __future__ import annotations

import json
import math
import os
import re
from urllib.parse import quote
from xml.sax.saxutils import escape

import numpy as np

import folder_paths

from .nodes import _autocast_dtype, _compute_dtype, _free_comfy_models, _free_cuda, _get_pipeline, _progress

MAX_GEN_SECONDS = 30.0
PEAK_DB = -1.0
CUE_FROM_TEXT = "text box (cue_sheet)"
CUE_FROM_INPUT = "connected input (cue_sheet_in, e.g. Qwen3-VL)"


def _clip_to_duration(sounds, duration):
    """Drop times outside the video and cut continuous sounds at the end of the video."""
    kept = []
    for s in sounds:
        if s["type"] == "oneshot":
            times = [float(t) for t in s["times"] if 0.0 <= float(t) < duration]
            if not times:
                continue
            s["times"] = times
        else:
            start = max(0.0, float(s.get("start", 0.0)))
            if start >= duration:
                continue
            s["start"] = start
            end = s.get("end")
            s["end"] = duration if end is None else min(float(end), duration)
            if s["end"] <= start:
                continue
        kept.append(s)
    names = {s["name"] for s in kept if not s.get("use")}
    return [s for s in kept if not s.get("use") or s["use"] in names]


# ---------------------------------------------------------------- cue sheet

def _parse_cue_sheet(text):
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("The cue sheet text box is empty. Paste a JSON cue sheet into cue_sheet, "
                         "or set cue_source to 'connected input' to use the Qwen3-VL cue sheet.")
    # Allow a pasted ```json ... ``` block, and curly quotes that some apps put in when copying.
    raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw).strip()
    raw = raw.translate(str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"}))
    # An LLM (Qwen3-VL) may add words around the JSON or leave trailing commas.
    if not raw.startswith("{"):
        s, e = raw.find("{"), raw.rfind("}")
        if s >= 0 and e > s:
            raw = raw[s:e + 1]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The cue sheet is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}")
    sounds = data.get("sounds") if isinstance(data, dict) else None
    if not sounds:
        raise ValueError('The cue sheet has no "sounds" list.')

    names = set()
    for i, s in enumerate(sounds):
        name = str(s.get("name") or f"sound{i + 1}")
        name = re.sub(r"[^0-9A-Za-z_\-]+", "_", name)
        if name in names:
            raise ValueError(f'Two sounds share the name "{name}". Every sound needs its own name.')
        names.add(name)
        s["name"] = name
        kind = str(s.get("type", "")).lower()
        if kind not in ("continuous", "oneshot"):
            raise ValueError(f'Sound "{name}": type must be "continuous" or "oneshot".')
        s["type"] = kind
        if not s.get("use") and not str(s.get("prompt", "")).strip():
            raise ValueError(f'Sound "{name}": needs a "prompt" (or "use" to reuse another oneshot sound).')
        if kind == "continuous":
            start, end = float(s.get("start", 0.0)), s.get("end")
            if end is None:
                s["end"] = None
            elif float(end) <= start:
                raise ValueError(f'Sound "{name}": "end" must be later than "start".')
        else:
            times = s.get("times")
            if not isinstance(times, list) or not times:
                raise ValueError(f'Sound "{name}": oneshot sounds need a "times" list, e.g. [1.2, 1.7].')
    for s in sounds:
        if s.get("use") and s["use"] not in names:
            raise ValueError(f'Sound "{s["name"]}": "use" points to "{s["use"]}", which is not in the cue sheet.')
    return data


# ---------------------------------------------------------------- audio helpers (numpy, shape (2, T))

def _to_stereo(wave):
    w = np.asarray(wave, dtype=np.float32)
    if w.ndim == 3:
        w = w[0]
    if w.ndim == 1:
        w = w[None]
    if w.shape[0] == 1:
        w = np.repeat(w, 2, axis=0)
    return np.ascontiguousarray(w[:2])


def _normalize(w, peak_db=PEAK_DB):
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    if peak < 1e-6:
        return w
    return w * (10 ** (peak_db / 20.0) / peak)


def _fade(w, sr, fade_in, fade_out):
    n = w.shape[1]
    a = min(int(fade_in * sr), n // 2)
    b = min(int(fade_out * sr), n - a)
    if a > 0:
        w[:, :a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)
    if b > 0:
        w[:, n - b:] *= np.linspace(1.0, 0.0, b, dtype=np.float32)
    return w


def _trim_to_onset(w, sr, threshold_db=-20.0, pre=0.004):
    """Cut the silence before the hit, so the sound starts exactly where it is placed."""
    env = np.max(np.abs(w), axis=0)
    peak = float(env.max()) if env.size else 0.0
    if peak < 1e-6:
        return w
    above = np.nonzero(env >= peak * (10 ** (threshold_db / 20.0)))[0]
    start = max(0, int(above[0]) - int(pre * sr))
    return w[:, start:]


def _make_oneshot(w, sr, length):
    w = _trim_to_onset(w, sr)
    n = max(1, int(length * sr))
    w = np.array(w[:, :n], copy=True)
    return _fade(w, sr, 0.002, max(0.02, w.shape[1] / sr * 0.3))


def _make_continuous(w, sr, length, fade):
    n = max(1, int(round(length * sr)))
    if w.shape[1] >= n:
        out = np.array(w[:, :n], copy=True)
    else:  # loop with a 1 s crossfade (only when the sound is longer than 30 s)
        xf = min(int(1.0 * sr), w.shape[1] // 3)
        out = np.array(w, copy=True)
        ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32) if xf > 0 else None
        while out.shape[1] < n:
            if xf > 0:
                mixed = out[:, -xf:] * (1.0 - ramp) + w[:, :xf] * ramp
                out = np.concatenate([out[:, :-xf], mixed, w[:, xf:]], axis=1)
            else:
                out = np.concatenate([out, w], axis=1)
        out = out[:, :n]
    return _fade(out, sr, fade, fade)


def _place(mix, clip, start_sample):
    if start_sample >= mix.shape[1]:
        return
    s = max(0, start_sample)
    c = clip[:, s - start_sample:]
    e = min(mix.shape[1], s + c.shape[1])
    mix[:, s:e] += c[:, : e - s]


def _write_wav(path, w, sr):
    import soundfile as sf

    sf.write(path, np.clip(w.T, -1.0, 1.0), sr, subtype="PCM_16")


# ---------------------------------------------------------------- FCP 7 XML

def _pathurl(path):
    p = os.path.abspath(path).replace("\\", "/")
    if not p.startswith("/"):
        p = "/" + p  # C:/... -> /C:/...
    return "file://localhost" + quote(p, safe="/:")


def _rate_xml(fps):
    timebase = int(round(fps))
    ntsc = "TRUE" if abs(fps - timebase) > 0.01 else "FALSE"
    if ntsc == "TRUE":
        timebase = int(math.ceil(fps))
    return f"<rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>", timebase


def _timeline_xml(name, fps, total_frames, width, height, sr, video_path, tracks):
    """tracks: list of (track_name, wav_path, clip_frames, [start_frames])"""
    rate, _ = _rate_xml(fps)
    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<!DOCTYPE xmeml>", '<xmeml version="5">',
           '<sequence id="sequence-1">', f"<name>{escape(name)}</name>", f"<duration>{total_frames}</duration>", rate,
           "<media>", "<video>", "<format><samplecharacteristics>", rate,
           f"<width>{width}</width><height>{height}</height>", "</samplecharacteristics></format>"]
    if video_path:
        vname = escape(os.path.basename(video_path))
        out += ["<track>", '<clipitem id="clipitem-video">', f"<name>{vname}</name>",
                f"<duration>{total_frames}</duration>", rate,
                f"<start>0</start><end>{total_frames}</end><in>0</in><out>{total_frames}</out>",
                '<file id="file-video">', f"<name>{vname}</name>",
                f"<pathurl>{escape(_pathurl(video_path))}</pathurl>", rate, f"<duration>{total_frames}</duration>",
                "<media><video><samplecharacteristics>", rate,
                f"<width>{width}</width><height>{height}</height>",
                "</samplecharacteristics></video></media>", "</file>", "</clipitem>", "</track>"]
    out += ["</video>", "<audio>",
            f"<format><samplecharacteristics><depth>16</depth><samplerate>{sr}</samplerate>"
            "</samplecharacteristics></format>"]
    clip_id = 0
    for t_index, (tname, wav, clip_frames, starts) in enumerate(tracks, 1):
        fid = f"file-{escape(tname)}"
        out.append("<track>")
        for k, st in enumerate(starts):
            end = min(st + clip_frames, total_frames)
            if end <= st:
                continue
            clip_id += 1
            out += [f'<clipitem id="clipitem-{clip_id}">', f"<name>{escape(tname)}</name>",
                    f"<duration>{clip_frames}</duration>", rate,
                    f"<start>{st}</start><end>{end}</end><in>0</in><out>{end - st}</out>"]
            if k == 0:
                out += [f'<file id="{fid}">', f"<name>{escape(os.path.basename(wav))}</name>",
                        f"<pathurl>{escape(_pathurl(wav))}</pathurl>", rate, f"<duration>{clip_frames}</duration>",
                        "<media><audio><samplecharacteristics><depth>16</depth>"
                        f"<samplerate>{sr}</samplerate></samplecharacteristics>"
                        "<channelcount>2</channelcount></audio></media>", "</file>"]
            else:
                out.append(f'<file id="{fid}"/>')
            out += ["<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack>",
                    "</clipitem>"]
        out.append("</track>")
    out += ["</audio>", "</media>", "</sequence>", "</xmeml>"]
    return "\n".join(out)


def _next_folder(prefix):
    base = folder_paths.get_output_directory()
    prefix = str(prefix or "sfx/cue").strip().replace("\\", "/").strip("/") or "sfx/cue"
    parent, stem = os.path.split(prefix)
    parent_dir = os.path.join(base, parent)
    os.makedirs(parent_dir, exist_ok=True)
    n = 1
    pat = re.compile(re.escape(stem) + r"_(\d{5})$")
    for d in os.listdir(parent_dir):
        m = pat.match(d)
        if m:
            n = max(n, int(m.group(1)) + 1)
    folder = os.path.join(parent_dir, f"{stem}_{n:05d}")
    os.makedirs(folder, exist_ok=True)
    return folder


def _video_info(video):
    path, fps, dur, w, h = None, None, None, 1920, 1080
    if video is None:
        return path, fps, dur, w, h
    try:
        src = video.get_stream_source()
        if isinstance(src, str) and os.path.isfile(src):
            path = os.path.abspath(src)
    except Exception:
        pass
    try:
        fps = float(video.get_frame_rate())
    except Exception:
        pass
    try:
        dur = float(video.get_duration())
    except Exception:
        pass
    try:
        w, h = video.get_dimensions()
    except Exception:
        pass
    return path, fps, dur, int(w), int(h)


# ---------------------------------------------------------------- node

class MossSfxCueSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moss_model": ("MOSS_SFX_MODEL",),
                "cue_source": ([CUE_FROM_TEXT, CUE_FROM_INPUT], {
                    "default": CUE_FROM_TEXT,
                    "tooltip": "Which cue sheet to use: the text box below, or the one connected to cue_sheet_in "
                               "(e.g. written by Qwen3-VL). The unused one is not run."}),
                "cue_sheet": ("STRING", {"multiline": True, "default": "",
                                         "tooltip": "JSON cue sheet: which sounds, and when (seconds)."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "music, voice, speech"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 250, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sigma_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "filename_prefix": ("STRING", {"default": "sfx/cue",
                                               "tooltip": "Output folder inside ComfyUI/output (a number is added)."}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "The source video. Used for length, fps and the XML video track."}),
                "cue_sheet_in": ("STRING", {"forceInput": True, "lazy": True,
                                            "tooltip": "Cue sheet text from another node (Generate Text)."}),
                "unload_other_models": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def check_lazy_status(cls, cue_source=CUE_FROM_TEXT, cue_sheet_in=None, **kwargs):
        # Only run the Qwen3-VL branch when it is selected.
        if cue_source == CUE_FROM_INPUT and cue_sheet_in is None:
            return ["cue_sheet_in"]
        return []

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "run"
    CATEGORY = "audio/MOSS SFX"
    OUTPUT_NODE = True
    DESCRIPTION = ("Generates every sound in a cue sheet with MOSS, places them on the timeline, "
                   "and saves WAV stems, mix.wav and an FCP 7 XML timeline.")

    def run(self, moss_model, cue_source, cue_sheet, negative_prompt, steps, cfg_scale, sigma_shift, seed,
            filename_prefix, video=None, cue_sheet_in=None, unload_other_models=True):
        import torch

        if cue_source == CUE_FROM_INPUT:
            if not str(cue_sheet_in or "").strip():
                raise ValueError("cue_source is set to the connected input, but nothing came in through "
                                 "cue_sheet_in. Connect Generate Text (or switch cue_source to the text box).")
            cue_text = cue_sheet_in
        else:
            cue_text = cue_sheet
        cue = _parse_cue_sheet(cue_text)
        vpath, vfps, vdur, width, height = _video_info(video)
        fps = float(vfps or cue.get("fps") or 24.0)
        duration = vdur or cue.get("duration")
        if duration:
            cue["sounds"] = _clip_to_duration(cue["sounds"], float(duration))
        sounds = cue["sounds"]
        if not sounds:
            raise ValueError("No sound in the cue sheet falls inside the video length.")
        if not duration:
            ends = []
            for s in sounds:
                if s["type"] == "continuous" and s.get("end") is not None:
                    ends.append(float(s["end"]))
                elif s["type"] == "oneshot":
                    ends.append(max(float(t) for t in s["times"]) + float(s.get("length", 1.0)))
            duration = max(ends) if ends else 5.0
        duration = float(duration)

        if unload_other_models:
            _free_comfy_models()
        pipe = _get_pipeline(moss_model)
        sr = int(pipe.sample_rate)
        clips = {}
        lines = []

        try:
            for i, s in enumerate(sounds):
                if s.get("use"):
                    continue
                if s["type"] == "continuous":
                    start = float(s.get("start", 0.0))
                    end = float(s["end"]) if s.get("end") is not None else duration
                    length = max(0.1, min(end, duration) - start)
                    gen_sec = min(length + 0.2, MAX_GEN_SECONDS)
                else:
                    length = float(s.get("length", 1.0))
                    # MOSS always denoises a 30 s latent, so a longer request costs no extra time
                    # and leaves room for the hit; the onset is found and trimmed afterwards.
                    gen_sec = min(max(length + 1.0, 2.0), MAX_GEN_SECONDS)
                neg = s.get("negative", negative_prompt)
                sd = int(s["seed"]) if s.get("seed") is not None else (int(seed) + i) % (2 ** 63)
                print(f"[MOSS-SFX] Cue {i + 1}/{len(sounds)} '{s['name']}' ({s['type']}, {gen_sec:.1f}s, seed {sd})",
                      flush=True)
                with _autocast_dtype(_compute_dtype(moss_model)):
                    wave = pipe(
                        prompt=str(s["prompt"]).strip(),
                        seconds=float(gen_sec),
                        num_inference_steps=int(steps),
                        cfg_scale=float(cfg_scale),
                        sigma_shift=float(sigma_shift),
                        seed=sd,
                        negative_prompt=str(neg or ""),
                        progress_bar_cmd=_progress(steps),
                    )
                w = _to_stereo(wave.detach().to(device="cpu", dtype=torch.float32).numpy())
                if s.get("normalize", True):
                    w = _normalize(w)
                if s["type"] == "continuous":
                    w = _make_continuous(w, sr, length, float(s.get("fade", 0.3)))
                else:
                    w = _make_oneshot(w, sr, length)
                clips[s["name"]] = w
                _free_cuda()
        finally:
            _free_cuda()

        folder = _next_folder(filename_prefix)
        total = int(math.ceil(duration * sr))
        mix = np.zeros((2, total), dtype=np.float32)
        tracks = []
        total_frames = int(round(duration * fps))

        for s in sounds:
            base = clips[s["use"]] if s.get("use") else clips[s["name"]]
            clip = base * (10 ** (float(s.get("volume_db", 0.0)) / 20.0))
            wav = os.path.join(folder, f"{s['name']}.wav")
            _write_wav(wav, clip, sr)
            clip_frames = max(1, int(math.ceil(clip.shape[1] / sr * fps)))
            if s["type"] == "continuous":
                starts_sec = [float(s.get("start", 0.0))]
            else:
                starts_sec = sorted(float(t) for t in s["times"])
            for t in starts_sec:
                _place(mix, clip, int(round(t * sr)))
            tracks.append((s["name"], wav, clip_frames, [int(round(t * fps)) for t in starts_sec]))
            when = ", ".join(f"{t:.2f}" for t in starts_sec)
            lines.append(f"{s['name']}: {s['type']}, {clip.shape[1] / sr:.2f}s, at {when}")

        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        limit = 10 ** (PEAK_DB / 20.0)
        if peak > limit:
            mix *= limit / peak
        _write_wav(os.path.join(folder, "mix.wav"), mix, sr)

        xml = _timeline_xml(os.path.basename(folder), fps, total_frames, width, height, sr, vpath, tracks)
        with open(os.path.join(folder, "timeline.xml"), "w", encoding="utf-8") as f:
            f.write(xml)

        report = f"Saved to: {folder}\n" + "\n".join(lines)
        report += "\n\nCue sheet used:\n" + json.dumps(cue, ensure_ascii=False, indent=1)
        with open(os.path.join(folder, "cue_sheet.json"), "w", encoding="utf-8") as f:
            json.dump(cue, f, ensure_ascii=False, indent=2)
        if not vpath:
            report += "\n(No video file connected: timeline.xml has audio tracks only.)"
        print("[MOSS-SFX] " + report.replace("\n", "\n[MOSS-SFX] "), flush=True)

        audio = {"waveform": torch.from_numpy(mix).unsqueeze(0), "sample_rate": sr}
        return {"ui": {"text": [report]}, "result": (audio, report)}


# ---------------------------------------------------------------- Qwen3-VL helper

QWEN_INSTRUCTIONS = """You are a foley sound designer for film.
The images are frames from ONE video, in time order. The time of each frame, in seconds, is written in the yellow box in its top-left corner.
The video is {duration:.2f} seconds long.

Find every sound effect this video needs. No music and no speech.
Use two kinds of sound:
- "continuous": a sound that keeps going, like rain, wind, room tone, crowd murmur, traffic, an engine. Give "start" and "end" in seconds.
- "oneshot": a short sound that happens at separate moments, like footsteps, knocks, a door closing, an impact. Give "times": every moment it happens, in seconds, read from the time written on the frame where it happens.
For footsteps, give one time for every step of the person: the frame where the foot touches the floor.
For every sound write an English "prompt" that describes the source, the material and the action. A oneshot prompt describes ONE single hit, for example "A single high heel footstep on a hard concrete floor, sharp click". Never describe the main action as silent, faint or soft.
"volume_db": 0 for the main action, -6 for secondary sounds, -12 to -18 for background ambience.
{notes}
Answer ONLY with JSON in exactly this form, nothing else:
{{"sounds": [
 {{"name": "room_tone", "type": "continuous", "prompt": "...", "start": 0.0, "end": {duration:.2f}, "volume_db": -12}},
 {{"name": "footsteps", "type": "oneshot", "prompt": "...", "length": 0.6, "times": [1.25, 1.75], "volume_db": 0}}
]}}"""


def _stamp(img_uint8, text):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(im)
    size = max(14, im.height // 14)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    pad = max(3, size // 5)
    draw.rectangle([0, 0, box[2] + pad * 2, box[3] + pad * 2], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 230, 0), font=font)
    return np.asarray(im)


class MossSfxCuePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "All video frames (Get Video Components > images)."}),
                "fps": ("FLOAT", {"forceInput": True, "tooltip": "Video fps (Get Video Components > fps)."}),
                "frames_per_second": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 16.0, "step": 0.5,
                                                "tooltip": "How many frames per second Qwen3-VL sees. "
                                                           "More = better timing, but more VRAM and time."}),
                "max_frames": ("INT", {"default": 32, "min": 2, "max": 128,
                                       "tooltip": "Upper limit on frames sent to Qwen3-VL."}),
                "max_size": ("INT", {"default": 448, "min": 128, "max": 1024, "step": 32,
                                     "tooltip": "Longest side of each frame (pixels) sent to Qwen3-VL."}),
                "notes": ("STRING", {"multiline": True, "default": "",
                                     "tooltip": "Optional extra request in English, e.g. 'Add a quiet audience murmur.'"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "prompt")
    FUNCTION = "build"
    CATEGORY = "audio/MOSS SFX"
    DESCRIPTION = ("Picks frames from the video, writes the time on each one, and builds the instruction that asks "
                   "Qwen3-VL (Generate Text) to write a cue sheet.")

    def build(self, images, fps, frames_per_second, max_frames, max_size, notes=""):
        import torch
        from PIL import Image

        total = int(images.shape[0])
        fps = float(fps) if fps else 24.0
        duration = total / fps
        count = int(math.floor(duration * float(frames_per_second))) + 1
        count = max(2, min(int(max_frames), count, total))
        idx = sorted({int(round(k * (total - 1) / (count - 1))) for k in range(count)})
        idx = [i for i in idx if i < total]

        out = []
        h, w = int(images.shape[1]), int(images.shape[2])
        scale = min(1.0, float(max_size) / max(h, w))
        nh, nw = max(32, int(h * scale) // 2 * 2), max(32, int(w * scale) // 2 * 2)
        for i in idx:
            arr = (images[i].detach().cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
            im = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)
            out.append(_stamp(np.asarray(im), f"{i / fps:.2f}s"))
        batch = torch.from_numpy(np.stack(out).astype(np.float32) / 255.0)

        extra = str(notes or "").strip()
        prompt = QWEN_INSTRUCTIONS.format(duration=duration, notes=("Extra request: " + extra) if extra else "")
        print(f"[MOSS-SFX] Cue prompt: {len(idx)} frames ({nw}x{nh}) of {total}, video {duration:.2f}s", flush=True)
        return (batch, prompt)
