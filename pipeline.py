"""Podcast-to-Everything engine.

Audio -> ffmpeg (normalise + split) -> speech-to-text -> LLM (summary, key points,
X / LinkedIn / Instagram) -> translation -> TTS (MP3) -> faithfulness check.
Providers: OpenAI (sk-...) or Google Gemini (AIza... / AQ....). Auto-detected from the key.
"""
import glob, io, json, os, re, shutil, subprocess, tempfile, time, zipfile

APP_VERSION = "v9"

LANGS = {"hi": "Hindi", "mr": "Marathi", "gu": "Gujarati", "ta": "Tamil", "te": "Telugu",
         "bn": "Bengali", "kn": "Kannada", "ur": "Urdu", "fr": "French", "es": "Spanish",
         "de": "German", "en": "English"}


class PipelineError(Exception):
    """Error with a message that is safe and useful to show to the end user."""


# ------------------------------------------------------------------ audio
def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise PipelineError("ffmpeg not found. Run: pip install imageio-ffmpeg (or install ffmpeg).")


def prepare_audio(src, workdir, seg_seconds=600):
    """Any audio/video -> mono 16 kHz 48 kbps MP3 segments (~3.6 MB each): fits every API limit."""
    pattern = os.path.join(workdir, "seg_%03d.mp3")
    cmd = [ffmpeg_bin(), "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
           "-f", "segment", "-segment_time", str(seg_seconds), "-reset_timestamps", "1", pattern]
    r = subprocess.run(cmd, capture_output=True, text=True)
    segs = sorted(glob.glob(os.path.join(workdir, "seg_*.mp3")))
    if r.returncode != 0 or not segs:
        raise PipelineError("This file could not be decoded as audio. Try re-exporting it as MP3/WAV.")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    dur = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    return segs, dur


# --------------------------------------------------------------- providers
def clean_key(key):
    """Remove quotes, spaces, 'Bearer ' and 'NAME=' prefixes that often sneak in when pasting a key."""
    k = re.sub(r"\s+", "", (key or "").replace("Bearer ", "").replace("bearer ", ""))
    if "=" in k:
        k = k.split("=")[-1]
    return k.strip("\"'`<>")


def detect_provider(key):
    k = clean_key(key)
    if k.startswith("sk-ant-"):
        return "anthropic"
    if k.startswith("gsk_"):
        return "groq"
    if k.startswith("sk-"):
        return "openai"
    if k.startswith(("AIza", "AQ.")):
        return "gemini"
    raise PipelineError("Unrecognised API key. Claude keys start with 'sk-ant-', Groq with 'gsk_', OpenAI with 'sk-', Gemini with 'AIza' or 'AQ.'.")


class LLM:
    DEFAULTS = {"openai": "gpt-4o-mini", "gemini": "gemini-3.8-flash", "groq": "openai/gpt-oss-120b", "anthropic": "claude-haiku-5-5"}

    def __init__(self, key, kind=None, model=None):
        self.key = clean_key(key)
        self.kind = kind or detect_provider(self.key)
        self.model = model or self.DEFAULTS[self.kind]
        self._switched = False
        self._think = True
        self.notify = None
        self.stt = "whisper-large-v3-turbo" if self.kind == "groq" else "whisper-1"
        self._reason = True
        self.cond_limit = 8000 if self.kind == "groq" else 30000   # Groq free tier has small per-minute token limits
        self.src_cap = 6000 if self.kind == "groq" else 60000
        if self.kind == "anthropic":
            import anthropic
            self.c = anthropic.Anthropic(api_key=self.key, timeout=120)
        elif self.kind in ("openai", "groq"):
            from openai import OpenAI
            extra = {"base_url": "https://api.groq.com/openai/v1"} if self.kind == "groq" else {}
            self.c = OpenAI(api_key=self.key, timeout=180, **extra)
        else:
            from google import genai
            from google.genai import types
            try:
                self.c = genai.Client(api_key=self.key, http_options=types.HttpOptions(timeout=90000))
            except Exception:
                self.c = genai.Client(api_key=self.key)

    def _gemini_models(self):
        """Available Gemini Flash models, best first: newest stable, then previews, then lite."""
        def ver(n):
            m = re.search(r"(\d+(?:\.\d+)?)", n)
            return float(m.group(1)) if m else 0.0
        found = []
        try:
            for m in self.c.models.list():
                n = (m.name or "").replace("models/", "")
                acts = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None)
                if "flash" in n and (acts is None or "generateContent" in acts):
                    found.append(n)
        except Exception:
            return []
        bad = ("image", "tts", "live", "audio", "thinking", "exp", "robotics")
        ok = [n for n in found if not any(b in n for b in bad)]
        full = sorted([n for n in ok if "lite" not in n and "preview" not in n], key=ver, reverse=True)
        prev = sorted([n for n in ok if "lite" not in n and "preview" in n], key=ver, reverse=True)
        lite = sorted([n for n in ok if "lite" in n], key=ver, reverse=True)
        return full + prev + lite

    def _groq_models(self):
        """Chat models available to this Groq key, best first (no reasoning/guard/audio models)."""
        try:
            ids = [m.id for m in self.c.models.list()]
        except Exception:
            return []
        bad = ("whisper", "tts", "playai", "orpheus", "guard", "compound", "deepseek", "distil", "embed")
        ok = [i for i in ids if not any(b in i.lower() for b in bad)]
        pref = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        rest = sorted(m for m in ok if m not in pref and ("llama" in m.lower() or "qwen" in m.lower()))
        return [m for m in pref if m in ok] + rest

    def _candidates(self):
        return {"gemini": self._gemini_models, "groq": self._groq_models}.get(self.kind, lambda: [])()

    def _pick_model(self):
        c = self._candidates()
        if c:
            self.model = c[0]
        return bool(c)

    def _next_model(self):
        """Another model (own quota) to try when the current one is busy or rate-limited."""
        self._used = getattr(self, "_used", set()) | {self.model}
        for n in self._candidates():
            if n not in self._used:
                return n
        return None

    def _retry(self, fn, tries=7, switch=True):
        for i in range(tries):
            try:
                return fn()
            except Exception as e:
                msg = str(e).lower()
                multi = self.kind in ("gemini", "groq")
                gone = any(t in msg for t in ("404", "not_found", "no longer available", "decommissioned", "model_not_found", "does not exist"))
                if multi and switch and not self._switched and gone:
                    self._switched = True              # model retired: auto-pick a current one
                    if self._pick_model():
                        continue
                    raise PipelineError("The model is unavailable. Type a current model name in the sidebar.") from e
                if self.kind == "gemini" and self._think and "thinking" in msg:
                    self._think = False                 # model rejects thinking_budget=0: retry without it
                    continue
                if self.kind == "groq" and self._reason and "reasoning" in msg and any(t in msg for t in ("400", "invalid", "unsupported", "unknown")):
                    self._reason = False                # model rejects reasoning_effort: retry without it
                    continue
                if any(t in msg for t in ("401", "403", "api key not valid", "api_key_invalid", "permission_denied", "unauthenticated", "authentication_error", "invalid x-api-key", "invalid api key")):
                    raise PipelineError("The API key was rejected or lacks access. Check the key and that the API is enabled.") from e
                if i == tries - 1:
                    raise PipelineError(f"{self.kind} API still failing after {tries} attempts: {str(e)[:300]}. "
                                        "Wait a minute and run again, or pick another model in the sidebar.") from e
                if "exceeded your current quota" in msg and i >= 2:
                    raise PipelineError("Your Gemini quota is used up (free-tier limits reset daily). Options: wait and retry, "
                                        "enable billing in Google AI Studio, or use a free Groq key (console.groq.com/keys) instead.") from e
                busy = any(t in msg for t in ("rate limit", "rate_limit", "overloaded", "529", "503", "unavailable", "high demand", "429", "resource_exhausted", "500", "502", "504"))
                hint = re.search(r"try again in (?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", msg)   # Groq says how long to wait
                wait = min(60, 3 * 2 ** i)
                long_wait = False
                if hint:
                    secs = int(hint.group(1) or 0) * 60 + float(hint.group(2))
                    long_wait, wait = secs > 90, min(60, secs + 1)
                if multi and switch and busy and (i % 2 == 1 or long_wait):
                    nxt = self._next_model()            # still limited: switch to a model with its own quota
                    if nxt:
                        old, self.model = self.model, nxt
                        if self.notify:
                            self.notify(f"{old} is limited, switching to {nxt}")
                        continue
                    if long_wait:
                        raise PipelineError("The free daily limit for this key looks used up. Try again later (limits reset daily).") from e
                if self.notify:
                    self.notify(f"API busy/error, retrying in {int(wait)}s ({i + 1}/{tries}): {str(e)[:90]}")
                time.sleep(wait)

    def chat(self, system, user, max_tokens=800, temperature=0.2):
        def go():
            if self.kind == "anthropic":
                r = self.c.messages.create(model=self.model, max_tokens=max_tokens, temperature=temperature,
                                           system=system, messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
            if self.kind in ("openai", "groq"):
                kw, mx = {}, max_tokens
                name = self.model.lower()
                if self.kind == "groq" and ("gpt-oss" in name or "qwen" in name):
                    mx = max_tokens * 2 + 700            # reasoning models spend tokens thinking first
                    if "gpt-oss" in name and self._reason:
                        kw["extra_body"] = {"reasoning_effort": "low"}
                r = self.c.chat.completions.create(
                    model=self.model, temperature=temperature, max_tokens=mx, **kw,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                out = (r.choices[0].message.content or "").strip()
                return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
            from google.genai import types
            cfg = dict(system_instruction=system, temperature=temperature, max_output_tokens=max_tokens * 3 + 512)
            if self._think:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            r = self.c.models.generate_content(model=self.model, contents=user,
                                               config=types.GenerateContentConfig(**cfg))
            return (r.text or "").strip()
        return self._retry(go)

    def transcribe(self, path):
        try:
            return self._transcribe(path)
        except PipelineError:
            if self.kind == "groq" and self.stt != "whisper-large-v3":
                self.stt = "whisper-large-v3"          # second Whisper model has its own limits
                if self.notify:
                    self.notify("Trying whisper-large-v3")
                return self._transcribe(path)
            raise

    def _transcribe(self, path):
        if self.kind == "anthropic":
            raise PipelineError("Claude cannot transcribe audio. Add a Groq (free), OpenAI or Gemini key as the transcription key.")
        def go():
            if self.kind in ("openai", "groq"):
                with open(path, "rb") as f:
                    return self.c.audio.transcriptions.create(model=self.stt, file=f).text.strip()
            from google.genai import types
            with open(path, "rb") as f:
                part = types.Part.from_bytes(data=f.read(), mime_type="audio/mpeg")
            r = self.c.models.generate_content(model=self.model, contents=[part,
                "Transcribe this audio verbatim in its original language. Output only the transcript, "
                "no timestamps, no speaker labels, no commentary. If there is no speech, output nothing."])
            return (r.text or "").strip()
        return self._retry(go, switch=(self.kind == "gemini"))

    def ping(self):
        if not self.chat("Reply with the single word OK.", "ping", max_tokens=50, temperature=0):
            raise PipelineError("The model returned an empty reply. Check the model name.")


# ------------------------------------------------------------ text helpers
def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", t) if s.strip()]


def clean_transcript(raw, per_para=6):
    t = re.sub(r"\s+", " ", raw).strip()
    t = re.sub(r"\b(\w+)(\s+\1\b){2,}", r"\1", t, flags=re.I)  # collapse runaway repeats only
    s = split_sentences(t)
    return "\n\n".join(" ".join(s[i:i + per_para]) for i in range(0, len(s), per_para))


def chunk_text(text, n):
    out, cur = [], ""
    for s in split_sentences(text.replace("\n", " ")):
        while len(s) > n:                      # sentence longer than a chunk (e.g. no punctuation)
            if cur: out.append(cur.strip()); cur = ""
            out.append(s[:n]); s = s[n:]
        if len(cur) + len(s) + 1 > n and cur:
            out.append(cur.strip()); cur = ""
        cur += " " + s
    if cur.strip():
        out.append(cur.strip())
    return out or [text]


GROUND = ("You work ONLY from the provided podcast transcript. Never add facts, numbers, names or claims "
          "that are not in it. Present opinions as the speaker's opinions. Keep numbers exactly as stated. "
          "Write in English even if the transcript is in another language.")


def condense(llm, transcript, progress=lambda *a: None, depth=0):
    lim = getattr(llm, "cond_limit", 30000)
    if len(transcript) <= lim:
        return transcript
    parts = chunk_text(transcript, int(lim * 0.4))
    notes = []
    for i, c in enumerate(parts, 1):
        progress(None, f"Condensing long transcript ({i}/{len(parts)})")
        notes.append(llm.chat(GROUND, "Write faithful bullet notes (max 12 bullets) of this part of a "
                                      "podcast transcript.\n\n" + c, max_tokens=600, temperature=0))
    out = "\n".join(notes)
    if len(out) > lim and len(out) < len(transcript) * 0.8 and depth < 3:
        return condense(llm, out, progress, depth + 1)      # still too long: condense the notes again
    return out


def _bullets(t):
    lines = [re.sub(r"^[\s\-\*•·\d.)]+", "", l).strip() for l in t.splitlines()]
    return "\n".join(f"- {l}" for l in lines if l)


def _fit_x(llm, post, limit=280):
    post = post.strip().strip('"')
    if len(post) > limit:
        post = llm.chat(GROUND, f"Rewrite under {limit} characters, same meaning, max 2 hashtags. "
                                f"Output only the post:\n\n{post}", max_tokens=120, temperature=0).strip('"')
    if len(post) > limit:
        post = post[:limit - 1].rsplit(" ", 1)[0] + "…"
    return post


def generate(llm, source):
    T = "\n\nTranscript:\n" + source
    social = {
        "x": _fit_x(llm, llm.chat(GROUND, "Write ONE engaging X (Twitter) post under 280 characters with at "
                                          "most 2 hashtags. Output only the post." + T, 150, 0.5)),
        "linkedin": llm.chat(GROUND, "Write ONE LinkedIn post (110-170 words): strong first line, 3 short "
                                     "paragraphs, then 3-5 hashtags. Professional tone. Output only the post." + T, 400, 0.5),
        "instagram": llm.chat(GROUND, "Write ONE Instagram caption: friendly, 2-4 short lines with a few emojis, "
                                      "then 6-10 hashtags. Output only the caption." + T, 300, 0.6),
    }
    return {
        "summary": llm.chat(GROUND, "Write a concise summary of this podcast in 120-180 words, plain prose, "
                                    "one or two paragraphs." + T, 450),
        "key_points": _bullets(llm.chat(GROUND, "List the 5-8 most important key points as short bullets, each "
                                                "starting with '- '. Output only the bullets." + T, 500)),
        "social": social}


# --------------------------------------------------- translation and speech
def translate(llm, text, lang):
    name = LANGS.get(lang, lang)
    return "\n".join(_tr_chunk(llm, c, name) for c in _line_chunks(text, 3000))


def _line_chunks(text, n):
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > n and cur:
            out.append(cur); cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur.rstrip("\n"))
    return out


def _tr_chunk(llm, c, name):
    return llm.chat("You are a professional translator. Output only the translation.",
                    f"Translate into {name}. Keep meaning, numbers, line breaks and '- ' bullet markers "
                    f"unchanged:\n\n{c}", 2000, 0)


def text_to_speech(text, lang):
    from gtts import gTTS
    clean = re.sub(r"\s+", " ", re.sub(r"[#*_`>\-•]+", " ", text)).strip()
    buf = io.BytesIO()
    gTTS(text=clean, lang=lang, slow=False).write_to_fp(buf)
    return buf.getvalue()


# ----------------------------------------------------- faithfulness check
def _claims(text):
    out = []
    for line in text.split("\n"):
        line = re.sub(r"#\w+", "", re.sub(r"^[\s\-\*•\d.)]+", "", line)).strip()
        out += [s for s in split_sentences(line) if len(s.split()) >= 4]
    return out[:25]


def _numbers(t):
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)?", t)}


def check_output(llm, text, source):
    claims = _claims(text)
    missing = sorted(_numbers(text) - _numbers(source))
    score, bad = None, []
    if claims:
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
        try:
            ans = llm.chat("You are a strict fact checker. Reply with JSON only.",
                           "Transcript:\n" + source[:getattr(llm, "src_cap", 60000)] + "\n\nClaims:\n" + numbered +
                           '\n\nReturn {"unsupported": [claim numbers the transcript does NOT support; '
                           'paraphrase is fine]}', 200, 0)
            idx = json.loads(re.search(r"\{.*\}", ans, re.S).group(0)).get("unsupported", [])
            bad = [claims[int(i) - 1] for i in idx if str(i).isdigit() and 0 < int(i) <= len(claims)]
            score = round(1 - len(bad) / len(claims), 2)
        except PipelineError:
            raise
        except Exception:
            score = None
    status = "UNVERIFIED" if score is None and claims else ("PASS" if (score or 1) >= 0.8 and not missing else "REVIEW")
    return {"support": score, "numbers_not_in_transcript": missing, "flagged": bad, "status": status}


# ---------------------------------------------------------------- full run
def run_pipeline(audio_path, key, langs=("hi",), model=None, check=True, tts=True,
                 progress=lambda frac, msg: None, stt_key=None, kind=None):
    t0 = time.time()
    llm = LLM(key, kind=kind, model=model)
    llm.notify = lambda m: progress(None, m)
    stt = LLM(stt_key) if stt_key else llm            # separate key for speech-to-text (needed with Claude)
    stt.notify = llm.notify
    if stt.kind == "anthropic":
        raise PipelineError("Claude cannot transcribe audio. Use a Groq (free), OpenAI or Gemini key for transcription.")
    progress(0.02, f"Validating API key ({llm.kind}: {llm.model})")
    llm.ping()
    warnings = []
    with tempfile.TemporaryDirectory() as wd:
        progress(0.05, "Preparing audio")
        segs, dur = prepare_audio(audio_path, wd)
        texts = []
        for i, s in enumerate(segs, 1):
            progress(0.05 + 0.30 * i / len(segs), f"Transcribing part {i}/{len(segs)}")
            texts.append(stt.transcribe(s))
    raw = " ".join(t for t in texts if t).strip()
    if len(raw.split()) < 5:
        raise PipelineError("No usable speech was detected in this audio.")
    transcript = clean_transcript(raw)
    lang = (re.findall(r"[a-z]{2}", llm.chat("Return only the ISO 639-1 code of the main language of the text.",
                                            raw[:600], 5, 0).lower()) or ["en"])[0]

    progress(0.38, "Understanding the podcast")
    source = condense(llm, transcript, progress)
    progress(0.45, "Writing summary, key points and social posts")
    res = generate(llm, source)
    res.update(transcript=transcript, language=lang, duration=dur, words=len(transcript.split()),
               provider=f"{llm.kind}:{llm.model}", translations={}, audio={}, quality={}, warnings=warnings)

    for n, lg in enumerate(langs, 1):
        progress(0.6 + 0.15 * n / len(langs), f"Translating to {LANGS.get(lg, lg)}")
        try:
            res["translations"][lg] = {"summary": translate(llm, res["summary"], lg),
                                       "key_points": translate(llm, res["key_points"], lg)}
        except PipelineError as e:
            warnings.append(f"Translation to {LANGS.get(lg, lg)} failed: {e}")
    if tts:
        for n, (lg, t) in enumerate(res["translations"].items(), 1):
            progress(0.75 + 0.10 * n / len(res["translations"]), f"Generating {LANGS.get(lg, lg)} audio")
            try:
                res["audio"][lg] = text_to_speech(t["summary"] + "\n" + t["key_points"], lg)
            except Exception as e:
                warnings.append(f"Audio for {LANGS.get(lg, lg)} skipped ({e.__class__.__name__}); needs internet access to Google TTS.")
    if check:
        outs = {"Summary": res["summary"], "Key points": res["key_points"], "X post": res["social"]["x"],
                "LinkedIn post": res["social"]["linkedin"], "Instagram caption": res["social"]["instagram"]}
        for i, (k, v) in enumerate(outs.items(), 1):
            progress(0.85 + 0.12 * i / len(outs), f"Fact-checking: {k}")
            res["quality"][k] = check_output(llm, v, source)

    res["zip"] = build_zip(res)
    res["seconds"] = round(time.time() - t0, 1)
    progress(1.0, "Done")
    return res


def build_zip(res):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("transcript.txt", res["transcript"])
        z.writestr("summary.md", res["summary"])
        z.writestr("key_points.md", res["key_points"])
        s = res["social"]
        z.writestr("social_posts.md", f"## X\n{s['x']}\n\n## LinkedIn\n{s['linkedin']}\n\n## Instagram\n{s['instagram']}\n")
        for lg, t in res["translations"].items():
            z.writestr(f"translations/summary_{lg}.md", t["summary"])
            z.writestr(f"translations/key_points_{lg}.md", t["key_points"])
        for lg, b in res["audio"].items():
            z.writestr(f"audio/podcast_summary_{lg}.mp3", b)
        if res["quality"]:
            z.writestr("faithfulness_report.json", json.dumps(res["quality"], indent=2, ensure_ascii=False))
    return buf.getvalue()
