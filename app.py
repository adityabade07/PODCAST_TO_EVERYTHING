import os, tempfile
import streamlit as st
import pipeline as p

st.set_page_config(page_title="Podcast-to-Everything", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1200px}
.hero{background:linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);border-radius:18px;
 padding:2rem 2.2rem;color:#fff;margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.1rem;font-weight:800;color:#fff}
.hero p{margin:.4rem 0 0;opacity:.92;font-size:1.02rem}
.pipe{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1rem}
.pipe span{background:rgba(255,255,255,.18);border-radius:999px;padding:.25rem .8rem;font-size:.82rem}
.stat{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:.9rem 1rem}
.stat b{font-size:1.35rem;display:block}.stat small{opacity:.65}
.pass{color:#16a34a;font-weight:700}.review{color:#d97706;font-weight:700}.unv{color:#6b7280;font-weight:700}
div[data-testid="stTabs"] button{font-weight:600}
</style>
<div class="hero"><h1>🎙️ Podcast-to-Everything</h1>
<p>Upload one episode. Get the transcript, summary, key points, social posts, translations, spoken audio and a fact-check.</p>
<div class="pipe"><span>Speech-to-text</span><span>Summary</span><span>Key points</span><span>X · LinkedIn · Instagram</span>
<span>Translation</span><span>Audio (MP3)</span><span>Faithfulness check</span></div></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    default_key = os.environ.get("API_KEY", "")
    for sp in (os.path.expanduser("~/.streamlit/secrets.toml"), os.path.join(".streamlit", "secrets.toml")):
        if os.path.exists(sp):                      # only touch st.secrets when a file exists (avoids the red box)
            try: default_key = st.secrets.get("API_KEY", default_key)
            except Exception: pass
    key = st.text_input("API key (Claude, Groq, OpenAI or Gemini)", type="password",
                        value=default_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY", ""),
                        help="Claude: sk-ant-… Groq: gsk_… (free at console.groq.com/keys). OpenAI: sk-… Gemini: AIza… or AQ. The key is used only for this session.")
    prov = st.selectbox("Provider", ["Auto-detect", "Groq", "OpenAI", "Gemini", "Claude"],
                        help="Leave on Auto-detect. If your key isn't recognised, choose the provider manually.")
    forced = {"Groq": "groq", "OpenAI": "openai", "Gemini": "gemini", "Claude": "anthropic"}.get(prov)
    kind = forced
    if key.strip():
        if forced:
            st.caption(f"✅ Using: **{forced}**")
        else:
            try:
                kind = p.detect_provider(key); st.caption(f"✅ Detected: **{kind}**")
            except p.PipelineError as e:
                st.caption(f"⚠️ {e} Or pick the provider above.")
    stt_key = ""
    if kind == "anthropic":
        stt_key = st.text_input("Transcription key (Groq / OpenAI / Gemini)", type="password",
                                help="Claude can't turn audio into text. A free Groq key (gsk_…) works well for this step.")
        st.caption("Claude writes the summary, posts and translations. This second key only transcribes.")
    model = st.text_input("Model (optional)", placeholder=p.LLM.DEFAULTS.get(kind, "auto"))
    langs = st.multiselect("Translate to", [k for k in p.LANGS if k != "en"], default=["hi"],
                           format_func=lambda c: p.LANGS[c])
    tts = st.toggle("Generate translated audio", True)
    check = st.toggle("Run faithfulness check", True)
    st.divider()
    st.caption("Large files are split into 10-minute parts automatically, so long podcasts work.")
    st.caption(f"Build {p.APP_VERSION} · default model: {p.LLM.DEFAULTS.get(kind, 'auto')}")

up = st.file_uploader("Upload podcast audio", type=["mp3", "wav", "m4a", "aac", "ogg", "flac", "webm", "mp4"])
if up:
    st.audio(up)
go = st.button("🚀 Run pipeline", type="primary", disabled=not (up and key.strip()))
if not key.strip():
    st.info("Add your API key in the sidebar to begin.")

if go:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1] or ".mp3") as f:
        f.write(up.getbuffer()); path = f.name
    bar, status = st.progress(0.0), st.empty()
    def prog(frac, msg):
        if frac is not None: bar.progress(min(frac, 1.0))
        status.info(f"⏳ {msg}…")
    try:
        st.session_state["res"] = p.run_pipeline(path, key, tuple(langs), model.strip() or None, check, tts, prog, stt_key=stt_key.strip() or None, kind=forced)
        status.success("✅ Finished")
    except p.PipelineError as e:
        st.session_state.pop("res", None); bar.empty(); status.error(f"❌ {e}")
    except Exception as e:
        st.session_state.pop("res", None); bar.empty()
        status.error(f"❌ Unexpected error: {e.__class__.__name__}: {str(e)[:300]}")
    finally:
        try: os.remove(path)
        except OSError: pass

res = st.session_state.get("res")
if res:
    for w in res["warnings"]:
        st.warning(w)
    m = int(res["duration"] // 60)
    cols = st.columns(5)
    for c, (v, l) in zip(cols, [(f"{m} min" if m else "—", "Duration"), (f"{res['words']:,}", "Words"),
                                (p.LANGS.get(res["language"], res["language"]), "Detected language"),
                                (f"{res['seconds']}s", "Processing time"), (res["provider"].split(":")[0], "Provider")]):
        c.markdown(f'<div class="stat"><b>{v}</b><small>{l}</small></div>', unsafe_allow_html=True)
    st.write("")
    st.download_button("⬇️ Download everything (.zip)", res["zip"], "podcast_outputs.zip", "application/zip", type="primary")

    tabs = st.tabs(["📝 Transcript", "📌 Summary", "🔑 Key points", "📣 Social posts", "🌍 Translations", "🔊 Audio", "🛡️ Quality"])
    with tabs[0]:
        st.text_area("Transcript", res["transcript"], height=420, label_visibility="collapsed")
        st.download_button("Download transcript", res["transcript"], "transcript.txt")
    with tabs[1]:
        st.write(res["summary"])
    with tabs[2]:
        st.markdown(res["key_points"])
    with tabs[3]:
        s = res["social"]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("X (Twitter)"); st.code(s["x"], language=None, wrap_lines=True)
            st.caption(f"{len(s['x'])}/280 characters")
            st.subheader("Instagram"); st.code(s["instagram"], language=None, wrap_lines=True)
        with c2:
            st.subheader("LinkedIn"); st.code(s["linkedin"], language=None, wrap_lines=True)
    with tabs[4]:
        if not res["translations"]:
            st.info("No translations were produced.")
        for lg, t in res["translations"].items():
            with st.expander(p.LANGS.get(lg, lg), expanded=True):
                st.write(t["summary"]); st.markdown(t["key_points"])
    with tabs[5]:
        if not res["audio"]:
            st.info("No audio was produced.")
        for lg, b in res["audio"].items():
            st.subheader(p.LANGS.get(lg, lg)); st.audio(b, format="audio/mp3")
            st.download_button(f"Download {p.LANGS.get(lg, lg)} MP3", b, f"podcast_summary_{lg}.mp3", "audio/mpeg", key=f"dl{lg}")
    with tabs[6]:
        if not res["quality"]:
            st.info("Faithfulness check was turned off.")
        for name, q in res["quality"].items():
            css = {"PASS": "pass", "REVIEW": "review"}.get(q["status"], "unv")
            sup = "n/a" if q["support"] is None else f"{int(q['support'] * 100)}%"
            st.markdown(f"**{name}** — <span class='{css}'>{q['status']}</span> · claims supported: {sup}", unsafe_allow_html=True)
            if q["numbers_not_in_transcript"]:
                st.caption("Numbers not found in transcript: " + ", ".join(q["numbers_not_in_transcript"]))
            for c in q["flagged"]:
                st.caption("⚠️ " + c)
        st.caption("REVIEW means a human should read the flagged lines. This check reduces, but cannot eliminate, errors.")
