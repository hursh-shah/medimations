"use client";

import { useEffect, useMemo, useRef, useState } from "react";

class ApiError extends Error {
  constructor(status, body) {
    super(body || `Request failed (${status})`);
    this.status = status;
    this.body = body || "";
  }
}

function backendBase() {
  const raw = process.env.NEXT_PUBLIC_BACKEND_URL || "";
  const cleaned = raw.trim().replace(/\/+$/, "");
  if (!cleaned) return "";
  if (/^https?:\/\//i.test(cleaned)) return cleaned;
  if (cleaned.startsWith("//")) return `https:${cleaned}`;
  if (/^(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$/i.test(cleaned)) return `http://${cleaned}`;
  return `https://${cleaned}`;
}

async function apiFetchJson(path, init) {
  const base = backendBase();
  const url = base ? `${base}${path}` : path;
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(res.status, text);
  }
  return res.json();
}

async function apiFetchText(urlOrPath) {
  const base = backendBase();
  const url = base && urlOrPath.startsWith("/") ? `${base}${urlOrPath}` : urlOrPath;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(res.status, text);
  }
  return res.text();
}

function clampIso(iso) {
  const s = String(iso || "").trim();
  if (!s) return "";
  return s.replace("T", " ").replace("Z", "");
}

function parseSrtTimestamp(ts) {
  const m = String(ts || "")
    .trim()
    .match(/^(\d+):(\d+):(\d+)[,.](\d+)$/);
  if (!m) return 0;
  const [, hh, mm, ss, ms] = m;
  return Number(hh) * 3600 + Number(mm) * 60 + Number(ss) + Number(ms) / 1000;
}

function parseSrt(srtText) {
  const text = String(srtText || "").replace(/\r/g, "");
  const blocks = text.split(/\n\n+/);
  const segments = [];
  for (const block of blocks) {
    const lines = block.split("\n").map((l) => l.trimEnd());
    const timeLineIndex = lines.findIndex((l) => l.includes("-->"));
    if (timeLineIndex === -1) continue;
    const timeLine = lines[timeLineIndex];
    const [startRaw, endRaw] = timeLine.split("-->").map((s) => s.trim());
    const start = parseSrtTimestamp(startRaw);
    const end = parseSrtTimestamp(endRaw);
    const captionLines = lines.slice(timeLineIndex + 1).filter((l) => l.trim());
    const caption = captionLines.join(" ").trim();
    if (!caption) continue;
    segments.push({ start, end, text: caption });
  }
  return segments;
}

function DownloadLink({ jobId, kind, children }) {
  const href = useMemo(() => {
    const base = backendBase();
    let path = "";
    if (kind === "video") path = `/api/download/videos/${jobId}.mp4`;
    if (kind === "narrated") path = `/api/download/videos/${jobId}/narrated.mp4`;
    if (kind === "captions_srt") path = `/api/download/captions/${jobId}.srt`;
    if (kind === "captions_json") path = `/api/download/captions/${jobId}.json`;
    if (kind === "audio") path = `/api/download/audio/${jobId}.mp3`;
    if (!path) return "";
    return base ? `${base}${path}` : path;
  }, [jobId, kind]);

  if (!href) return null;
  return (
    <a className="actionBtn" href={href} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
      {children}
    </a>
  );
}

function FeedItem({
  item,
  isActive,
  muted,
  onToggleMute,
  postprocess,
  onPostprocess,
  onRemoveCaptions,
  onExtend,
  extendStatus,
  setRef
}) {
  const videoRef = useRef(null);
  const captionsRef = useRef(null);
  const lastCaptionRef = useRef("");
  const [caption, setCaption] = useState("");

  const src = item?.narrated_video_url || item?.video_url || "";
  const hasCaptions = Boolean(item?.captions_srt_url);
  const hasVoiceover = Boolean(item?.narrated_video_url);

  useEffect(() => {
    if (hasCaptions) return;
    captionsRef.current = null;
    lastCaptionRef.current = "";
    setCaption("");
  }, [hasCaptions, item?.captions_srt_url]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (!isActive) {
      v.pause();
      return;
    }
    v.play().catch(() => {});
  }, [isActive, src]);

  useEffect(() => {
    if (!isActive) return;
    if (!hasCaptions) return;
    if (captionsRef.current) return;
    let cancelled = false;
    apiFetchText(item.captions_srt_url)
      .then((txt) => {
        if (cancelled) return;
        captionsRef.current = parseSrt(txt);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [isActive, hasCaptions, item?.captions_srt_url]);

  function togglePlay() {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => {});
    else v.pause();
  }

  function onTimeUpdate() {
    if (!isActive) return;
    const v = videoRef.current;
    const segs = captionsRef.current || [];
    if (!v || segs.length === 0) return;
    const t = v.currentTime || 0;
    const seg = segs.find((s) => t >= s.start && t <= s.end);
    const next = seg?.text || "";
    if (next === lastCaptionRef.current) return;
    lastCaptionRef.current = next;
    setCaption(next);
  }

  const busy = postprocess?.status === "running";
  const err = postprocess?.status === "error" ? postprocess?.error : "";
  const extending = extendStatus?.status === "running";
  const isExtension = Boolean(item?.extended_from);

  return (
    <div ref={setRef} className="feedItem" data-job-id={item.job_id}>
      <div className="videoFrame" onClick={togglePlay}>
        <video
          className="video"
          ref={videoRef}
          src={src}
          playsInline
          loop
          preload="metadata"
          muted={muted}
          onTimeUpdate={onTimeUpdate}
        />
        <div className="videoOverlayTop" />
        <div className="videoOverlayBottom" />

        <div className="meta">
          <div className="metaTitle">
            {isExtension ? "↳ " : ""}{item.prompt || item.job_id}
          </div>
          <div className="metaSubtitle">
            {clampIso(item.created_at)}
            {item.report_summary ? ` • ${item.report_summary}` : ""}
            {isExtension ? " • extended" : ""}
            {err ? ` • ${String(err).slice(0, 120)}` : ""}
          </div>
        </div>

        {caption ? <div className="caption">{caption}</div> : null}

        <div className="actions" onClick={(e) => e.stopPropagation()}>
          <button
            className="actionBtn"
            disabled={busy}
            onClick={() => (hasCaptions ? onRemoveCaptions(item.job_id) : onPostprocess(item.job_id, "captions"))}
          >
            {busy && (postprocess?.mode === "captions" || postprocess?.mode === "captions_remove")
              ? "…"
              : hasCaptions
                ? "CC ✓"
                : "CC"}
          </button>
          <button
            className="actionBtn"
            disabled={busy || hasVoiceover}
            onClick={() => onPostprocess(item.job_id, "voiceover")}
          >
            {busy && postprocess?.mode === "voiceover" ? "…" : hasVoiceover ? "VO ✓" : "VO"}
          </button>
          <button
            className="actionBtn actionBtnExtend"
            disabled={extending}
            onClick={() => onExtend(item.job_id)}
          >
            {extending ? "…" : "Extend"}
          </button>
          <button className="actionBtn" onClick={onToggleMute}>
            {muted ? "Sound" : "Mute"}
          </button>
          <DownloadLink jobId={item.job_id} kind={hasVoiceover ? "narrated" : "video"}>
            DL
          </DownloadLink>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [creatorOpen, setCreatorOpen] = useState(false);
  const [muted, setMuted] = useState(true);

  const [imageDataUrl, setImageDataUrl] = useState("");
  const [imagePrompt, setImagePrompt] = useState("");
  const [imageModel, setImageModel] = useState("imagen-3.0-generate-001");
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  const [isCheckingImage, setIsCheckingImage] = useState(false);
  const [imageValidation, setImageValidation] = useState(null);

  const [prompt, setPrompt] = useState("");
  const [target, setTarget] = useState("");
  const [rewriteMode, setRewriteMode] = useState("gemini");
  const [veoModel, setVeoModel] = useState("veo-3.1-generate-preview");
  const [useBiomedclip, setUseBiomedclip] = useState(true);
  const [postprocessMode, setPostprocessMode] = useState("voiceover");

  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");

  const [library, setLibrary] = useState([]);
  const [activeJobId, setActiveJobId] = useState(null);
  const [postprocessById, setPostprocessById] = useState({});
  const [extendStatusById, setExtendStatusById] = useState({});

  // Extension modal state
  const [extendModalOpen, setExtendModalOpen] = useState(false);
  const [extendJobId, setExtendJobId] = useState(null);
  const [extendPrompt, setExtendPrompt] = useState("");
  const [extendUseGemini, setExtendUseGemini] = useState(true);
  const [extendPostprocess, setExtendPostprocess] = useState("voiceover");
  const [isExtending, setIsExtending] = useState(false);
  const [extendError, setExtendError] = useState("");
  const [extendNewJobId, setExtendNewJobId] = useState(null);

  const feedRef = useRef(null);
  const itemEls = useRef(new Map());
  const pollRef = useRef(null);
  const postprocessPollersRef = useRef({});
  const extendPollersRef = useRef({});

  const videoUrl = useMemo(() => {
    if (!jobId) return "";
    const base = backendBase();
    const path = `/api/videos/${jobId}.mp4`;
    return base ? `${base}${path}` : path;
  }, [jobId]);

  async function refreshLibrary() {
    try {
      const items = await apiFetchJson("/api/library");
      setLibrary(items);
      if (!activeJobId && items.length > 0) setActiveJobId(items[0].job_id);
    } catch {
      // Non-fatal if backend isn't up yet.
    }
  }

  async function refreshJob(id) {
    try {
      const data = await apiFetchJson(`/api/jobs/${id}`);
      setJob(data.job);
      if (data.job.status === "done" || data.job.status === "error") {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
        await refreshLibrary();
      }
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
        setError("Job not found on backend (server restarted or job expired). Please generate again.");
        await refreshLibrary();
        return;
      }
      throw e;
    }
  }

  async function onGenerate() {
    setError("");
    setJob(null);
    setJobId(null);
    if (!prompt.trim()) {
      setError("Enter an animation prompt.");
      return;
    }
    if (!imageDataUrl) {
      setError("Select or generate a medical image first.");
      return;
    }

    try {
      const resp = await apiFetchJson("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          backend: "veo",
          input_image: imageDataUrl,
          prompt_rewrite: rewriteMode,
          gemini_model: "gemini-3.0-flash",
          veo_model: veoModel,
          postprocess_mode: postprocessMode,
          use_biomedclip: false,
          biomedclip_target: null
        })
      });
      setJobId(resp.job_id);
      await refreshJob(resp.job_id);
      pollRef.current = setInterval(() => refreshJob(resp.job_id).catch(() => {}), 2000);
    } catch (e) {
      setError(String(e?.message || e));
    }
  }

  async function readFileAsDataUrl(file) {
    const maxBytes = 8 * 1024 * 1024;
    if (file.size > maxBytes) {
      throw new Error("Image is too large (max 8MB).");
    }
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Failed to read image."));
      reader.onload = () => resolve(String(reader.result || ""));
      reader.readAsDataURL(file);
    });
  }

  async function onUploadImage(e) {
    const file = e?.target?.files?.[0];
    if (!file) return;
    setError("");
    try {
      const dataUrl = await readFileAsDataUrl(file);
      setImageDataUrl(dataUrl);
      setImageValidation(null);
    } catch (err) {
      setError(String(err?.message || err));
    }
  }

  async function onGenerateImage() {
    setError("");
    if (!imagePrompt.trim()) {
      setError("Enter an image prompt.");
      return;
    }
    setIsGeneratingImage(true);
    setImageValidation(null);
    try {
      const resp = await apiFetchJson("/api/images/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: imagePrompt,
          model: imageModel,
          aspect_ratio: "1:1",
          prompt_rewrite: "gemini",
          gemini_model: "gemini-3.0-flash",
          use_biomedclip: Boolean(useBiomedclip),
          biomedclip_target: useBiomedclip && target.trim() ? target.trim() : null,
          max_rounds: 2
        })
      });
      setImageDataUrl(resp.image_data_url || "");
      setImageValidation(resp || null);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setIsGeneratingImage(false);
    }
  }

  async function onValidateImage() {
    if (!imageDataUrl) return;
    setError("");
    setIsCheckingImage(true);
    try {
      const resp = await apiFetchJson("/api/images/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_image: imageDataUrl,
          prompt: imagePrompt.trim() ? imagePrompt.trim() : null,
          biomedclip_target: useBiomedclip && target.trim() ? target.trim() : null,
          biomedclip_threshold: 0.85
        })
      });
      setImageValidation(resp || null);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setIsCheckingImage(false);
    }
  }

  async function onPostprocess(jobId, mode) {
    setError("");
    setPostprocessById((prev) => ({
      ...prev,
      [jobId]: { status: "running", mode, error: "" }
    }));

    if (postprocessPollersRef.current[jobId]) {
      clearInterval(postprocessPollersRef.current[jobId]);
      delete postprocessPollersRef.current[jobId];
    }

    try {
      const resp = await apiFetchJson(`/api/jobs/${jobId}/postprocess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, force: false })
      });

      const poll = async () => {
        try {
          const data = await apiFetchJson(resp.status_url);
          const ps = String(data?.job?.postprocess_status || "").trim();
          const pe = String(data?.job?.postprocess_error || "").trim();
          if (ps && ps !== "running") {
            clearInterval(postprocessPollersRef.current[jobId]);
            delete postprocessPollersRef.current[jobId];
            setPostprocessById((prev) => ({
              ...prev,
              [jobId]: { status: ps === "error" ? "error" : "done", mode, error: pe }
            }));
            await refreshLibrary();
          }
        } catch {
          // Keep polling for a bit.
        }
      };
      postprocessPollersRef.current[jobId] = setInterval(() => poll(), 2000);
      await poll();
    } catch (e) {
      setPostprocessById((prev) => ({
        ...prev,
        [jobId]: { status: "error", mode, error: String(e?.message || e) }
      }));
      setError(String(e?.message || e));
    }
  }

  async function onRemoveCaptions(jobId) {
    setError("");
    setPostprocessById((prev) => ({
      ...prev,
      [jobId]: { status: "running", mode: "captions_remove", error: "" }
    }));
    try {
      await apiFetchJson(`/api/jobs/${jobId}/captions`, { method: "DELETE" });
      setPostprocessById((prev) => ({
        ...prev,
        [jobId]: { status: "done", mode: "captions_remove", error: "" }
      }));
      await refreshLibrary();
    } catch (e) {
      setPostprocessById((prev) => ({
        ...prev,
        [jobId]: { status: "error", mode: "captions_remove", error: String(e?.message || e) }
      }));
      setError(String(e?.message || e));
    }
  }

  function openExtendModal(jobId) {
    setExtendJobId(jobId);
    setExtendPrompt("");
    setExtendUseGemini(true);
    setExtendPostprocess("voiceover");
    setExtendError("");
    setExtendNewJobId(null);
    setExtendModalOpen(true);
  }

  async function submitExtend() {
    if (!extendJobId) return;
    if (!extendUseGemini && !extendPrompt.trim()) {
      setExtendError("Please enter a prompt for the video extension.");
      return;
    }

    setExtendError("");
    setIsExtending(true);
    setExtendStatusById((prev) => ({
      ...prev,
      [extendJobId]: { status: "running" }
    }));

    try {
      const resp = await apiFetchJson(`/api/jobs/${extendJobId}/extend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: extendPrompt.trim() || null,
          use_gemini: extendUseGemini,
          postprocess_mode: extendPostprocess
        })
      });

      setExtendNewJobId(resp.new_job_id);

      // Poll for completion
      const pollExtend = async () => {
        try {
          const data = await apiFetchJson(resp.status_url);
          const status = String(data?.job?.status || "").trim();
          if (status === "done" || status === "error") {
            if (extendPollersRef.current[extendJobId]) {
              clearInterval(extendPollersRef.current[extendJobId]);
              delete extendPollersRef.current[extendJobId];
            }
            setExtendStatusById((prev) => ({
              ...prev,
              [extendJobId]: { status: status === "error" ? "error" : "done" }
            }));
            setIsExtending(false);
            if (status === "done") {
              await refreshLibrary();
            } else {
              setExtendError(data?.job?.error || "Extension failed.");
            }
          }
        } catch {
          // Keep polling
        }
      };

      extendPollersRef.current[extendJobId] = setInterval(pollExtend, 2000);
      await pollExtend();
    } catch (e) {
      setExtendStatusById((prev) => ({
        ...prev,
        [extendJobId]: { status: "error" }
      }));
      setExtendError(String(e?.message || e));
      setIsExtending(false);
    }
  }

  useEffect(() => {
    refreshLibrary();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      const pollers = postprocessPollersRef.current || {};
      for (const k of Object.keys(pollers)) clearInterval(pollers[k]);
      const extendPollers = extendPollersRef.current || {};
      for (const k of Object.keys(extendPollers)) clearInterval(extendPollers[k]);
    };
  }, []);

  useEffect(() => {
    const root = feedRef.current;
    if (!root) return;

    const obs = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const id = entry.target?.getAttribute?.("data-job-id");
          if (id) setActiveJobId(id);
        }
      },
      { root, threshold: 0.6 }
    );

    for (const el of itemEls.current.values()) obs.observe(el);
    return () => obs.disconnect();
  }, [library]);

  return (
    <div className="app">
      <header className="topBar">
        <div>
          <div className="brandTitle">Medical Diffusion</div>
          <div className="brandSubtitle">Image + text → Veo 3.1 video • BiomedCLIP checks images • Add captions/VO anytime</div>
        </div>
        <div className="topBarRight">
          <div className="pill">{backendBase() || "same origin"}</div>
          <button className="btn btnGhost btnSmall" onClick={() => setMuted((m) => !m)}>
            {muted ? "Sound" : "Mute"}
          </button>
          <button className="btn btnGhost btnSmall" onClick={refreshLibrary}>
            Refresh
          </button>
          <button className="btn btnPrimary btnSmall" onClick={() => setCreatorOpen(true)}>
            Create
          </button>
        </div>
      </header>

      <main className="main">
        {library.length === 0 ? (
          <div className="feedEmpty">
            <div>
              <div style={{ fontWeight: 700, marginBottom: 8 }}>No videos yet.</div>
              <div className="muted">Tap Create to generate your first medical animation.</div>
            </div>
          </div>
        ) : (
          <div ref={feedRef} className="feed">
            {library.map((item) => (
              <FeedItem
                key={item.job_id}
                setRef={(el) => {
                  if (el) itemEls.current.set(item.job_id, el);
                  else itemEls.current.delete(item.job_id);
                }}
                item={item}
                isActive={item.job_id === activeJobId}
                muted={muted}
                onToggleMute={() => setMuted((m) => !m)}
                postprocess={postprocessById[item.job_id]}
                onPostprocess={onPostprocess}
                onRemoveCaptions={onRemoveCaptions}
                onExtend={openExtendModal}
                extendStatus={extendStatusById[item.job_id]}
              />
            ))}
          </div>
        )}
      </main>

      {creatorOpen ? (
        <div className="modalOverlay" onClick={() => setCreatorOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div>
                <div className="modalTitle">Create</div>
                <div className="muted">Upload or generate an image, then animate it.</div>
              </div>
              <button className="btn btnGhost btnSmall" onClick={() => setCreatorOpen(false)}>
                Close
              </button>
            </div>

            <div className="modalBody">
              <div className="grid2">
                <label className="field">
                  <div className="label">Upload medical image</div>
                  <input className="input" type="file" accept="image/*" onChange={onUploadImage} />
                </label>
                <div className="field">
                  <div className="label">Or generate one with AI</div>
                  <div className="row">
                    <input
                      className="input"
                      value={imagePrompt}
                      onChange={(e) => setImagePrompt(e.target.value)}
                      placeholder='e.g. "axial CT slice of the liver, clinical, grayscale, no text"'
                    />
                    <button className="btn btnGhost btnSmall" onClick={onGenerateImage} disabled={isGeneratingImage}>
                      {isGeneratingImage ? "…" : "Generate"}
                    </button>
                  </div>
                  <div className="row">
                    <select className="select" value={imageModel} onChange={(e) => setImageModel(e.target.value)}>
                      <option value="imagen-3.0-generate-001">imagen-3.0-generate-001</option>
                      <option value="imagen-3.0-fast-generate-001">imagen-3.0-fast-generate-001</option>
                    </select>
                    <label className="chip">
                      <input
                        type="checkbox"
                        checked={useBiomedclip}
                        onChange={(e) => setUseBiomedclip(e.target.checked)}
                      />
                      <span className="muted">BiomedCLIP verify (reprompt once)</span>
                    </label>
                    {imageDataUrl ? (
                      <button
                        className="btn btnGhost btnSmall"
                        onClick={() => {
                          setImageDataUrl("");
                          setImageValidation(null);
                        }}
                      >
                        Clear
                      </button>
                    ) : null}
                  </div>
                  <label className="field">
                    <div className="label">BiomedCLIP target (optional)</div>
                    <input
                      className="input"
                      value={target}
                      onChange={(e) => setTarget(e.target.value)}
                      placeholder='e.g. "heart"'
                      disabled={!useBiomedclip}
                    />
                  </label>
                </div>
              </div>

              {imageDataUrl ? <img className="imagePreview" src={imageDataUrl} alt="Selected medical" /> : null}

              {imageDataUrl && useBiomedclip ? (
                <div className="row">
                  <button className="btn btnGhost btnSmall" onClick={onValidateImage} disabled={isCheckingImage}>
                    {isCheckingImage ? "Checking…" : "Check image"}
                  </button>
                  {imageValidation?.report_summary ? (
                    <div className="muted">Image: {imageValidation.report_summary}</div>
                  ) : null}
                </div>
              ) : null}

              <div className="divider" />

              <label className="field">
                <div className="label">Animation prompt</div>
                <input
                  className="input"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder='e.g. "animate red blood cells flowing through the vessel"'
                />
              </label>

              <div className="grid2">
                <label className="field">
                  <div className="label">Post-process</div>
                  <select className="select" value={postprocessMode} onChange={(e) => setPostprocessMode(e.target.value)}>
                    <option value="off">Off</option>
                    <option value="captions">Captions</option>
                    <option value="voiceover">Voiceover + captions</option>
                  </select>
                </label>
                <label className="field">
                  <div className="label">Prompt rewrite</div>
                  <select className="select" value={rewriteMode} onChange={(e) => setRewriteMode(e.target.value)}>
                    <option value="gemini">Gemini</option>
                    <option value="rule">Rule-based</option>
                    <option value="none">None</option>
                  </select>
                </label>
                <label className="field">
                  <div className="label">Veo model</div>
                  <select className="select" value={veoModel} onChange={(e) => setVeoModel(e.target.value)}>
                    <option value="veo-3.1-generate-preview">veo-3.1-generate-preview</option>
                    <option value="veo-3.1-fast-generate-preview">veo-3.1-fast-generate-preview</option>
                  </select>
                </label>
              </div>

              <div className="row">
                <button className="btn btnPrimary" onClick={onGenerate} disabled={job?.status === "running"}>
                  {job?.status === "running" ? "Generating…" : "Generate"}
                </button>
                {jobId ? <div className="muted">Job: {jobId}</div> : null}
                {job ? <div className="muted">Status: {job.status}</div> : null}
              </div>

              {error ? <div className="error">{error}</div> : null}
              {job?.status === "error" ? <div className="error">{job.error || "Generation failed."}</div> : null}

              {job?.status === "done" ? (
                <>
                  <div className="divider" />
                  <div className="muted">
                    Done. Scroll the feed to find it. {job.report_summary ? `Scores: ${job.report_summary}` : ""}
                  </div>
                  <video className="videoPreview" controls src={videoUrl} />
                </>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}

      {extendModalOpen ? (
        <div className="modalOverlay" onClick={() => !isExtending && setExtendModalOpen(false)}>
          <div className="modal modalSmall" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div>
                <div className="modalTitle">Extend Video</div>
                <div className="muted">Continue this video with Veo</div>
              </div>
              <button
                className="btn btnGhost btnSmall"
                onClick={() => setExtendModalOpen(false)}
                disabled={isExtending}
              >
                Close
              </button>
            </div>

            <div className="modalBody">
              <label className="chip">
                <input
                  type="checkbox"
                  checked={extendUseGemini}
                  onChange={(e) => setExtendUseGemini(e.target.checked)}
                  disabled={isExtending}
                />
                <span>Gemini-supervised (auto-generate continuation prompt)</span>
              </label>

              <label className="field">
                <div className="label">
                  {extendUseGemini ? "Extension hint (optional)" : "Extension prompt (required)"}
                </div>
                <input
                  className="input"
                  value={extendPrompt}
                  onChange={(e) => setExtendPrompt(e.target.value)}
                  placeholder={
                    extendUseGemini
                      ? 'e.g. "zoom out to show the full organ" (Gemini will expand this)'
                      : 'e.g. "The camera slowly pulls back to reveal the surrounding tissue..."'
                  }
                  disabled={isExtending}
                />
                {extendUseGemini ? (
                  <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
                    Leave blank for Gemini to decide what happens next based on the original animation.
                  </div>
                ) : null}
              </label>

              <label className="field">
                <div className="label">Post-process</div>
                <select
                  className="select"
                  value={extendPostprocess}
                  onChange={(e) => setExtendPostprocess(e.target.value)}
                  disabled={isExtending}
                >
                  <option value="off">Off</option>
                  <option value="captions">Captions</option>
                  <option value="voiceover">Voiceover + captions</option>
                </select>
              </label>

              <div className="row">
                <button
                  className="btn btnPrimary"
                  onClick={submitExtend}
                  disabled={isExtending}
                >
                  {isExtending ? "Extending…" : "Extend Video"}
                </button>
                {extendNewJobId ? (
                  <div className="muted">New job: {extendNewJobId.slice(0, 8)}…</div>
                ) : null}
              </div>

              {extendError ? <div className="error">{extendError}</div> : null}

              {extendStatusById[extendJobId]?.status === "done" ? (
                <>
                  <div className="divider" />
                  <div className="muted">
                    Done! The extended video has been added to your library. You can close this modal and scroll to find it.
                  </div>
                </>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
