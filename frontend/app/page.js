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

async function apiFetch(path, init) {
  const base = backendBase();
  const url = base ? `${base}${path}` : path;
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(res.status, text);
  }
  return res.json();
}

export default function Home() {
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
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [library, setLibrary] = useState([]);

  const pollRef = useRef(null);
  const videoUrl = useMemo(() => {
    if (!jobId) return "";
    const base = backendBase();
    const path = `/api/videos/${jobId}.mp4`;
    return base ? `${base}${path}` : path;
  }, [jobId]);

  async function refreshLibrary() {
    try {
      const items = await apiFetch("/api/library");
      setLibrary(items);
    } catch (e) {
      // Non-fatal if the backend isn't up yet.
    }
  }

  async function refreshJob(id) {
    try {
      const data = await apiFetch(`/api/jobs/${id}`);
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
      setError("Enter a prompt.");
      return;
    }
    if (!imageDataUrl) {
      setError("Select or generate a medical image first.");
      return;
    }

    try {
      const resp = await apiFetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          backend: "veo",
          input_image: imageDataUrl,
          prompt_rewrite: rewriteMode,
          gemini_model: "gemini-2.0-flash",
          veo_model: veoModel,
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

  async function onTestNoVeo() {
    setError("");
    setJob(null);
    setJobId(null);

    try {
      const resp = await apiFetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt.trim() ? prompt.trim() : "demo: falling red dot",
          backend: "demo",
          input_image: imageDataUrl || null,
          prompt_rewrite: "none",
          use_biomedclip: false,
          max_rounds: 1,
          candidates: 1,
          medical_threshold: 0.0,
          physics_threshold: 0.0,
          fps: 8,
          duration_s: 2.0,
          width: 256,
          height: 256
        })
      });
      setJobId(resp.job_id);
      await refreshJob(resp.job_id);
      pollRef.current = setInterval(() => refreshJob(resp.job_id).catch(() => {}), 500);
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
      const resp = await apiFetch("/api/images/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: imagePrompt,
          model: imageModel,
          aspect_ratio: "1:1",
          prompt_rewrite: "gemini",
          gemini_model: "gemini-2.0-flash",
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
      const resp = await apiFetch("/api/images/validate", {
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

  useEffect(() => {
    refreshLibrary();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  return (
    <main className="stack">
      <header className="header">
        <div>
          <h1>Medical Diffusion</h1>
          <p className="muted">Image (BiomedCLIP-checked) + text → Veo 3.1 animation (max 1 image reprompt).</p>
        </div>
        <a className="link" href="https://vercel.com" target="_blank" rel="noreferrer">
          Deploy on Vercel
        </a>
      </header>

      <section className="card">
        <h2>Generate</h2>
        <div className="grid2">
          <label className="field">
            <span>Upload a medical image (PNG/JPG)</span>
            <input type="file" accept="image/*" onChange={onUploadImage} />
          </label>
          <div className="field">
            <span>Or generate one with AI</span>
            <div className="row" style={{ marginTop: 0 }}>
              <input
                value={imagePrompt}
                onChange={(e) => setImagePrompt(e.target.value)}
                placeholder='e.g. "axial CT slice of the liver, clinical, grayscale, no text"'
              />
              <button className="btnSecondary" onClick={onGenerateImage} disabled={isGeneratingImage}>
                {isGeneratingImage ? "Generating…" : "Generate image"}
              </button>
            </div>
            <div className="row" style={{ marginTop: 8 }}>
              <select value={imageModel} onChange={(e) => setImageModel(e.target.value)}>
                <option value="imagen-3.0-generate-001">imagen-3.0-generate-001</option>
                <option value="imagen-3.0-fast-generate-001">imagen-3.0-fast-generate-001</option>
              </select>
              <label className="row muted" style={{ gap: 8, marginTop: 0 }}>
                <input
                  type="checkbox"
                  checked={useBiomedclip}
                  onChange={(e) => setUseBiomedclip(e.target.checked)}
                />
                <span>Validate image with BiomedCLIP (reprompt once)</span>
              </label>
              {imageDataUrl ? (
                <button
                  className="btnSecondary"
                  onClick={() => {
                    setImageDataUrl("");
                    setImageValidation(null);
                  }}
                >
                  Clear image
                </button>
              ) : null}
            </div>
            <label className="field" style={{ marginTop: 10 }}>
              <span>BiomedCLIP target (optional)</span>
              <input
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder='e.g. "heart" or "liver"'
                disabled={!useBiomedclip}
              />
            </label>
          </div>
        </div>

        <div className="imageRow">
          {imageDataUrl ? (
            <img className="imagePreview" src={imageDataUrl} alt="Selected medical image" />
          ) : (
            <div className="imagePlaceholder muted">No image selected yet.</div>
          )}
        </div>
        {imageDataUrl && useBiomedclip ? (
          <div className="row" style={{ marginTop: 8 }}>
            <button className="btnSecondary" onClick={onValidateImage} disabled={isCheckingImage}>
              {isCheckingImage ? "Checking…" : "Check image"}
            </button>
          </div>
        ) : null}
        {imageValidation?.report_summary ? (
          <div className="muted" style={{ marginTop: 8 }}>
            Image check: {imageValidation.report_summary}
            {typeof imageValidation.accepted === "boolean" ? (imageValidation.accepted ? " • accepted" : " • not accepted") : ""}
            {imageValidation.rounds ? ` • rounds=${imageValidation.rounds}` : ""}
          </div>
        ) : null}

        <div className="grid2">
          <label className="field">
            <span>Animation prompt (what should happen over time)</span>
            <input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder='e.g. "animate red blood cells flowing through the vessel"'
            />
          </label>
        </div>
        <div className="grid2">
          <label className="field">
            <span>Prompt rewriting</span>
            <select value={rewriteMode} onChange={(e) => setRewriteMode(e.target.value)}>
              <option value="gemini">Gemini (best quality, uses quota)</option>
              <option value="rule">Rule-based (no quota)</option>
              <option value="none">None</option>
            </select>
          </label>
          <label className="field">
            <span>Veo model</span>
            <select value={veoModel} onChange={(e) => setVeoModel(e.target.value)}>
              <option value="veo-3.1-generate-preview">veo-3.1-generate-preview</option>
              <option value="veo-3.1-fast-generate-preview">veo-3.1-fast-generate-preview</option>
            </select>
          </label>
        </div>
        <div className="row">
          <button className="btn" onClick={onGenerate} disabled={job?.status === "running"}>
            {job?.status === "running" ? "Generating…" : "Generate (Veo)"}
          </button>
          <button className="btnSecondary" onClick={onTestNoVeo} disabled={job?.status === "running"}>
            Test (no Veo)
          </button>
          {jobId ? <span className="muted">Job: {jobId}</span> : null}
          {job ? <span className="muted">Status: {job.status}</span> : null}
        </div>
        {error ? <div className="error">{error}</div> : null}
        {job?.status === "error" ? <div className="error">{job.error || "Generation failed."}</div> : null}

        {job?.status === "done" ? (
          <div className="preview">
            <h3>Result</h3>
            <video className="video" controls src={videoUrl} />
            <div className="row">
              <a className="btnSecondary" href={videoUrl} target="_blank" rel="noreferrer">
                Open video
              </a>
              <a className="btnSecondary" href={videoUrl} download={`${jobId}.mp4`}>
                Download
              </a>
            </div>
            {job.report_summary ? <p className="muted">Scores: {job.report_summary}</p> : null}
          </div>
        ) : null}
      </section>

      <section className="card">
        <div className="row between">
          <h2>Library</h2>
          <button className="btnSecondary" onClick={refreshLibrary}>
            Refresh
          </button>
        </div>
        {library.length === 0 ? (
          <p className="muted">No saved videos yet.</p>
        ) : (
          <div className="library">
            {library.map((item) => (
              <div key={item.job_id} className="libraryItem">
                <div className="libraryText">
                  <div className="libraryTitle">{item.prompt || item.job_id}</div>
                  <div className="muted">
                    {item.created_at} {item.report_summary ? `• ${item.report_summary}` : ""}
                  </div>
                </div>
                <div className="row">
                  <a className="btnSecondary" href={item.video_url} target="_blank" rel="noreferrer">
                    Open
                  </a>
                  <a className="btnSecondary" href={item.video_url} download={`${item.job_id}.mp4`}>
                    Download
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <footer className="footer muted">
        Backend URL: {backendBase() || "(same origin)"} • Set <code>NEXT_PUBLIC_BACKEND_URL</code> in Vercel env vars.
      </footer>
    </main>
  );
}
