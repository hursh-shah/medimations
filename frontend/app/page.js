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

    try {
      const resp = await apiFetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          backend: "veo",
          prompt_rewrite: rewriteMode,
          gemini_model: "gemini-2.0-flash",
          veo_model: veoModel,
          use_biomedclip: Boolean(useBiomedclip),
          biomedclip_target: useBiomedclip && target.trim() ? target.trim() : null
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
          <p className="muted">Veo 3.1 generation + BiomedCLIP verification + up to 1 reprompt.</p>
        </div>
        <a className="link" href="https://vercel.com" target="_blank" rel="noreferrer">
          Deploy on Vercel
        </a>
      </header>

      <section className="card">
        <h2>Generate</h2>
        <div className="grid2">
          <label className="field">
            <span>Prompt</span>
            <input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder='e.g. "heart surgery video" or "red blood cells moving through a capillary"'
            />
          </label>
          <label className="field">
            <span>BiomedCLIP target (optional)</span>
            <input
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder='e.g. "heart" or "capillary"'
              disabled={!useBiomedclip}
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
        <label className="row" style={{ gap: 8 }}>
          <input
            type="checkbox"
            checked={useBiomedclip}
            onChange={(e) => setUseBiomedclip(e.target.checked)}
          />
          <span className="muted">Use BiomedCLIP medical validator</span>
        </label>
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
