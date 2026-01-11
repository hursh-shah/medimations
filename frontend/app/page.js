"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Play,
  Pause,
  Volume2,
  VolumeX,
  Download,
  Sparkles,
  FolderOpen,
  Plus,
  Image as ImageIcon,
  Wand2,
  Film,
  Settings2,
  RefreshCw,
  Upload,
  X,
  Check,
  AlertCircle,
  Loader2,
  Captions,
  Mic,
  Zap
} from "lucide-react";

// ============================================
// API UTILITIES
// ============================================

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

function formatDate(iso) {
  const s = String(iso || "").trim();
  if (!s) return "";
  const d = new Date(s);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function parseSrtTimestamp(ts) {
  const m = String(ts || "").trim().match(/^(\d+):(\d+):(\d+)[,.](\d+)$/);
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

// ============================================
// VIDEO CARD
// ============================================

function VideoCard({ item, onClick, onPostprocess, onRemoveCaptions, postprocessStatus }) {
  const videoRef = useRef(null);
  const [isHovering, setIsHovering] = useState(false);
  
  const src = item?.narrated_video_url || item?.video_url || "";
  const hasCaptions = Boolean(item?.captions_srt_url);
  const hasVoiceover = Boolean(item?.narrated_video_url);
  const isExtension = Boolean(item?.extended_from);
  const busy = postprocessStatus?.status === "running";

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (isHovering) {
      v.play().catch(() => {});
    } else {
      v.pause();
      v.currentTime = 0;
    }
  }, [isHovering]);

  const downloadUrl = useMemo(() => {
    const base = backendBase();
    const path = hasVoiceover 
      ? `/api/download/videos/${item.job_id}/narrated.mp4`
      : `/api/download/videos/${item.job_id}.mp4`;
    return base ? `${base}${path}` : path;
  }, [item.job_id, hasVoiceover]);

  return (
    <motion.div
      className="videoCard"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      onClick={onClick}
    >
      <div className="videoCardThumb">
        <video ref={videoRef} src={src} muted loop playsInline preload="metadata" />
        <div className="videoCardOverlay">
          <div className="playBtn"><Play /></div>
        </div>
        <div className="videoCardBadges">
          {hasVoiceover && <span className="badge badgeVo">VO</span>}
          {hasCaptions && !hasVoiceover && <span className="badge badgeCc">CC</span>}
          {isExtension && <span className="badge badgeExt">EXT</span>}
        </div>
      </div>
      
      <div className="videoCardInfo">
        <div className="videoCardTitle">{item.prompt || item.job_id.slice(0, 8)}</div>
        <div className="videoCardMeta">{formatDate(item.created_at)}</div>
      </div>
      
      <div className="videoCardActions" onClick={(e) => e.stopPropagation()}>
        <button
          className="btn btnSecondary btnSm"
          disabled={busy}
          onClick={() => hasCaptions ? onRemoveCaptions(item.job_id) : onPostprocess(item.job_id, "captions")}
        >
          {busy && (postprocessStatus?.mode === "captions" || postprocessStatus?.mode === "captions_remove") ? (
            <Loader2 className="animate-spin" size={14} />
          ) : (
            <Captions size={14} />
          )}
        </button>
        <button
          className="btn btnSecondary btnSm"
          disabled={busy || hasVoiceover}
          onClick={() => onPostprocess(item.job_id, "voiceover")}
        >
          {busy && postprocessStatus?.mode === "voiceover" ? (
            <Loader2 className="animate-spin" size={14} />
          ) : (
            <Mic size={14} />
          )}
        </button>
        <a
          className="btn btnSecondary btnSm"
          href={downloadUrl}
          target="_blank"
          rel="noreferrer"
          onClick={(e) => e.stopPropagation()}
        >
          <Download size={14} />
        </a>
      </div>
    </motion.div>
  );
}

// ============================================
// VIDEO PLAYER MODAL
// ============================================

function VideoPlayerModal({ item, isOpen, onClose, muted, onToggleMute, onExtend, extendStatus }) {
  const videoRef = useRef(null);
  const captionsRef = useRef(null);
  const [caption, setCaption] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);

  const src = item?.narrated_video_url || item?.video_url || "";
  const hasCaptions = Boolean(item?.captions_srt_url);
  const hasVoiceover = Boolean(item?.narrated_video_url);
  const extending = extendStatus?.status === "running";

  useEffect(() => {
    if (!isOpen) return;
    captionsRef.current = null;
    setCaption("");
    if (hasCaptions) {
      apiFetchText(item.captions_srt_url).then((txt) => {
        captionsRef.current = parseSrt(txt);
      }).catch(() => {});
    }
  }, [isOpen, hasCaptions, item?.captions_srt_url]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !isOpen) return;
    v.play().catch(() => {});
    setIsPlaying(true);
  }, [isOpen, src]);

  function togglePlay() {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play().catch(() => {}); setIsPlaying(true); }
    else { v.pause(); setIsPlaying(false); }
  }

  function onTimeUpdate() {
    const v = videoRef.current;
    const segs = captionsRef.current || [];
    if (!v || segs.length === 0) return;
    const t = v.currentTime || 0;
    const seg = segs.find((s) => t >= s.start && t <= s.end);
    setCaption(seg?.text || "");
  }

  const downloadUrl = useMemo(() => {
    if (!item) return "";
    const base = backendBase();
    const path = hasVoiceover 
      ? `/api/download/videos/${item.job_id}/narrated.mp4`
      : `/api/download/videos/${item.job_id}.mp4`;
    return base ? `${base}${path}` : path;
  }, [item?.job_id, hasVoiceover]);

  if (!isOpen || !item) return null;

  return (
    <div className="modalOverlay" onClick={onClose}>
      <motion.div
        className="modal"
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modalHeader">
          <div className="modalTitle">{item.prompt || "Animation"}</div>
          <button className="btn btnGhost btnIcon" onClick={onClose}><X size={18} /></button>
        </div>
        
        <div className="modalBody">
          <div className="playerContainer" onClick={togglePlay}>
            <video ref={videoRef} src={src} loop playsInline muted={muted} onTimeUpdate={onTimeUpdate} />
            {caption && <div className="playerCaption">{caption}</div>}
          </div>
          
          <div className="playerInfo">
            <div className="playerMeta">{formatDate(item.created_at)}</div>
            <div className="playerActions">
              <button className="btn btnSecondary btnSm" onClick={togglePlay}>
                {isPlaying ? <Pause size={14} /> : <Play size={14} />}
              </button>
              <button className="btn btnSecondary btnSm" onClick={onToggleMute}>
                {muted ? <VolumeX size={14} /> : <Volume2 size={14} />}
              </button>
              <button className="btn btnPrimary btnSm" disabled={extending} onClick={() => onExtend(item.job_id)}>
                {extending ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}
                Extend
              </button>
              <a className="btn btnSecondary btnSm" href={downloadUrl} target="_blank" rel="noreferrer">
                <Download size={14} />
              </a>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

// ============================================
// LIBRARY VIEW
// ============================================

function LibraryView({ library, onOpenPlayer, onPostprocess, onRemoveCaptions, postprocessById, onCreateNew }) {
  if (library.length === 0) {
    return (
      <div className="emptyState animate-fadeIn">
        <div className="emptyIcon"><Film /></div>
        <div className="emptyTitle">No animations yet</div>
        <div className="emptyDesc">Create your first medical animation to get started.</div>
        <button className="btn btnPrimary" onClick={onCreateNew}>
          <Plus size={16} /> Create
        </button>
      </div>
    );
  }

  return (
    <div className="animate-fadeIn">
      <div className="libraryGrid">
        {library.map((item) => (
          <VideoCard
            key={item.job_id}
            item={item}
            onClick={() => onOpenPlayer(item)}
            onPostprocess={onPostprocess}
            onRemoveCaptions={onRemoveCaptions}
            postprocessStatus={postprocessById[item.job_id]}
          />
        ))}
      </div>
    </div>
  );
}

// ============================================
// CREATE VIEW
// ============================================

function CreateView({ onGenerate, isGenerating, generationStatus, error }) {
  const [imageDataUrl, setImageDataUrl] = useState("");
  const [imagePrompt, setImagePrompt] = useState("");
  const [imageModel, setImageModel] = useState("imagen-4.0-generate-001");
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  
  const [prompt, setPrompt] = useState("");
  const [target, setTarget] = useState("");
  const [rewriteMode, setRewriteMode] = useState("gemini");
  const [veoModel, setVeoModel] = useState("veo-3.1-generate-preview");
  const [useBiomedclip, setUseBiomedclip] = useState(false);
  const [postprocessMode, setPostprocessMode] = useState("voiceover");

  async function readFileAsDataUrl(file) {
    const maxBytes = 8 * 1024 * 1024;
    if (file.size > maxBytes) throw new Error("Image is too large (max 8MB).");
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Failed to read image."));
      reader.onload = () => resolve(String(reader.result || ""));
      reader.readAsDataURL(file);
    });
  }

  async function handleUploadImage(e) {
    const file = e?.target?.files?.[0];
    if (!file) return;
    try {
      const dataUrl = await readFileAsDataUrl(file);
      setImageDataUrl(dataUrl);
    } catch (err) {
      console.error(err);
    }
  }

  async function handleGenerateImage() {
    if (!imagePrompt.trim()) return;
    setIsGeneratingImage(true);
    try {
      const resp = await apiFetchJson("/api/images/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: imagePrompt,
          model: imageModel,
          aspect_ratio: "1:1",
          prompt_rewrite: "gemini",
          gemini_model: "gemini-3-flash-preview",
          use_biomedclip: Boolean(useBiomedclip),
          biomedclip_target: useBiomedclip && target.trim() ? target.trim() : null,
          max_rounds: 2
        })
      });
      setImageDataUrl(resp.image_data_url || "");
    } catch (e) {
      console.error(e);
    } finally {
      setIsGeneratingImage(false);
    }
  }

  function handleSubmit() {
    if (!prompt.trim() || !imageDataUrl) return;
    onGenerate({ prompt, imageDataUrl, rewriteMode, veoModel, postprocessMode, useBiomedclip, target });
  }

  return (
    <div className="createContainer animate-fadeIn">
      <div className="createHeader">
        <h2 className="createTitle">New Animation</h2>
        <p className="createSubtitle">Generate a medical animation from an image and description.</p>
      </div>

      <div className="createForm">
        {/* Source Image */}
        <div className="formGroup">
          <div className="formGroupLabel">Source Image</div>
          {!imageDataUrl ? (
            <div className="formGroupContent">
              <div className="formGrid">
                <label className="uploadZone">
                  <input type="file" accept="image/*" onChange={handleUploadImage} style={{ display: "none" }} />
                  <div className="uploadIcon"><Upload /></div>
                  <div className="uploadText">Upload</div>
                  <div className="uploadHint">PNG, JPG</div>
                </label>
                
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  <input
                    className="input"
                    value={imagePrompt}
                    onChange={(e) => setImagePrompt(e.target.value)}
                    placeholder="Or describe image to generate..."
                  />
                  <div style={{ display: "flex", gap: 8 }}>
                    <select className="select" value={imageModel} onChange={(e) => setImageModel(e.target.value)} style={{ flex: 1 }}>
                      <option value="imagen-4.0-generate-001">Imagen 4</option>
                      <option value="imagen-4.0-ultra-generate-001">Imagen 4 Ultra</option>
                      <option value="imagen-4.0-fast-generate-001">Imagen 4 Fast</option>
                    </select>
                    <button 
                      className="btn btnPrimary"
                      onClick={handleGenerateImage}
                      disabled={isGeneratingImage || !imagePrompt.trim()}
                    >
                      {isGeneratingImage ? <Loader2 size={16} className="animate-spin" /> : "Generate"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="imagePreview">
              <img src={imageDataUrl} alt="Medical" />
              <div className="imagePreviewOverlay">
                <button className="btn btnSecondary btnSm" onClick={() => setImageDataUrl("")}>
                  <X size={14} /> Remove
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Animation */}
        <div className="formGroup">
          <div className="formGroupLabel">Animation</div>
          <div className="formGroupContent">
            <textarea
              className="textarea"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe what should happen in the animation..."
              style={{ marginBottom: 16 }}
            />
            <div className="formGrid">
              <div className="formField">
                <div className="formLabel">Output</div>
                <select className="select" value={postprocessMode} onChange={(e) => setPostprocessMode(e.target.value)}>
                  <option value="off">Video only</option>
                  <option value="captions">With captions</option>
                  <option value="voiceover">With voiceover</option>
                </select>
              </div>
              <div className="formField">
                <div className="formLabel">Video Model</div>
                <select className="select" value={veoModel} onChange={(e) => setVeoModel(e.target.value)}>
                  <option value="veo-3.1-generate-preview">Veo 3.1</option>
                  <option value="veo-3.1-fast-generate-preview">Veo 3.1 Fast</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Options - collapsed style */}
        <div className="formGroup">
          <div className="formGroupLabel">Options</div>
          <div className="formGroupContent">
            <div className="formGrid">
              <div className="formField">
                <div className="formLabel">Prompt rewrite</div>
                <select className="select" value={rewriteMode} onChange={(e) => setRewriteMode(e.target.value)}>
                  <option value="gemini">Gemini 3</option>
                  <option value="rule">Rule-based</option>
                  <option value="none">Off</option>
                </select>
              </div>
              <div className="formField">
                <label className="checkbox" style={{ marginTop: 24 }}>
                  <input type="checkbox" checked={useBiomedclip} onChange={(e) => setUseBiomedclip(e.target.checked)} />
                  <span>BiomedCLIP validation</span>
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* Submit */}
        <div className="formActions">
          <div className="textSm textTertiary">
            {!imageDataUrl && "Add an image to continue"}
            {imageDataUrl && !prompt.trim() && "Add animation description"}
            {imageDataUrl && prompt.trim() && "Ready"}
          </div>
          <button
            className="btn btnPrimary"
            onClick={handleSubmit}
            disabled={isGenerating || !prompt.trim() || !imageDataUrl}
          >
            {isGenerating ? <><Loader2 size={16} className="animate-spin" /> Generating...</> : "Generate Animation"}
          </button>
        </div>

        {generationStatus && (
          <div className={`statusMsg ${generationStatus.status === "done" ? "success" : generationStatus.status === "error" ? "error" : "info"}`}>
            {generationStatus.status === "done" ? <Check size={16} /> : generationStatus.status === "error" ? <AlertCircle size={16} /> : <Loader2 size={16} className="animate-spin" />}
            {generationStatus.message}
          </div>
        )}

        {error && <div className="statusMsg error"><AlertCircle size={16} />{error}</div>}
      </div>
    </div>
  );
}

// ============================================
// EXTEND MODAL
// ============================================

function ExtendModal({ isOpen, onClose, jobId, onSubmit, isExtending, error, success }) {
  const [prompt, setPrompt] = useState("");
  const [useGemini, setUseGemini] = useState(true);
  const [postprocess, setPostprocess] = useState("voiceover");

  useEffect(() => {
    if (isOpen) { setPrompt(""); setUseGemini(true); setPostprocess("voiceover"); }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modalOverlay" onClick={() => !isExtending && onClose()}>
      <motion.div className="modal modalSm" initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} onClick={(e) => e.stopPropagation()}>
        <div className="modalHeader">
          <div className="modalTitle">Extend Animation</div>
          <button className="btn btnGhost btnIcon" onClick={onClose} disabled={isExtending}><X size={18} /></button>
        </div>
        
        <div className="modalBody">
          <label className="checkbox mb4">
            <input type="checkbox" checked={useGemini} onChange={(e) => setUseGemini(e.target.checked)} disabled={isExtending} />
            <span>Gemini-supervised continuation</span>
          </label>

          <div className="formField mb4">
            <div className="formLabel">{useGemini ? "Hint (optional)" : "Prompt (required)"}</div>
            <textarea
              className="textarea"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={useGemini ? 'e.g. "zoom out"' : 'Describe continuation...'}
              disabled={isExtending}
            />
          </div>

          <div className="formField">
            <div className="formLabel">Post-process</div>
            <select className="select" value={postprocess} onChange={(e) => setPostprocess(e.target.value)} disabled={isExtending}>
              <option value="off">None</option>
              <option value="captions">Captions</option>
              <option value="voiceover">Voiceover</option>
            </select>
          </div>

          {error && <div className="statusMsg mt4 error"><AlertCircle size={16} />{error}</div>}
          {success && <div className="statusMsg mt4 success"><Check size={16} />Done!</div>}
        </div>

        <div className="modalFooter">
          <button className="btn btnSecondary" onClick={onClose} disabled={isExtending}>Cancel</button>
          <button className="btn btnPrimary" onClick={() => onSubmit({ prompt, useGemini, postprocess })} disabled={isExtending || (!useGemini && !prompt.trim())}>
            {isExtending ? <><Loader2 size={16} className="animate-spin" /> Extending...</> : <><Zap size={16} /> Extend</>}
          </button>
        </div>
      </motion.div>
    </div>
  );
}

// ============================================
// MAIN APP
// ============================================

export default function Home() {
  const [activeTab, setActiveTab] = useState("library");
  const [muted, setMuted] = useState(true);
  const [library, setLibrary] = useState([]);
  const [postprocessById, setPostprocessById] = useState({});
  const [extendStatusById, setExtendStatusById] = useState({});
  
  const [playerItem, setPlayerItem] = useState(null);
  
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationStatus, setGenerationStatus] = useState(null);
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState(null);
  
  const [extendModalOpen, setExtendModalOpen] = useState(false);
  const [extendJobId, setExtendJobId] = useState(null);
  const [isExtending, setIsExtending] = useState(false);
  const [extendError, setExtendError] = useState("");
  const [extendSuccess, setExtendSuccess] = useState(false);

  const pollRef = useRef(null);
  const postprocessPollersRef = useRef({});
  const extendPollersRef = useRef({});

  async function refreshLibrary() {
    try { const items = await apiFetchJson("/api/library"); setLibrary(items); } catch {}
  }

  async function refreshJob(id) {
    try {
      const data = await apiFetchJson(`/api/jobs/${id}`);
      if (data.job.status === "done") {
        if (pollRef.current) clearInterval(pollRef.current);
        setIsGenerating(false);
        setGenerationStatus({ status: "done", message: "Animation created!" });
        await refreshLibrary();
        setTimeout(() => { setActiveTab("library"); setGenerationStatus(null); }, 1500);
      } else if (data.job.status === "error") {
        if (pollRef.current) clearInterval(pollRef.current);
        setIsGenerating(false);
        setGenerationStatus({ status: "error", message: data.job.error || "Failed" });
      } else {
        setGenerationStatus({ status: "running", message: `${data.job.status}...` });
      }
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        if (pollRef.current) clearInterval(pollRef.current);
        setError("Job not found");
      }
    }
  }

  async function onGenerate(params) {
    setError("");
    setGenerationStatus({ status: "running", message: "Starting..." });
    setIsGenerating(true);
    try {
      const resp = await apiFetchJson("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: params.prompt, backend: "veo", input_image: params.imageDataUrl,
          prompt_rewrite: params.rewriteMode, gemini_model: "gemini-3-flash-preview",
          veo_model: params.veoModel, postprocess_mode: params.postprocessMode,
          use_biomedclip: false, biomedclip_target: null
        })
      });
      setJobId(resp.job_id);
      await refreshJob(resp.job_id);
      pollRef.current = setInterval(() => refreshJob(resp.job_id).catch(() => {}), 2000);
    } catch (e) {
      setError(String(e?.message || e));
      setIsGenerating(false);
      setGenerationStatus(null);
    }
  }

  async function onPostprocess(jobId, mode) {
    setPostprocessById((prev) => ({ ...prev, [jobId]: { status: "running", mode, error: "" } }));
    if (postprocessPollersRef.current[jobId]) { clearInterval(postprocessPollersRef.current[jobId]); delete postprocessPollersRef.current[jobId]; }
    try {
      const resp = await apiFetchJson(`/api/jobs/${jobId}/postprocess`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode, force: false }) });
      const poll = async () => {
        try {
          const data = await apiFetchJson(resp.status_url);
          const ps = String(data?.job?.postprocess_status || "").trim();
          if (ps && ps !== "running") {
            clearInterval(postprocessPollersRef.current[jobId]);
            delete postprocessPollersRef.current[jobId];
            setPostprocessById((prev) => ({ ...prev, [jobId]: { status: ps === "error" ? "error" : "done", mode } }));
            await refreshLibrary();
          }
        } catch {}
      };
      postprocessPollersRef.current[jobId] = setInterval(poll, 2000);
      await poll();
    } catch (e) {
      setPostprocessById((prev) => ({ ...prev, [jobId]: { status: "error", mode, error: String(e?.message || e) } }));
    }
  }

  async function onRemoveCaptions(jobId) {
    setPostprocessById((prev) => ({ ...prev, [jobId]: { status: "running", mode: "captions_remove" } }));
    try {
      await apiFetchJson(`/api/jobs/${jobId}/captions`, { method: "DELETE" });
      setPostprocessById((prev) => ({ ...prev, [jobId]: { status: "done", mode: "captions_remove" } }));
      await refreshLibrary();
    } catch (e) {
      setPostprocessById((prev) => ({ ...prev, [jobId]: { status: "error", mode: "captions_remove" } }));
    }
  }

  function openExtendModal(jobId) { setExtendJobId(jobId); setExtendError(""); setExtendSuccess(false); setExtendModalOpen(true); }

  async function submitExtend(params) {
    if (!extendJobId) return;
    if (!params.useGemini && !params.prompt.trim()) { setExtendError("Prompt required"); return; }
    setExtendError(""); setIsExtending(true);
    setExtendStatusById((prev) => ({ ...prev, [extendJobId]: { status: "running" } }));
    try {
      const resp = await apiFetchJson(`/api/jobs/${extendJobId}/extend`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: params.prompt.trim() || null, use_gemini: params.useGemini, postprocess_mode: params.postprocess }) });
      const pollExtend = async () => {
        try {
          const data = await apiFetchJson(resp.status_url);
          const status = String(data?.job?.status || "").trim();
          if (status === "done" || status === "error") {
            if (extendPollersRef.current[extendJobId]) { clearInterval(extendPollersRef.current[extendJobId]); delete extendPollersRef.current[extendJobId]; }
            setExtendStatusById((prev) => ({ ...prev, [extendJobId]: { status: status === "error" ? "error" : "done" } }));
            setIsExtending(false);
            if (status === "done") { setExtendSuccess(true); await refreshLibrary(); setTimeout(() => { setExtendModalOpen(false); setPlayerItem(null); }, 1200); }
            else { setExtendError(data?.job?.error || "Failed"); }
          }
        } catch {}
      };
      extendPollersRef.current[extendJobId] = setInterval(pollExtend, 2000);
      await pollExtend();
    } catch (e) {
      setExtendStatusById((prev) => ({ ...prev, [extendJobId]: { status: "error" } }));
      setExtendError(String(e?.message || e));
      setIsExtending(false);
    }
  }

  useEffect(() => {
    refreshLibrary();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      Object.values(postprocessPollersRef.current || {}).forEach(clearInterval);
      Object.values(extendPollersRef.current || {}).forEach(clearInterval);
    };
  }, []);

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebarHeader">
          <div className="logo">
            <div className="logoIcon"><Activity /></div>
            <span className="logoText">Medimations</span>
          </div>
        </div>
        
        <nav className="sidebarNav">
          <button className={`navItem ${activeTab === "library" ? "active" : ""}`} onClick={() => setActiveTab("library")}>
            <FolderOpen /> Library
          </button>
          <button className={`navItem ${activeTab === "create" ? "active" : ""}`} onClick={() => setActiveTab("create")}>
            <Plus /> Create
          </button>
        </nav>
        
        <div className="sidebarFooter">
          <div className="statusPill">
            <div className="statusDot" />
            <span>{backendBase() ? "Railway status: online" : "Local mode"}</span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="mainContent">
        <header className="contentHeader">
          <h1 className="contentTitle">{activeTab === "library" ? "Library" : "Create"}</h1>
          <div className="headerActions">
            <button className="btn btnGhost btnSm" onClick={() => setMuted((m) => !m)}>
              {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
            </button>
            <button className="btn btnSecondary btnSm" onClick={refreshLibrary}>
              <RefreshCw size={16} />
            </button>
            {activeTab === "library" && (
              <button className="btn btnPrimary btnSm" onClick={() => setActiveTab("create")}>
                <Plus size={16} /> New
              </button>
            )}
          </div>
        </header>

        <div className="contentBody">
          {activeTab === "library" && (
            <LibraryView
              library={library}
              onOpenPlayer={setPlayerItem}
              onPostprocess={onPostprocess}
              onRemoveCaptions={onRemoveCaptions}
              postprocessById={postprocessById}
              onCreateNew={() => setActiveTab("create")}
            />
          )}
          {activeTab === "create" && (
            <CreateView
              onGenerate={onGenerate}
              isGenerating={isGenerating}
              generationStatus={generationStatus}
              error={error}
            />
          )}
        </div>
      </main>

      {/* Modals */}
      <AnimatePresence>
        {playerItem && (
          <VideoPlayerModal
            item={playerItem}
            isOpen={!!playerItem}
            onClose={() => setPlayerItem(null)}
            muted={muted}
            onToggleMute={() => setMuted((m) => !m)}
            onExtend={openExtendModal}
            extendStatus={extendStatusById[playerItem?.job_id]}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        <ExtendModal
          isOpen={extendModalOpen}
          onClose={() => setExtendModalOpen(false)}
          jobId={extendJobId}
          onSubmit={submitExtend}
          isExtending={isExtending}
          error={extendError}
          success={extendSuccess}
        />
      </AnimatePresence>
    </div>
  );
}
