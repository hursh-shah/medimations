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
  ArrowRight,
  ChevronRight,
  Clock,
  Maximize2,
  MoreHorizontal,
  Trash2,
  Share,
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

function clampIso(iso) {
  const s = String(iso || "").trim();
  if (!s) return "";
  const d = new Date(s);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
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

// ============================================
// VIDEO CARD COMPONENT
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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      onClick={onClick}
    >
      <div className="videoCardThumbnail">
        <video
          ref={videoRef}
          src={src}
          muted
          loop
          playsInline
          preload="metadata"
        />
        <div className="videoCardOverlay">
          <div className="playButton">
            <Play />
          </div>
        </div>
        
        <div className="videoCardBadges">
          {hasVoiceover && (
            <span className="badge badgeVoiceover">Voiceover</span>
          )}
          {hasCaptions && !hasVoiceover && (
            <span className="badge badgeCaptions">CC</span>
          )}
          {isExtension && (
            <span className="badge badgeExtended">Extended</span>
          )}
        </div>
      </div>
      
      <div className="videoCardInfo">
        <div className="videoCardTitle">
          {item.prompt || item.job_id.slice(0, 8)}
        </div>
        <div className="videoCardMeta">
          {clampIso(item.created_at)}
          {item.report_summary ? ` • ${item.report_summary}` : ""}
        </div>
      </div>
      
      <div className="videoCardActions" onClick={(e) => e.stopPropagation()}>
        <button
          className="btn btnSecondary btnSm"
          disabled={busy}
          onClick={() => hasCaptions ? onRemoveCaptions(item.job_id) : onPostprocess(item.job_id, "captions")}
        >
          {busy && (postprocessStatus?.mode === "captions" || postprocessStatus?.mode === "captions_remove") ? (
            <Loader2 className="animate-pulse" size={14} />
          ) : (
            <Captions size={14} />
          )}
          {hasCaptions ? "Remove CC" : "Add CC"}
        </button>
        <button
          className="btn btnSecondary btnSm"
          disabled={busy || hasVoiceover}
          onClick={() => onPostprocess(item.job_id, "voiceover")}
        >
          {busy && postprocessStatus?.mode === "voiceover" ? (
            <Loader2 className="animate-pulse" size={14} />
          ) : (
            <Mic size={14} />
          )}
          {hasVoiceover ? "VO Added" : "Add VO"}
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
      apiFetchText(item.captions_srt_url)
        .then((txt) => {
          captionsRef.current = parseSrt(txt);
        })
        .catch(() => {});
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
    if (v.paused) {
      v.play().catch(() => {});
      setIsPlaying(true);
    } else {
      v.pause();
      setIsPlaying(false);
    }
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
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modalHeader">
          <div className="modalTitle">{item.prompt || "Animation"}</div>
          <button className="btn btnGhost btnIcon" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        
        <div className="modalBody">
          <div className="playerContainer" onClick={togglePlay}>
            <video
              ref={videoRef}
              src={src}
              loop
              playsInline
              muted={muted}
              onTimeUpdate={onTimeUpdate}
            />
            {caption && <div className="captionOverlay">{caption}</div>}
          </div>
          
          <div className="playerDetails">
            <div className="flex itemsCenter justifyBetween">
              <div>
                <div className="videoCardMeta">
                  {clampIso(item.created_at)}
                  {item.report_summary ? ` • ${item.report_summary}` : ""}
                </div>
              </div>
              <div className="flex gap2">
                <button className="btn btnSecondary btnSm" onClick={togglePlay}>
                  {isPlaying ? <Pause size={16} /> : <Play size={16} />}
                  {isPlaying ? "Pause" : "Play"}
                </button>
                <button className="btn btnSecondary btnSm" onClick={onToggleMute}>
                  {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
                  {muted ? "Unmute" : "Mute"}
                </button>
                <button
                  className="btn btnPrimary btnSm"
                  disabled={extending}
                  onClick={() => onExtend(item.job_id)}
                >
                  {extending ? <Loader2 size={16} className="animate-pulse" /> : <Zap size={16} />}
                  Extend
                </button>
                <a
                  className="btn btnSecondary btnSm"
                  href={downloadUrl}
                  target="_blank"
                  rel="noreferrer"
                >
                  <Download size={16} />
                  Download
                </a>
              </div>
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

function LibraryView({ 
  library, 
  onRefresh, 
  onOpenPlayer,
  onPostprocess,
  onRemoveCaptions,
  postprocessById,
  onCreateNew
}) {
  if (library.length === 0) {
    return (
      <div className="emptyState animate-fadeIn">
        <div className="emptyIcon">
          <Film />
        </div>
        <div className="emptyTitle">Your library is empty</div>
        <div className="emptyDescription">
          Create your first medical animation to see it here. 
          Generate stunning AI-powered visualizations in seconds.
        </div>
        <button className="btn btnPrimary" onClick={onCreateNew}>
          <Plus size={18} />
          Create Animation
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

function CreateView({ 
  onBack,
  onGenerate,
  isGenerating,
  generationStatus,
  error
}) {
  const [imageDataUrl, setImageDataUrl] = useState("");
  const [imagePrompt, setImagePrompt] = useState("");
  const [imageModel, setImageModel] = useState("imagen-4.0-generate-001");
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  const [isCheckingImage, setIsCheckingImage] = useState(false);
  const [imageValidation, setImageValidation] = useState(null);
  
  const [prompt, setPrompt] = useState("");
  const [target, setTarget] = useState("");
  const [rewriteMode, setRewriteMode] = useState("gemini");
  const [veoModel, setVeoModel] = useState("veo-3.1-generate-preview");
  const [useBiomedclip, setUseBiomedclip] = useState(true);
  const [postprocessMode, setPostprocessMode] = useState("voiceover");

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

  async function handleUploadImage(e) {
    const file = e?.target?.files?.[0];
    if (!file) return;
    try {
      const dataUrl = await readFileAsDataUrl(file);
      setImageDataUrl(dataUrl);
      setImageValidation(null);
    } catch (err) {
      console.error(err);
    }
  }

  async function handleGenerateImage() {
    if (!imagePrompt.trim()) return;
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
          gemini_model: "gemini-3-flash-preview",
          use_biomedclip: Boolean(useBiomedclip),
          biomedclip_target: useBiomedclip && target.trim() ? target.trim() : null,
          max_rounds: 2
        })
      });
      setImageDataUrl(resp.image_data_url || "");
      setImageValidation(resp || null);
    } catch (e) {
      console.error(e);
    } finally {
      setIsGeneratingImage(false);
    }
  }

  async function handleValidateImage() {
    if (!imageDataUrl) return;
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
      console.error(e);
    } finally {
      setIsCheckingImage(false);
    }
  }

  function handleSubmit() {
    if (!prompt.trim() || !imageDataUrl) return;
    onGenerate({
      prompt,
      imageDataUrl,
      rewriteMode,
      veoModel,
      postprocessMode,
      useBiomedclip,
      target
    });
  }

  return (
    <div className="createContainer animate-slideUp">
      <div className="createCard">
        {/* Step 1: Image */}
        <div className="createSection">
          <div className="sectionHeader">
            <div className="sectionIcon">
              <ImageIcon />
            </div>
            <div>
              <div className="sectionTitle">Medical Image</div>
              <div className="sectionDescription">Upload or generate a base image for your animation</div>
            </div>
          </div>

          {!imageDataUrl ? (
            <div className="formGrid">
              <label className="uploadZone">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleUploadImage}
                  style={{ display: "none" }}
                />
                <div className="uploadIcon">
                  <Upload />
                </div>
                <div className="uploadText">Upload Medical Image</div>
                <div className="uploadHint">PNG, JPG up to 8MB</div>
              </label>
              
              <div className="formField">
                <div className="formLabel">Or generate with AI</div>
                <div className="inlineRow">
                  <input
                    className="input"
                    value={imagePrompt}
                    onChange={(e) => setImagePrompt(e.target.value)}
                    placeholder='e.g. "axial CT slice of the liver"'
                  />
                  <button 
                    className="btn btnPrimary btnSm"
                    onClick={handleGenerateImage}
                    disabled={isGeneratingImage || !imagePrompt.trim()}
                  >
                    {isGeneratingImage ? (
                      <Loader2 size={16} className="animate-pulse" />
                    ) : (
                      <Wand2 size={16} />
                    )}
                    Generate
                  </button>
                </div>
                <div className="mt2">
                  <select
                    className="select"
                    value={imageModel}
                    onChange={(e) => setImageModel(e.target.value)}
                  >
                    <option value="imagen-4.0-generate-001">Imagen 4.0</option>
                    <option value="imagen-4.0-ultra-generate-001">Imagen 4.0 Ultra</option>
                    <option value="imagen-4.0-fast-generate-001">Imagen 4.0 Fast</option>
                  </select>
                </div>
              </div>
            </div>
          ) : (
            <div>
              <div className="imagePreview">
                <img src={imageDataUrl} alt="Selected medical" />
                <div className="imagePreviewActions">
                  <button
                    className="btn btnSecondary btnSm"
                    onClick={() => {
                      setImageDataUrl("");
                      setImageValidation(null);
                    }}
                  >
                    <X size={14} />
                    Remove
                  </button>
                </div>
              </div>
              
              {useBiomedclip && (
                <div className="mt3 flex itemsCenter gap3">
                  <button
                    className="btn btnSecondary btnSm"
                    onClick={handleValidateImage}
                    disabled={isCheckingImage}
                  >
                    {isCheckingImage ? (
                      <Loader2 size={14} className="animate-pulse" />
                    ) : (
                      <Check size={14} />
                    )}
                    Validate Image
                  </button>
                  {imageValidation?.report_summary && (
                    <span className="textSm textSecondary">
                      {imageValidation.report_summary}
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Step 2: Animation */}
        <div className="createSection">
          <div className="sectionHeader">
            <div className="sectionIcon">
              <Film />
            </div>
            <div>
              <div className="sectionTitle">Animation Details</div>
              <div className="sectionDescription">Describe what you want to animate</div>
            </div>
          </div>

          <div className="formField mb3">
            <div className="formLabel">Animation Prompt</div>
            <textarea
              className="textarea"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder='e.g. "Animate red blood cells flowing through the vessel, showing pulsating movement with each heartbeat"'
            />
          </div>

          <div className="formGrid">
            <div className="formField">
              <div className="formLabel">Post-processing</div>
              <select 
                className="select"
                value={postprocessMode}
                onChange={(e) => setPostprocessMode(e.target.value)}
              >
                <option value="off">None</option>
                <option value="captions">Captions Only</option>
                <option value="voiceover">Voiceover + Captions</option>
              </select>
            </div>
            <div className="formField">
              <div className="formLabel">Veo Model</div>
              <select
                className="select"
                value={veoModel}
                onChange={(e) => setVeoModel(e.target.value)}
              >
                <option value="veo-3.1-generate-preview">Veo 3.1</option>
                <option value="veo-3.1-fast-generate-preview">Veo 3.1 Fast</option>
              </select>
            </div>
          </div>
        </div>

        {/* Step 3: Settings */}
        <div className="createSection">
          <div className="sectionHeader">
            <div className="sectionIcon">
              <Settings2 />
            </div>
            <div>
              <div className="sectionTitle">Advanced Settings</div>
              <div className="sectionDescription">Fine-tune generation parameters</div>
            </div>
          </div>

          <div className="formGrid">
            <div className="formField">
              <div className="formLabel">Prompt Enhancement</div>
              <select
                className="select"
                value={rewriteMode}
                onChange={(e) => setRewriteMode(e.target.value)}
              >
                <option value="gemini">Gemini AI</option>
                <option value="rule">Rule-based</option>
                <option value="none">None</option>
              </select>
            </div>
            <div className="formField">
              <div className="formLabel">BiomedCLIP Target (Optional)</div>
              <input
                className="input"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder='e.g. "heart"'
                disabled={!useBiomedclip}
              />
            </div>
          </div>

          <div className="mt3">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={useBiomedclip}
                onChange={(e) => setUseBiomedclip(e.target.checked)}
              />
              <span>Enable BiomedCLIP validation for medical accuracy</span>
            </label>
          </div>
        </div>

        {/* Generate Button */}
        <div className="createSection">
          <div className="flex itemsCenter justifyBetween">
            <div className="textSm textSecondary">
              {imageDataUrl ? "Image ready" : "Upload or generate an image first"}
            </div>
            <button
              className="btn btnPrimary"
              onClick={handleSubmit}
              disabled={isGenerating || !prompt.trim() || !imageDataUrl}
            >
              {isGenerating ? (
                <>
                  <Loader2 size={18} className="animate-pulse" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Generate Animation
                </>
              )}
            </button>
          </div>

          {generationStatus && (
            <div className={`statusMessage mt3 ${generationStatus.status === "done" ? "success" : generationStatus.status === "error" ? "error" : "info"}`}>
              {generationStatus.status === "done" ? (
                <Check size={18} />
              ) : generationStatus.status === "error" ? (
                <AlertCircle size={18} />
              ) : (
                <Loader2 size={18} className="animate-pulse" />
              )}
              {generationStatus.message}
            </div>
          )}

          {error && (
            <div className="statusMessage mt3 error">
              <AlertCircle size={18} />
              {error}
            </div>
          )}
        </div>
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
    if (isOpen) {
      setPrompt("");
      setUseGemini(true);
      setPostprocess("voiceover");
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modalOverlay" onClick={() => !isExtending && onClose()}>
      <motion.div
        className="modal modalSmall"
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modalHeader">
          <div className="modalTitle">Extend Animation</div>
          <button 
            className="btn btnGhost btnIcon" 
            onClick={onClose}
            disabled={isExtending}
          >
            <X size={20} />
          </button>
        </div>
        
        <div className="modalBody">
          <div className="formField mb3">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={useGemini}
                onChange={(e) => setUseGemini(e.target.checked)}
                disabled={isExtending}
              />
              <span>Gemini-supervised (auto-generate continuation)</span>
            </label>
          </div>

          <div className="formField mb3">
            <div className="formLabel">
              {useGemini ? "Extension hint (optional)" : "Extension prompt (required)"}
            </div>
            <textarea
              className="textarea"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={
                useGemini
                  ? 'e.g. "zoom out to show the full organ"'
                  : 'e.g. "The camera slowly pulls back..."'
              }
              disabled={isExtending}
            />
            {useGemini && (
              <div className="textSm textTertiary mt2">
                Leave blank for Gemini to decide what happens next.
              </div>
            )}
          </div>

          <div className="formField">
            <div className="formLabel">Post-process</div>
            <select
              className="select"
              value={postprocess}
              onChange={(e) => setPostprocess(e.target.value)}
              disabled={isExtending}
            >
              <option value="off">None</option>
              <option value="captions">Captions</option>
              <option value="voiceover">Voiceover + Captions</option>
            </select>
          </div>

          {error && (
            <div className="statusMessage mt3 error">
              <AlertCircle size={18} />
              {error}
            </div>
          )}

          {success && (
            <div className="statusMessage mt3 success">
              <Check size={18} />
              Extension complete! Check your library.
            </div>
          )}
        </div>

        <div className="modalFooter">
          <button
            className="btn btnSecondary"
            onClick={onClose}
            disabled={isExtending}
          >
            Cancel
          </button>
          <button
            className="btn btnPrimary"
            onClick={() => onSubmit({ prompt, useGemini, postprocess })}
            disabled={isExtending || (!useGemini && !prompt.trim())}
          >
            {isExtending ? (
              <>
                <Loader2 size={18} className="animate-pulse" />
                Extending...
              </>
            ) : (
              <>
                <Zap size={18} />
                Extend Video
              </>
            )}
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
  
  // Player modal
  const [playerItem, setPlayerItem] = useState(null);
  
  // Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationStatus, setGenerationStatus] = useState(null);
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState(null);
  
  // Extend modal
  const [extendModalOpen, setExtendModalOpen] = useState(false);
  const [extendJobId, setExtendJobId] = useState(null);
  const [isExtending, setIsExtending] = useState(false);
  const [extendError, setExtendError] = useState("");
  const [extendSuccess, setExtendSuccess] = useState(false);

  const pollRef = useRef(null);
  const postprocessPollersRef = useRef({});
  const extendPollersRef = useRef({});

  async function refreshLibrary() {
    try {
      const items = await apiFetchJson("/api/library");
      setLibrary(items);
    } catch {
      // Non-fatal
    }
  }

  async function refreshJob(id) {
    try {
      const data = await apiFetchJson(`/api/jobs/${id}`);
      if (data.job.status === "done") {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
        setIsGenerating(false);
        setGenerationStatus({ status: "done", message: "Animation created successfully!" });
        await refreshLibrary();
        setTimeout(() => {
          setActiveTab("library");
          setGenerationStatus(null);
        }, 2000);
      } else if (data.job.status === "error") {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
        setIsGenerating(false);
        setGenerationStatus({ status: "error", message: data.job.error || "Generation failed" });
      } else {
        setGenerationStatus({ status: "running", message: `Generating... (${data.job.status})` });
      }
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
        setError("Job not found. Please try again.");
      }
    }
  }

  async function onGenerate(params) {
    setError("");
    setGenerationStatus({ status: "running", message: "Starting generation..." });
    setIsGenerating(true);

    try {
      const resp = await apiFetchJson("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: params.prompt,
          backend: "veo",
          input_image: params.imageDataUrl,
          prompt_rewrite: params.rewriteMode,
          gemini_model: "gemini-3-flash-preview",
          veo_model: params.veoModel,
          postprocess_mode: params.postprocessMode,
          use_biomedclip: false,
          biomedclip_target: null
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
        } catch {}
      };
      postprocessPollersRef.current[jobId] = setInterval(() => poll(), 2000);
      await poll();
    } catch (e) {
      setPostprocessById((prev) => ({
        ...prev,
        [jobId]: { status: "error", mode, error: String(e?.message || e) }
      }));
    }
  }

  async function onRemoveCaptions(jobId) {
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
    }
  }

  function openExtendModal(jobId) {
    setExtendJobId(jobId);
    setExtendError("");
    setExtendSuccess(false);
    setExtendModalOpen(true);
  }

  async function submitExtend(params) {
    if (!extendJobId) return;
    if (!params.useGemini && !params.prompt.trim()) {
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
          prompt: params.prompt.trim() || null,
          use_gemini: params.useGemini,
          postprocess_mode: params.postprocess
        })
      });

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
              setExtendSuccess(true);
              await refreshLibrary();
              setTimeout(() => {
                setExtendModalOpen(false);
                setPlayerItem(null);
              }, 1500);
            } else {
              setExtendError(data?.job?.error || "Extension failed.");
            }
          }
        } catch {}
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

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebarHeader">
          <div className="logo">
            <div className="logoIcon">
              <Activity />
            </div>
            <span className="logoText">Medimations</span>
          </div>
        </div>
        
        <nav className="sidebarNav">
          <button
            className={`navItem ${activeTab === "library" ? "active" : ""}`}
            onClick={() => setActiveTab("library")}
          >
            <FolderOpen />
            Library
          </button>
          <button
            className={`navItem ${activeTab === "create" ? "active" : ""}`}
            onClick={() => setActiveTab("create")}
          >
            <Plus />
            Create
          </button>
        </nav>
        
        <div className="sidebarFooter">
          <div className="statusPill">
            <div className="statusDot" />
            <span>{backendBase() || "Local"}</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="mainContent">
        <header className="contentHeader">
          <h1 className="contentTitle">
            {activeTab === "library" ? "Library" : "Create Animation"}
          </h1>
          <div className="headerActions">
            {activeTab === "library" && (
              <>
                <button className="btn btnGhost btnSm" onClick={() => setMuted((m) => !m)}>
                  {muted ? <VolumeX size={18} /> : <Volume2 size={18} />}
                  {muted ? "Unmute" : "Mute"}
                </button>
                <button className="btn btnSecondary btnSm" onClick={refreshLibrary}>
                  <RefreshCw size={18} />
                  Refresh
                </button>
                <button className="btn btnPrimary" onClick={() => setActiveTab("create")}>
                  <Plus size={18} />
                  Create
                </button>
              </>
            )}
          </div>
        </header>

        <div className="contentBody">
          <AnimatePresence mode="wait">
            {activeTab === "library" ? (
              <LibraryView
                key="library"
                library={library}
                onRefresh={refreshLibrary}
                onOpenPlayer={setPlayerItem}
                onPostprocess={onPostprocess}
                onRemoveCaptions={onRemoveCaptions}
                postprocessById={postprocessById}
                onCreateNew={() => setActiveTab("create")}
              />
            ) : (
              <CreateView
                key="create"
                onBack={() => setActiveTab("library")}
                onGenerate={onGenerate}
                isGenerating={isGenerating}
                generationStatus={generationStatus}
                error={error}
              />
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Video Player Modal */}
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

      {/* Extend Modal */}
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
