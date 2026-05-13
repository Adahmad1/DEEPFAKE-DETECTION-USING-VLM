import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileVideo,
  Radio,
  ScanLine,
  Server,
  ShieldAlert,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";

type FrameLog = {
  frame_id: number;
  label: "FAKE" | "REAL";
  cnn_conf: number;
  vlm_conf: number;
  fused_conf: number;
  hot_regions: { name: string; score: number }[];
  explanation: string;
};

type AnalysisResult = {
  verdict: "FAKE" | "REAL";
  votes: { FAKE: number; REAL: number };
  faces_processed: number;
  frames_scanned: number;
  report: string;
  frame_logs?: FrameLog[];
  output_video_url?: string;
};

const DEFAULT_API =
  (import.meta.env.VITE_API_URL as string | undefined) ??
  "http://localhost:8000/analyze";

const Index = () => {
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const videoPreviewRef = useRef<HTMLVideoElement | null>(null);

  const [apiUrl, setApiUrl] = useState<string>(() => {
    return localStorage.getItem("deepfake_api_url") ?? DEFAULT_API;
  });
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.title = "Deepfake Forensic Analyzer | VLM + CNN";
    const meta =
      document.querySelector('meta[name="description"]') ??
      (() => {
        const m = document.createElement("meta");
        m.name = "description";
        document.head.appendChild(m);
        return m;
      })();
    meta.setAttribute(
      "content",
      "Forensic deepfake detection: upload a video, run ResNet18 + CLIP fusion with Grad-CAM, get an LLM-written verdict.",
    );
  }, []);

  useEffect(() => {
    localStorage.setItem("deepfake_api_url", apiUrl);
  }, [apiUrl]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const acceptFile = (f: File | null | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("video/")) {
      toast({
        title: "Invalid file",
        description: "Please select a video file (mp4, mov, webm, avi).",
        variant: "destructive",
      });
      return;
    }
    if (f.size > 500 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Maximum file size is 500MB.",
        variant: "destructive",
      });
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    acceptFile(e.dataTransfer.files?.[0]);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setProgress(0);
    setError(null);
    setResult(null);

    // Indeterminate-ish progress while waiting on the backend
    const tick = window.setInterval(() => {
      setProgress((p) => (p < 92 ? p + Math.random() * 4 : p));
    }, 500);

    try {
      const formData = new FormData();
      formData.append("video", file);
      formData.append("filename", file.name);

      const res = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Backend returned ${res.status} ${res.statusText}`);
      }

      const data = (await res.json()) as AnalysisResult;
      setProgress(100);
      setResult(data);
      toast({
        title: `Analysis complete: ${data.verdict}`,
        description: `${data.faces_processed} faces processed across ${data.frames_scanned} frames.`,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      toast({
        title: "Analysis failed",
        description: msg,
        variant: "destructive",
      });
    } finally {
      window.clearInterval(tick);
      setIsAnalyzing(false);
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
  };

  const isFake = result?.verdict === "FAKE";

  return (
    <main className="min-h-screen px-4 py-8 md:px-8 md:py-12">
      <div className="mx-auto max-w-6xl">
        {/* Header */}
        <header className="mb-8 flex flex-col gap-4 border-b border-border pb-6 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-primary">
              <Radio className="h-3 w-3 pulse-dot" />
              <span>SYSTEM ONLINE</span>
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-glow md:text-4xl">
              DEEPFAKE.FORENSIC
            </h1>
            <p className="mt-2 max-w-xl text-sm text-muted-foreground">
              VLM + CNN fusion pipeline · ResNet18 · CLIP · Grad-CAM ·
              LLM-narrated verdict
            </p>
          </div>
          <div className="flex items-center gap-2 rounded border border-border bg-card px-3 py-2 text-xs">
            <Server className="h-3.5 w-3.5 text-primary" />
            <span className="text-muted-foreground">ENDPOINT</span>
            <Input
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="h-6 w-64 border-0 bg-transparent px-1 text-xs focus-visible:ring-0"
              placeholder="http://localhost:8000/analyze"
            />
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
          {/* Upload panel */}
          <section className="flex flex-col gap-4">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-muted-foreground">
              <FileVideo className="h-3.5 w-3.5" />
              <span>// EVIDENCE INTAKE</span>
            </div>

            {!file ? (
              <label
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                className={`group relative flex min-h-[280px] cursor-pointer flex-col items-center justify-center gap-3 rounded border-2 border-dashed bg-card p-8 transition-colors ${
                  isDragging
                    ? "border-primary bg-primary/5 border-glow"
                    : "border-border hover:border-primary/60"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => acceptFile(e.target.files?.[0])}
                />
                <Upload className="h-10 w-10 text-primary" strokeWidth={1.5} />
                <div className="text-center">
                  <p className="text-sm font-medium">
                    Drop video file or click to browse
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    MP4 · MOV · WEBM · AVI · max 500 MB
                  </p>
                </div>
              </label>
            ) : (
              <div className="relative overflow-hidden rounded border border-border bg-card">
                <div className="flex items-center justify-between border-b border-border px-4 py-2 text-xs">
                  <div className="flex items-center gap-2 truncate">
                    <FileVideo className="h-3.5 w-3.5 text-primary" />
                    <span className="truncate font-medium">{file.name}</span>
                    <span className="text-muted-foreground">
                      {(file.size / (1024 * 1024)).toFixed(1)}MB
                    </span>
                  </div>
                  <button
                    onClick={reset}
                    className="rounded p-1 text-muted-foreground hover:bg-secondary hover:text-foreground"
                    aria-label="Remove file"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
                <div
                  className={`relative aspect-video bg-black ${isAnalyzing ? "scan-animate" : ""}`}
                >
                  {previewUrl && (
                    <video
                      ref={videoPreviewRef}
                      src={previewUrl}
                      controls={!isAnalyzing}
                      className="h-full w-full object-contain"
                    />
                  )}
                </div>
              </div>
            )}

            {/* Action bar */}
            <div className="flex flex-col gap-3">
              {isAnalyzing && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-2 text-primary">
                      <Activity className="h-3.5 w-3.5 pulse-dot" />
                      ANALYZING FRAMES
                    </span>
                    <span className="font-mono text-muted-foreground">
                      {Math.floor(progress)}%
                    </span>
                  </div>
                  <Progress value={progress} className="h-1" />
                </div>
              )}

              <div className="flex gap-2">
                <Button
                  onClick={handleAnalyze}
                  disabled={!file || isAnalyzing}
                  className="flex-1 gap-2 bg-primary font-semibold uppercase tracking-wider text-primary-foreground hover:bg-primary/90"
                >
                  <ScanLine className="h-4 w-4" />
                  {isAnalyzing ? "Scanning..." : "Run Analysis"}
                </Button>
                {file && !isAnalyzing && (
                  <Button
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                    className="border-border"
                  >
                    Change
                  </Button>
                )}
              </div>

              {error && (
                <div className="flex items-start gap-2 rounded border border-destructive/40 bg-destructive/10 p-3 text-xs">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive" />
                  <div>
                    <p className="font-medium text-destructive">
                      Connection failed
                    </p>
                    <p className="mt-0.5 text-muted-foreground">{error}</p>
                    <p className="mt-2 text-muted-foreground">
                      Verify the backend is running and CORS allows this origin.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* Results panel */}
          <section className="flex flex-col gap-4">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-muted-foreground">
              <ShieldAlert className="h-3.5 w-3.5" />
              <span>// VERDICT</span>
            </div>

            {!result && !isAnalyzing && (
              <div className="flex min-h-[280px] flex-col items-center justify-center gap-2 rounded border border-dashed border-border bg-card/50 p-8 text-center">
                <ShieldCheck
                  className="h-10 w-10 text-muted-foreground/40"
                  strokeWidth={1.5}
                />
                <p className="text-sm text-muted-foreground">
                  Awaiting evidence...
                </p>
                <p className="text-xs text-muted-foreground/60">
                  Results will appear once analysis completes.
                </p>
              </div>
            )}

            {isAnalyzing && !result && (
              <div className="flex min-h-[280px] flex-col items-center justify-center gap-3 rounded border border-primary/40 bg-card p-8 text-center border-glow">
                <ScanLine
                  className="h-10 w-10 animate-pulse text-primary"
                  strokeWidth={1.5}
                />
                <p className="text-sm font-medium text-primary">
                  Pipeline executing
                </p>
                <p className="font-mono text-xs text-muted-foreground">
                  CNN → CLIP → FUSION → GRAD-CAM → LLM
                </p>
              </div>
            )}

            {result && (
              <>
                <div
                  className={`relative overflow-hidden rounded border p-6 ${
                    isFake
                      ? "border-threat/50 border-glow-threat"
                      : "border-success/50"
                  }`}
                  style={{
                    background: isFake
                      ? "var(--gradient-threat)"
                      : "var(--gradient-safe)",
                  }}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="mb-1 text-xs uppercase tracking-[0.2em] text-muted-foreground">
                        Final Verdict
                      </p>
                      <p
                        className={`text-4xl font-bold tracking-tight ${
                          isFake
                            ? "text-threat text-glow-threat"
                            : "text-success"
                        }`}
                      >
                        {result.verdict}
                      </p>
                    </div>
                    {isFake ? (
                      <ShieldAlert className="h-10 w-10 text-threat" />
                    ) : (
                      <CheckCircle2 className="h-10 w-10 text-success" />
                    )}
                  </div>

                  <div className="mt-6 grid grid-cols-2 gap-3 text-xs">
                    <Stat
                      label="Frames Scanned"
                      value={String(result.frames_scanned)}
                    />
                    <Stat
                      label="Faces Processed"
                      value={String(result.faces_processed)}
                    />
                  </div>
                </div>

                {/* Frame Log */}
                {result.frame_logs && result.frame_logs.length > 0 && (
                  <div className="rounded border border-border bg-card">
                    <p className="border-b border-border px-4 py-2 text-xs uppercase tracking-[0.2em] text-muted-foreground">
                      // Frame log ({result.frame_logs.length})
                    </p>
                    <div className="max-h-72 divide-y divide-border overflow-y-auto">
                      {result.frame_logs.map((log) => (
                        <div
                          key={log.frame_id}
                          className="flex items-start gap-3 px-4 py-2 text-xs"
                        >
                          <span className="font-mono text-muted-foreground">
                            #{String(log.frame_id).padStart(4, "0")}
                          </span>
                          <span
                            className={`shrink-0 rounded px-1.5 py-0.5 font-semibold ${
                              log.label === "FAKE"
                                ? "bg-threat/20 text-threat"
                                : "bg-success/20 text-success"
                            }`}
                          >
                            {log.label}
                          </span>
                          <span className="font-mono text-muted-foreground">
                            {log.fused_conf.toFixed(2)}
                          </span>
                          <span className="flex-1 text-muted-foreground">
                            {log.explanation}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Forensic Report — full width below, horizontal layout */}
                <div className="rounded border border-border bg-card p-4">
                  <p className="mb-3 text-xs uppercase tracking-[0.2em] text-muted-foreground">
                    // Forensic Report
                  </p>
                  {/* key stats in a horizontal row */}
                  <div className="mb-4 flex flex-wrap gap-3">
                    {[
                      { label: "Verdict",  value: result.verdict,                tone: result.verdict === "FAKE" ? "threat" : "success" },
                      { label: "Frames",   value: String(result.frames_scanned)               },
                      { label: "Faces",    value: String(result.faces_processed)              },
                    ].map(({ label, value, tone }) => (
                      <div key={label} className="rounded border border-border/60 bg-background/40 px-3 py-2 text-xs">
                        <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
                        <p className={`mt-1 font-mono font-semibold ${
                          tone === "threat"  ? "text-threat"   :
                          tone === "success" ? "text-success"  : "text-foreground"
                        }`}>{value}</p>
                      </div>
                    ))}
                  </div>
                  {/* report body — wraps naturally full width */}
                  <p className="whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
                    {(result as any).overall_report ?? result.report}
                  </p>
                </div>
              </>
            )}
          </section>
        </div>

        <footer className="mt-12 border-t border-border pt-4 text-center text-xs text-muted-foreground">
          POST multipart/form-data → field <code className="text-primary">video</code> ·
          expects JSON{" "}
          <code className="text-primary">
            {"{ verdict, votes, faces_processed, frames_scanned, report, frame_logs }"}
          </code>
        </footer>
      </div>
    </main>
  );
};

const Stat = ({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "threat" | "success";
}) => (
  <div className="rounded border border-border/60 bg-background/40 px-3 py-2">
    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
      {label}
    </p>
    <p
      className={`mt-1 font-mono text-sm font-semibold ${
        tone === "threat"
          ? "text-threat"
          : tone === "success"
            ? "text-success"
            : "text-foreground"
      }`}
    >
      {value}
    </p>
  </div>
);

export default Index;
