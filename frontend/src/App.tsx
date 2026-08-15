import { ChangeEvent, type CSSProperties, DragEvent, useEffect, useMemo, useRef, useState } from "react";

type Settings = {
  profile: "auto" | "person" | "background" | "lineart" | "background_lineart";
  columns: number;
  detail: number;
  abstraction: number;
  thresholdLow: number;
  thresholdHigh: number;
  minComponent: number;
  maxRows: number;
  cropX: number;
  cropY: number;
  cropWidth: number;
  cropHeight: number;
};

type ConvertResult = {
  ascii: string;
  rows: number;
  columns: number;
  processedPng: string;
  renderedPng: string;
  crop: [number, number, number, number];
  options: {
    columns: number;
    font_size: number;
    detail: number;
    threshold_low: number;
    threshold_high: number;
    min_component: number;
    abstraction: number;
    crop_x: number;
    crop_y: number;
    crop_width: number;
    crop_height: number;
    max_rows: number;
    profile: Settings["profile"];
  };
  pipeline: {
    generator_version: string;
    recipe_version: string;
    enabled: boolean;
    applied: boolean;
    reason?: string;
    stages: Array<Record<string, unknown>>;
  };
};

type SourceDimensions = {
  width: number;
  height: number;
};

type CropPixelBox = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

type ReviewDecision = "accept" | "hold" | "reject";

type CharacterPaletteGroup = {
  id: string;
  label: string;
  characters: string[];
};

const characterPaletteGroups: CharacterPaletteGroup[] = [
  {
    id: "lines",
    label: "基本線",
    characters: ["|", "｜", "l", "ｌ", "/", "／", "\\", "＼", "_", "＿", "-", "‐", "―", "ｰ", "ー", "=", "＝", "~", "～", "￣"],
  },
  {
    id: "joins",
    label: "角・接続",
    characters: ["┌", "┐", "└", "┘", "├", "┤", "┬", "┴", "┼", "┣", "┫", "┳", "┻", "╋", "┏", "┓", "┗", "┛"],
  },
  {
    id: "curves",
    label: "曲線・輪郭",
    characters: ["(", ")", "（", "）", "〈", "〉", "《", "》", "〔", "〕", "{", "}", "｛", "｝", "∧", "∨", "⌒", "へ", "く", "つ", "し", "ノ", "ヽ"],
  },
  {
    id: "tone",
    label: "点・濃淡",
    characters: [".", ",", ":", ";", "'", "\"", "`", "´", "｀", "・", "･", "、", "。", "ﾟ", "゜", "¨", "′", "…", "：", "；"],
  },
  {
    id: "parts",
    label: "AA部品",
    characters: ["ω", "д", "Д", "∀", "益", "皿", "人", "八", "入", "ハ", "ﾊ", "爪", "川", "州", "三", "彡", "ミ", "乂", "メ", "ヾ", "ゝ", "ゞ", "ﾉ"],
  },
  {
    id: "spacing",
    label: "空白・調整",
    characters: [" ", "　", "･", "ｰ", "＿", "￣"],
  },
];

function paletteCharacterLabel(character: string): string {
  if (character === " ") return "半角空白";
  if (character === "　") return "全角空白";
  return character;
}

type ReviewStatus = {
  available: boolean;
  message?: string;
  archiveVersion?: string;
  reviewedCount?: number;
  acceptedCount?: number;
  targetCount?: number;
  decisionCounts?: Record<ReviewDecision, number>;
  lastReviewedEntryId?: number | null;
  categories?: string[];
  issueTags?: string[];
  reviewTargets?: Record<string, number>;
  acceptedByCategory?: Record<string, number>;
  checkpointReady?: boolean;
  pendingCandidateCount?: number;
  sensitivePolicy?: string;
};

type StoredReview = {
  decision: ReviewDecision;
  category: string;
  qualityScore: number | null;
  issueTags: string[];
  comment: string;
  updatedAtUtc: string;
};

type ReviewCandidate = {
  id: number;
  text: string;
  sourcePath: string;
  sourceEncoding: string;
  chunkIndex: number;
  section: string | null;
  category: string;
  sensitive: boolean;
  normalizedSha256: string;
  lineCount: number;
  maxColumns: number;
  characterCount: number;
  artScore: number;
  preview: string;
  review: StoredReview | null;
};

const categoryLabels: Record<string, string> = {
  character: "人物",
  multiple_people: "複数人物",
  face: "顔・表情",
  upper_body: "上半身",
  full_body: "全身",
  background: "背景",
  architecture: "建築",
  nature: "自然",
  object: "小物",
  mecha: "メカ",
  effect: "効果",
  text_or_joke: "文字・ネタ",
  unknown: "未分類",
};

const issueLabels: Record<string, string> = {
  fragment: "断片",
  multiple_aa: "複数AA混在",
  dialog_or_frame: "台詞・枠中心",
  broken_alignment: "位置崩れ",
  too_dense: "過密・黒塊",
  too_sparse: "疎すぎる",
  text_or_logo: "文字・ロゴ",
  category_mismatch: "カテゴリ違い",
};

const defaults: Settings = {
  profile: "auto",
  columns: 72,
  detail: 72,
  abstraction: 35,
  thresholdLow: 55,
  thresholdHigh: 150,
  minComponent: 10,
  maxRows: 58,
  cropX: 0,
  cropY: 0,
  cropWidth: 1,
  cropHeight: 1,
};

const portraitPreset: Partial<Settings> = {
  profile: "person",
  cropX: 0.2,
  cropY: 0.015,
  cropWidth: 0.64,
  cropHeight: 0.61,
  columns: 72,
  maxRows: 55,
  abstraction: 40,
};

const facePreset: Partial<Settings> = {
  profile: "person",
  cropX: 0.27,
  cropY: 0.015,
  cropWidth: 0.42,
  cropHeight: 0.36,
  columns: 72,
  maxRows: 48,
  abstraction: 30,
};

const backgroundPreset: Partial<Settings> = {
  profile: "background",
  cropX: 0,
  cropY: 0,
  cropWidth: 1,
  cropHeight: 1,
  columns: 120,
  detail: 58,
  abstraction: 15,
  thresholdLow: 65,
  thresholdHigh: 170,
  minComponent: 14,
  maxRows: 60,
};

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function normalizedCropBox(settings: Settings, sourceDimensions: SourceDimensions): CropPixelBox {
  const cropX = clamp(settings.cropX, 0, 0.95);
  const cropY = clamp(settings.cropY, 0, 0.95);
  const cropWidth = clamp(settings.cropWidth, 0.05, 1 - cropX);
  const cropHeight = clamp(settings.cropHeight, 0.05, 1 - cropY);
  return {
    left: Math.round(cropX * sourceDimensions.width),
    top: Math.round(cropY * sourceDimensions.height),
    right: Math.round((cropX + cropWidth) * sourceDimensions.width),
    bottom: Math.round((cropY + cropHeight) * sourceDimensions.height),
  };
}

function autoRowCalculation(
  settings: Settings,
  sourceDimensions: SourceDimensions,
): { maxRows: number; naturalRows: number } {
  const crop = normalizedCropBox(settings, sourceDimensions);
  const cropWidth = Math.max(1, crop.right - crop.left);
  const cropHeight = Math.max(1, crop.bottom - crop.top);
  const targetWidth = settings.columns * 8;
  const naturalHeight = Math.round(cropHeight * targetWidth / cropWidth);
  const naturalRows = Math.max(1, Math.round(naturalHeight / 18));
  return {
    maxRows: clamp(naturalRows, 12, 100),
    naturalRows,
  };
}

function conversionForm(file: File, settings: Settings, cumulative?: boolean): FormData {
  const body = new FormData();
  body.append("image", file);
  body.append("profile", settings.profile);
  body.append("columns", String(settings.columns));
  body.append("detail", String(settings.detail));
  body.append("abstraction", String(settings.abstraction));
  body.append("threshold_low", String(settings.thresholdLow));
  body.append("threshold_high", String(settings.thresholdHigh));
  body.append("min_component", String(settings.minComponent));
  body.append("max_rows", String(settings.maxRows));
  body.append("crop_x", String(settings.cropX));
  body.append("crop_y", String(settings.cropY));
  body.append("crop_width", String(settings.cropWidth));
  body.append("crop_height", String(settings.cropHeight));
  if (cumulative !== undefined) body.append("cumulative", String(cumulative));
  return body;
}

function decimalPlaces(value: number): number {
  if (Number.isInteger(value)) return 0;
  const text = String(value);
  return text.includes("e-")
    ? Number(text.split("e-")[1])
    : (text.split(".")[1]?.length ?? 0);
}

function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  unit = "",
  disabled = false,
  directInput = false,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  disabled?: boolean;
  directInput?: boolean;
  onChange: (value: number) => void;
}) {
  const places = decimalPlaces(step);
  const formattedValue = value.toFixed(places);
  const [draftValue, setDraftValue] = useState(formattedValue);
  useEffect(() => setDraftValue(formattedValue), [formattedValue]);

  function commitDirectInput() {
    const parsed = Number(draftValue);
    if (!Number.isFinite(parsed)) {
      setDraftValue(formattedValue);
      return;
    }
    const next = clamp(parsed, min, max);
    onChange(next);
    setDraftValue(next.toFixed(places));
  }

  return (
    <label className="control">
      <span>
        {label}
        {directInput ? (
          <span className="control-value-editor">
            <input
              className="control-number"
              type="text"
              inputMode="decimal"
              aria-label={`${label}の数値`}
              value={draftValue}
              disabled={disabled}
              onChange={(event) => setDraftValue(event.target.value)}
              onBlur={commitDirectInput}
              onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
                if (event.key === "Escape") {
                  setDraftValue(formattedValue);
                  event.currentTarget.blur();
                }
              }}
            />
            {unit && <span>{unit}</span>}
          </span>
        ) : <output>{formattedValue}{unit}</output>}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function ReviewWorkspace() {
  const [status, setStatus] = useState<ReviewStatus | null>(null);
  const [candidate, setCandidate] = useState<ReviewCandidate | null>(null);
  const [decision, setDecision] = useState<ReviewDecision | null>(null);
  const [category, setCategory] = useState("unknown");
  const [qualityScore, setQualityScore] = useState<number | null>(null);
  const [issueTags, setIssueTags] = useState<string[]>([]);
  const [comment, setComment] = useState("");
  const [fontSize, setFontSize] = useState(16);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [lastEntryId, setLastEntryId] = useState<number | null>(null);

  function applyCandidate(next: ReviewCandidate) {
    setCandidate(next);
    setDecision(next.review?.decision ?? null);
    setCategory(next.review?.category ?? next.category);
    setQualityScore(next.review?.qualityScore ?? null);
    setIssueTags(next.review?.issueTags ?? []);
    setComment(next.review?.comment ?? "");
    setError("");
  }

  async function readJson<T>(response: Response): Promise<T> {
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail ?? "通信に失敗しました。");
    return payload as T;
  }

  async function loadStatus() {
    const response = await fetch("/api/review/status");
    const payload = await readJson<ReviewStatus>(response);
    setStatus(payload);
    setLastEntryId(payload.lastReviewedEntryId ?? null);
    return payload;
  }

  async function loadNext(skipCurrent = false) {
    setBusy(true);
    setError("");
    setNotice("");
    const query = new URLSearchParams();
    if (skipCurrent && candidate) query.set("skip_entry_id", String(candidate.id));
    try {
      const response = await fetch(`/api/review/next?${query.toString()}`);
      if (response.status === 404) {
        setCandidate(null);
        setNotice("この条件に合う未評価AAはありません。");
        return;
      }
      applyCandidate(await readJson<ReviewCandidate>(response));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "候補の取得に失敗しました。");
    } finally {
      setBusy(false);
    }
  }

  async function initializeReview() {
    setBusy(true);
    try {
      const nextStatus = await loadStatus();
      if (nextStatus.available) await loadNext();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "レビューを開始できません。");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void initializeReview();
  }, []);

  async function saveReview() {
    if (!candidate || !decision) {
      setError("採否を選択してください。");
      return;
    }
    if (decision === "accept" && category === "unknown") {
      setError("採用するAAにはカテゴリを設定してください。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/review/entries/${candidate.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          decision,
          category,
          quality_score: qualityScore,
          issue_tags: issueTags,
          comment,
        }),
      });
      const payload = await readJson<{ candidate: ReviewCandidate; status: ReviewStatus }>(response);
      setLastEntryId(candidate.id);
      setStatus(payload.status);
      if (payload.status.checkpointReady) {
        setCandidate(null);
        setNotice(`${payload.status.targetCount ?? 750} acceptedに到達しました。checkpoint監査を行います。`);
        return;
      }
      await loadNext();
      setNotice(`ID ${candidate.id} の評価を保存しました。`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "評価の保存に失敗しました。");
    } finally {
      setBusy(false);
    }
  }

  async function loadEntry(entryId: number) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/review/entries/${entryId}`);
      applyCandidate(await readJson<ReviewCandidate>(response));
      setNotice("直前の評価を開きました。保存すると上書きされます。");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "評価済みAAを開けませんでした。");
    } finally {
      setBusy(false);
    }
  }

  function toggleIssue(tag: string) {
    setIssueTags((current) => current.includes(tag)
      ? current.filter((item) => item !== tag)
      : [...current, tag]);
  }

  const reviewed = status?.reviewedCount ?? 0;
  const accepted = status?.acceptedCount ?? status?.decisionCounts?.accept ?? 0;
  const target = status?.targetCount ?? 100;
  const progress = Math.min(100, Math.round(accepted / Math.max(1, target) * 100));

  return (
    <section className="review-workspace">
      <section className="panel review-toolbar">
        <div className="review-progress-copy">
          <p className="eyebrow">CURATION QUEUE / {status?.archiveVersion ?? "---"}</p>
          <h2>accepted-v2 品質レビュー</h2>
          <p>旧レビューとは独立に、完成AAとしての構造と教師データ適性を一から判断します。</p>
        </div>
        <div className="progress-block">
          <div><strong>{accepted}</strong><span> / {target} accepted</span></div>
          <div className="progress-track"><i style={{ width: `${progress}%` }} /></div>
          <small>
            レビュー {reviewed}　保留 {status?.decisionCounts?.hold ?? 0}　不採用 {status?.decisionCounts?.reject ?? 0}
          </small>
        </div>
        <div className="review-filters">
          <button type="button" onClick={() => void loadNext(true)} disabled={busy || !candidate}>別の候補</button>
          <button type="button" onClick={() => lastEntryId !== null && void loadEntry(lastEntryId)} disabled={busy || lastEntryId === null}>直前を修正</button>
        </div>
      </section>

      {status && !status.available ? (
        <section className="panel review-unavailable">
          <h2>レビュー用データがありません</h2>
          <p>{status.message}</p>
          <code>python -m training.import_yaruyomi index</code>
        </section>
      ) : (
        <div className="review-grid">
          <section className="panel review-preview-panel">
            <div className="panel-heading">
              <span>AA</span>
              <h2>{candidate ? `候補 #${candidate.id}` : "候補を取得中"}</h2>
              {candidate?.sensitive && <b className="sensitive-chip">SENSITIVE</b>}
              <label className="font-size-control">表示 <input type="range" min="12" max="24" value={fontSize} onChange={(event) => setFontSize(Number(event.target.value))} /> {fontSize}px</label>
            </div>
            {candidate ? (
              <>
                <div className="review-source">
                  <strong>{candidate.sourcePath}</strong>
                  <span>{candidate.section || "見出しなし"}</span>
                  <small>{candidate.maxColumns}列 × {candidate.lineCount}行 / {candidate.sourceEncoding}</small>
                </div>
                <div className="aa-review-stage">
                  <pre style={{ fontSize: `${fontSize}px`, lineHeight: `${Math.round(fontSize * 1.125)}px` }}>{candidate.text}</pre>
                </div>
              </>
            ) : (
              <div className="review-empty">{busy ? "候補を読み込んでいます…" : notice || "候補がありません。"}</div>
            )}
          </section>

          <aside className="panel review-form-panel">
            <div className="panel-heading"><span>01</span><h2>採否</h2></div>
            <div className="decision-grid">
              {(["accept", "hold", "reject"] as ReviewDecision[]).map((item) => (
                <button key={item} type="button" className={`${item} ${decision === item ? "active" : ""}`} onClick={() => setDecision(item)}>
                  {{ accept: "採用", hold: "保留", reject: "不採用" }[item]}
                </button>
              ))}
            </div>

            <div className="review-field">
              <div className="panel-heading"><span>02</span><h2>完成度（任意）</h2></div>
              <p>必要な場合だけ0～5で補助評価します。</p>
              <div className="score-grid">
                {[0, 1, 2, 3, 4, 5].map((score) => <button key={score} type="button" className={qualityScore === score ? "active" : ""} onClick={() => setQualityScore(score)}>{score}</button>)}
              </div>
              <div className="score-legend"><span>崩れている</span><span>完成度が高い</span></div>
            </div>

            <label className="review-field">
              <span className="field-label">正しいカテゴリ</span>
              <select value={category} onChange={(event) => setCategory(event.target.value)}>
                {(status?.categories ?? []).map((item) => <option key={item} value={item}>{categoryLabels[item] ?? item}</option>)}
              </select>
              {candidate && candidate.category !== category && <small>自動分類: {categoryLabels[candidate.category] ?? candidate.category}</small>}
            </label>

            <div className="review-field">
              <span className="field-label">問題点（複数可）</span>
              <div className="issue-grid">
                {(status?.issueTags ?? []).map((tag) => <button key={tag} type="button" className={issueTags.includes(tag) ? "active" : ""} onClick={() => toggleIssue(tag)}>{issueLabels[tag] ?? tag}</button>)}
              </div>
            </div>

            <label className="review-field">
              <span className="field-label">メモ（任意）</span>
              <textarea value={comment} maxLength={2000} onChange={(event) => setComment(event.target.value)} placeholder="判断理由や修正点" />
            </label>

            <button className="primary review-save" type="button" onClick={() => void saveReview()} disabled={!candidate || busy}>
              {busy ? "処理中…" : "保存して次へ"}
            </button>
            {error && <p className="error">{error}</p>}
            {notice && !error && <p className="review-notice">{notice}</p>}
          </aside>
        </div>
      )}
    </section>
  );
}

function App() {
  const [mode, setMode] = useState<"convert" | "review">("convert");
  const [file, setFile] = useState<File | null>(null);
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceDimensions, setSourceDimensions] = useState<SourceDimensions | null>(null);
  const [settings, setSettings] = useState<Settings>(defaults);
  const [cumulative, setCumulative] = useState(true);
  const [autoRows, setAutoRows] = useState(true);
  const [result, setResult] = useState<ConvertResult | null>(null);
  const [ascii, setAscii] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [cp932Copied, setCp932Copied] = useState(false);
  const [savingPair, setSavingPair] = useState(false);
  const [pairNotice, setPairNotice] = useState("");
  const [paletteGroupId, setPaletteGroupId] = useState("used");
  const [traceSourceUrl, setTraceSourceUrl] = useState("");
  const [traceVisible, setTraceVisible] = useState(true);
  const [traceTransparency, setTraceTransparency] = useState(65);
  const editorRef = useRef<HTMLTextAreaElement>(null);

  const sourceName = useMemo(() => file?.name ?? "画像未選択", [file]);
  const cropPreview = useMemo(() => {
    if (!sourceDimensions) return null;
    return normalizedCropBox(settings, sourceDimensions);
  }, [settings.cropHeight, settings.cropWidth, settings.cropX, settings.cropY, sourceDimensions]);
  const calculatedRows = useMemo(() => {
    if (!sourceDimensions) return null;
    return autoRowCalculation(settings, sourceDimensions);
  }, [settings.columns, settings.cropHeight, settings.cropWidth, settings.cropX, settings.cropY, sourceDimensions]);
  const usedCharacters = useMemo(() => {
    const counts = new Map<string, number>();
    for (const character of ascii) {
      if (character === "\r" || character === "\n" || character === "\t") continue;
      counts.set(character, (counts.get(character) ?? 0) + 1);
    }
    return [...counts]
      .sort(([leftCharacter, leftCount], [rightCharacter, rightCount]) =>
        rightCount - leftCount || leftCharacter.localeCompare(rightCharacter, "ja"),
      )
      .slice(0, 48)
      .map(([character]) => character);
  }, [ascii]);
  const activePaletteCharacters = paletteGroupId === "used"
    ? usedCharacters
    : characterPaletteGroups.find((group) => group.id === paletteGroupId)?.characters ?? [];
  const editorTraceStyle = useMemo<CSSProperties>(() => {
    if (!result || !traceSourceUrl || !traceVisible) return {};
    const editorScale = 12 / result.options.font_size;
    return {
      backgroundAttachment: "local",
      backgroundColor: "#f2f1ec",
      backgroundImage: `linear-gradient(rgba(242, 241, 236, ${traceTransparency / 100}), rgba(242, 241, 236, ${traceTransparency / 100})), url("${traceSourceUrl}")`,
      backgroundPosition: "18px 18px",
      backgroundRepeat: "no-repeat",
      backgroundSize: `${result.columns * 8 * editorScale}px ${result.rows * 18 * editorScale}px`,
    };
  }, [result, traceSourceUrl, traceTransparency, traceVisible]);

  useEffect(() => {
    return () => {
      if (sourceUrl.startsWith("blob:")) URL.revokeObjectURL(sourceUrl);
    };
  }, [sourceUrl]);

  useEffect(() => {
    if (!sourceUrl) {
      setSourceDimensions(null);
      return;
    }
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (!cancelled) {
        setSourceDimensions({ width: image.naturalWidth, height: image.naturalHeight });
      }
    };
    image.onerror = () => {
      if (!cancelled) setSourceDimensions(null);
    };
    image.src = sourceUrl;
    return () => {
      cancelled = true;
    };
  }, [sourceUrl]);

  useEffect(() => {
    if (!autoRows || !calculatedRows) return;
    setSettings((current) => current.maxRows === calculatedRows.maxRows
      ? current
      : { ...current, maxRows: calculatedRows.maxRows });
  }, [autoRows, calculatedRows]);

  useEffect(() => {
    if (!sourceUrl || !result) {
      setTraceSourceUrl("");
      return;
    }
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      const [left, top, right, bottom] = result.crop;
      const cropWidth = Math.max(1, right - left);
      const cropHeight = Math.max(1, bottom - top);
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, result.columns * 8);
      canvas.height = Math.max(18, result.rows * 18);
      const context = canvas.getContext("2d");
      if (!context) return;
      context.fillStyle = "#fff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(
        image,
        left,
        top,
        cropWidth,
        cropHeight,
        0,
        0,
        canvas.width,
        canvas.height,
      );
      if (!cancelled) setTraceSourceUrl(canvas.toDataURL("image/png"));
    };
    image.onerror = () => {
      if (!cancelled) setTraceSourceUrl("");
    };
    image.src = sourceUrl;
    return () => {
      cancelled = true;
    };
  }, [result, sourceUrl]);

  function selectFile(next: File) {
    if (!next.type.startsWith("image/")) {
      setError("画像ファイルを選択してください。");
      return;
    }
    setFile(next);
    setSourceUrl(URL.createObjectURL(next));
    setSourceDimensions(null);
    setResult(null);
    setAscii("");
    setTraceSourceUrl("");
    setPairNotice("");
    setError("");
  }

  async function runConversion() {
    if (!file) {
      setError("最初に画像を選択してください。");
      return;
    }
    setBusy(true);
    setError("");
    setCopied(false);
    setCp932Copied(false);
    setPairNotice("");
    try {
      const response = await fetch("/api/convert", { method: "POST", body: conversionForm(file, settings, cumulative) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "変換に失敗しました。");
      setResult(payload as ConvertResult);
      setAscii(payload.ascii);
      setTraceVisible(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "変換に失敗しました。");
    } finally {
      setBusy(false);
    }
  }

  function update<K extends keyof Settings>(key: K, value: Settings[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function toggleAutoRows() {
    setAutoRows((current) => !current);
  }

  function setCrop(preset: "full" | "portrait" | "face" | "background") {
    setSettings((current) => ({
      ...current,
      ...(preset === "full"
        ? { cropX: 0, cropY: 0, cropWidth: 1, cropHeight: 1 }
        : preset === "face"
          ? facePreset
          : preset === "background"
            ? backgroundPreset
            : portraitPreset),
    }));
  }

  function setProfile(profile: Settings["profile"]) {
    const abstraction = {
      auto: 30,
      person: 40,
      background: 15,
      lineart: 35,
      background_lineart: 35,
    }[profile];
    setSettings((current) => ({ ...current, profile, abstraction }));
  }

  async function copyAscii() {
    await navigator.clipboard.writeText(ascii);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  async function copyCp932Ascii() {
    setError("");
    try {
      const response = await fetch("/api/text/cp932-ncr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: ascii }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "CP932用テキストの生成に失敗しました。");
      await navigator.clipboard.writeText(payload.text);
      setCp932Copied(true);
      window.setTimeout(() => setCp932Copied(false), 1600);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "CP932用テキストの生成に失敗しました。");
    }
  }

  function downloadAscii() {
    const url = URL.createObjectURL(new Blob([ascii], { type: "text/plain;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${file?.name.replace(/\.[^.]+$/, "") ?? "yaruo-aa"}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function insertPaletteCharacter(character: string) {
    const editor = editorRef.current;
    const start = editor?.selectionStart ?? ascii.length;
    const end = editor?.selectionEnd ?? start;
    const nextAscii = `${ascii.slice(0, start)}${character}${ascii.slice(end)}`;
    setAscii(nextAscii);
    setPairNotice("");
    window.requestAnimationFrame(() => {
      const nextCursor = start + character.length;
      editor?.focus();
      editor?.setSelectionRange(nextCursor, nextCursor);
    });
  }

  async function saveCorrectionPair() {
    if (!file || !result) {
      setError("先に画像からAAを生成してください。");
      return;
    }
    if (ascii === result.ascii) {
      setError("AAを修正してから対応対を保存してください。");
      return;
    }
    setSavingPair(true);
    setPairNotice("");
    setError("");
    const body = new FormData();
    body.append("source", file, file.name);
    body.append("options_json", JSON.stringify(result.options));
    body.append("crop_json", JSON.stringify(result.crop));
    body.append("draft_text", result.ascii);
    body.append("corrected_text", ascii);
    body.append("processed_png", result.processedPng);
    body.append("draft_rendered_png", result.renderedPng);
    body.append("generation_json", JSON.stringify(result.pipeline));
    body.append("source_origin", "local-user-provided");
    body.append(
      "rights_status",
      "user-provided; local use only; redistribution not granted",
    );
    try {
      const response = await fetch("/api/correction-pairs", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "修正対応対の保存に失敗しました。");
      const recordId = payload.record?.manifest?.record_id ?? "unknown";
      setPairNotice(`修正対応対を保存しました: ${recordId}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "修正対応対の保存に失敗しました。");
    } finally {
      setSavingPair(false);
    }
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const next = event.dataTransfer.files[0];
    if (next) selectFile(next);
  }

  return (
    <main>
      <header className="masthead">
        <div>
          <p className="eyebrow">STRUCTURE-BASED SHIFT-JIS ART</p>
          <h1>Yaruo AA Studio</h1>
          <p className="lede">輪郭を読み、Saitamaarの字形で組み直す半自動AA制作環境。</p>
        </div>
        <div className="masthead-actions">
          <nav className="mode-switch" aria-label="画面切り替え">
            <button type="button" className={mode === "convert" ? "active" : ""} onClick={() => setMode("convert")}>AA生成</button>
            <button type="button" className={mode === "review" ? "active" : ""} onClick={() => setMode("review")}>教師データレビュー</button>
          </nav>
          <div className="status-badge"><i /> {mode === "convert" ? "DeepAA light v0.2" : "LOCAL CURATION"}</div>
        </div>
      </header>

      {mode === "review" ? <ReviewWorkspace /> : <section className="workspace">
        <aside className="sidebar panel">
          <div className="panel-heading">
            <span>01</span>
            <h2>入力と設定</h2>
          </div>

          <div
            className={`dropzone ${file ? "has-file" : ""}`}
            onDragOver={(event) => event.preventDefault()}
            onDrop={onDrop}
          >
            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={(event: ChangeEvent<HTMLInputElement>) => {
                const next = event.target.files?.[0];
                if (next) selectFile(next);
              }}
            />
            <label htmlFor="file-input">
              <b>{file ? sourceName : "画像をドロップ"}</b>
              <span>{file ? "クリックして変更" : "またはクリックして選択"}</span>
            </label>
          </div>
          <div className="control-group">
            <div className="group-title"><h3>構図</h3><span>CROP</span></div>
            <div className="segmented">
              <button type="button" onClick={() => setCrop("full")}>全体</button>
              <button type="button" onClick={() => setCrop("portrait")}>顔・上半身</button>
              <button type="button" onClick={() => setCrop("face")}>顔</button>
              <button type="button" onClick={() => setCrop("background")}>背景</button>
            </div>
            <div className="crop-grid">
              <Slider label="左" value={settings.cropX} min={0} max={0.9} step={0.005} directInput onChange={(v) => update("cropX", v)} />
              <Slider label="上" value={settings.cropY} min={0} max={0.9} step={0.005} directInput onChange={(v) => update("cropY", v)} />
              <Slider label="幅" value={settings.cropWidth} min={0.1} max={1} step={0.005} directInput onChange={(v) => update("cropWidth", v)} />
              <Slider label="高さ" value={settings.cropHeight} min={0.1} max={1} step={0.005} directInput onChange={(v) => update("cropHeight", v)} />
            </div>
          </div>

          <div className="control-group">
            <div className="group-title"><h3>抽出</h3><span>EDGE</span></div>
            <div className="segmented">
              <button type="button" className={settings.profile === "auto" ? "active" : ""} onClick={() => setProfile("auto")}>自動</button>
              <button type="button" className={settings.profile === "person" ? "active" : ""} onClick={() => setProfile("person")}>人物</button>
              <button type="button" className={settings.profile === "background" ? "active" : ""} onClick={() => setProfile("background")}>背景</button>
              <button type="button" className={settings.profile === "lineart" ? "active" : ""} onClick={() => setProfile("lineart")}>線画</button>
              <button type="button" className={settings.profile === "background_lineart" ? "active" : ""} onClick={() => setProfile("background_lineart")}>背景線画</button>
            </div>
            <Slider label="横幅" value={settings.columns} min={32} max={120} unit="字" onChange={(v) => update("columns", v)} />
            <div className="auto-row-control">
              <button
                type="button"
                className={autoRows ? "active" : ""}
                aria-pressed={autoRows}
                onClick={toggleAutoRows}
              >行数自動 {autoRows ? "ON" : "OFF"}</button>
              <small>{autoRows
                ? calculatedRows
                  ? calculatedRows.naturalRows > 100
                    ? `縦横比では${calculatedRows.naturalRows}行（上限100行）`
                    : calculatedRows.naturalRows < 12
                      ? `縦横比では${calculatedRows.naturalRows}行（設定下限12行）`
                      : `切り抜き比率から${calculatedRows.naturalRows}行`
                  : "画像を選ぶと切り抜き比率から計算"
                : "最大行数を手動で指定"}</small>
            </div>
            <Slider
              label="最大行数"
              value={settings.maxRows}
              min={12}
              max={100}
              unit="行"
              disabled={autoRows}
              onChange={(v) => update("maxRows", v)}
            />
            <Slider label="細部" value={settings.detail} min={0} max={100} onChange={(v) => update("detail", v)} />
            <Slider label="デフォルメ" value={settings.abstraction} min={0} max={100} onChange={(v) => update("abstraction", v)} />
            <Slider label="輪郭下限" value={settings.thresholdLow} min={0} max={200} onChange={(v) => update("thresholdLow", v)} />
            <Slider label="輪郭上限" value={settings.thresholdHigh} min={40} max={255} onChange={(v) => update("thresholdHigh", v)} />
            <Slider label="ノイズ除去" value={settings.minComponent} min={0} max={80} onChange={(v) => update("minComponent", v)} />
          </div>

          <div className="generate-actions">
            <button
              className={cumulative ? "experiment active" : "experiment"}
              type="button"
              aria-pressed={cumulative}
              onClick={() => setCumulative((current) => !current)}
            >累積改善 {cumulative ? "ON" : "OFF"}</button>
            <button className="primary" type="button" onClick={runConversion} disabled={!file || busy}>
              {busy ? <><span className="spinner" /> 解析中</> : "AAを1案生成"}
            </button>
            <small className="experiment-note">既定ON。線回収、面回収、塗り、情報回収、T3を順に適用し、失敗した段階だけ戻します。</small>
          </div>
          {error && <p className="error">{error}</p>}
        </aside>

        <div className="results">
          <section className="panel visual-panel">
            <div className="panel-heading">
              <span>02</span>
              <h2>構造の比較</h2>
              {result && <small>{result.columns}字 × {result.rows}行</small>}
            </div>
            <div className="comparison">
              <figure>
                <figcaption>ORIGINAL</figcaption>
                <div className="image-stage">
                  {sourceUrl && sourceDimensions && cropPreview ? (
                    <svg
                      className="original-crop-preview"
                      viewBox={`0 0 ${sourceDimensions.width} ${sourceDimensions.height}`}
                      preserveAspectRatio="xMidYMid meet"
                      role="img"
                      aria-label="入力画像と現在のクロップ範囲"
                    >
                      <image
                        href={sourceUrl}
                        x="0"
                        y="0"
                        width={sourceDimensions.width}
                        height={sourceDimensions.height}
                      />
                      <g className="crop-outside" aria-hidden="true">
                        <rect x="0" y="0" width={sourceDimensions.width} height={cropPreview.top} />
                        <rect x="0" y={cropPreview.top} width={cropPreview.left} height={cropPreview.bottom - cropPreview.top} />
                        <rect x={cropPreview.right} y={cropPreview.top} width={sourceDimensions.width - cropPreview.right} height={cropPreview.bottom - cropPreview.top} />
                        <rect x="0" y={cropPreview.bottom} width={sourceDimensions.width} height={sourceDimensions.height - cropPreview.bottom} />
                      </g>
                      <rect
                        className="crop-frame-shadow"
                        x={cropPreview.left}
                        y={cropPreview.top}
                        width={cropPreview.right - cropPreview.left}
                        height={cropPreview.bottom - cropPreview.top}
                      />
                      <rect
                        className="crop-frame"
                        x={cropPreview.left}
                        y={cropPreview.top}
                        width={cropPreview.right - cropPreview.left}
                        height={cropPreview.bottom - cropPreview.top}
                      />
                    </svg>
                  ) : sourceUrl ? <img src={sourceUrl} alt="入力画像" /> : <div className="empty">INPUT</div>}
                </div>
              </figure>
              <figure>
                <figcaption>EXTRACTED LINES</figcaption>
                <div className="image-stage checker">
                  {result ? <img src={result.processedPng} alt="抽出した輪郭" /> : <div className="empty">EDGE</div>}
                </div>
              </figure>
              <figure>
                <figcaption>SAITAMAAR RENDER</figcaption>
                <div className="image-stage checker">
                  {result ? <img src={result.renderedPng} alt="AAレンダリング" /> : <div className="empty">AA</div>}
                </div>
              </figure>
            </div>
          </section>

          <section className="panel editor-panel">
            <div className="panel-heading">
              <span>03</span>
              <h2>AAエディタ</h2>
              <div className="editor-actions">
                <button type="button" onClick={() => void saveCorrectionPair()} disabled={!file || !result || ascii === result.ascii || savingPair}>{savingPair ? "保存中…" : "修正対を保存"}</button>
                <button type="button" onClick={copyAscii} disabled={!ascii}>{copied ? "コピー済み" : "Unicodeコピー"}</button>
                <button type="button" onClick={copyCp932Ascii} disabled={!ascii}>{cp932Copied ? "コピー済み" : "CP932/NCRコピー"}</button>
                <button type="button" onClick={downloadAscii} disabled={!ascii}>TXT保存</button>
              </div>
            </div>
            <div className="editor-trace-controls" aria-label="元絵トレース設定">
              <button
                type="button"
                className={traceVisible ? "active" : ""}
                aria-pressed={traceVisible}
                disabled={!result || !traceSourceUrl}
                onClick={() => setTraceVisible((visible) => !visible)}
              >元絵トレース {traceVisible ? "ON" : "OFF"}</button>
              <label>
                <span>透過度</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={traceTransparency}
                  aria-label="元絵の透過度"
                  disabled={!result || !traceSourceUrl || !traceVisible}
                  onChange={(event) => setTraceTransparency(Number(event.target.value))}
                />
                <output aria-label={`元絵の透過度 ${traceTransparency}%`}>{traceTransparency}%</output>
              </label>
              <small>0%で不透明、100%で非表示</small>
            </div>
            <div className="character-palette" aria-label="候補文字パレット">
              <div className="character-palette-heading">
                <div>
                  <b>候補文字パレット</b>
                  <span>クリックでカーソル位置へ挿入、選択範囲があれば置換</span>
                </div>
                <div className="character-palette-tabs" role="tablist" aria-label="候補文字カテゴリ">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={paletteGroupId === "used"}
                    className={paletteGroupId === "used" ? "active" : ""}
                    onClick={() => setPaletteGroupId("used")}
                  >使用中</button>
                  {characterPaletteGroups.map((group) => (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={paletteGroupId === group.id}
                      className={paletteGroupId === group.id ? "active" : ""}
                      key={group.id}
                      onClick={() => setPaletteGroupId(group.id)}
                    >{group.label}</button>
                  ))}
                </div>
              </div>
              <div className="character-palette-grid" role="tabpanel">
                {activePaletteCharacters.length > 0 ? activePaletteCharacters.map((character, index) => (
                  <button
                    type="button"
                    className={character === " " || character === "　" ? "space-character" : ""}
                    key={`${paletteGroupId}-${character}-${index}`}
                    title={`${paletteCharacterLabel(character)}を挿入`}
                    aria-label={`${paletteCharacterLabel(character)}を挿入`}
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => insertPaletteCharacter(character)}
                  >{paletteCharacterLabel(character)}</button>
                )) : <span className="character-palette-empty">AAを生成すると使用文字が表示されます。</span>}
              </div>
            </div>
            <textarea
              ref={editorRef}
              aria-label="ASCIIアート編集欄"
              value={ascii}
              onChange={(event) => {
                setAscii(event.target.value);
                setPairNotice("");
              }}
              style={editorTraceStyle}
              spellCheck={false}
              placeholder="生成されたAAはここで直接修正できます。"
            />
            {pairNotice && <p className="pair-notice">{pairNotice}</p>}
          </section>
        </div>
      </section>}
    </main>
  );
}

export default App;
