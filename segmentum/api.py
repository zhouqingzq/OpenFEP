"""Segmentum Personality Analysis Web UI + API.

Provides a browser-based interface and JSON API for personality analysis.
Requires the ``api`` optional dependency group::

    pip install segmentum[api]
"""

from __future__ import annotations

import json
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, PlainTextResponse
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for the API layer.  "
        "Install with: pip install segmentum[api]"
    ) from exc

from .personality_analyzer import PersonalityAnalyzer

app = FastAPI(
    title="Segmentum Personality Analyzer",
    version="0.1.0",
    description=(
        "Inverse personality inference based on Free Energy Principle / "
        "Active Inference framework.  Accepts text materials and produces "
        "a structured personality generative model."
    ),
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AnalysisRequest(BaseModel):
    """Input for personality analysis endpoints."""
    materials: list[str] = Field(
        ..., description="Text segments to analyze (conversations, diary entries, etc.)",
        min_length=1,
    )
    material_types: list[str] = Field(
        default_factory=list,
        description='Optional labels: "conversation", "diary", "behavior", "biography"',
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the analysis",
    )
    llm_enhanced: bool = Field(
        default=False,
        description="Use LLM for deeper semantic extraction (requires llm_config)",
    )
    llm_config: dict[str, Any] | None = Field(
        default=None,
        description="LLM API config: api_key, model, base_url, timeout_seconds",
    )


class SimulationRequest(BaseModel):
    """Input for forward simulation from inferred personality."""
    personality: dict[str, Any] = Field(
        ..., description="Inferred personality parameters (from /analyze)",
    )
    scenario: str = Field(
        ..., description="Situation description for simulation",
    )
    cycles: int = Field(
        default=50, ge=1, le=500,
        description="Number of simulation cycles",
    )


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

_HTML_PAGE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Segmentum Personality Analyzer</title>
<style>
:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #232733;
  --border: #2e3348;
  --text: #e0e0e8;
  --text2: #9498b0;
  --accent: #6c8cff;
  --accent2: #4a6adf;
  --green: #4caf82;
  --orange: #e8a44a;
  --red: #e05555;
  --radius: 8px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
  background: var(--bg); color: var(--text);
  line-height: 1.6; min-height: 100vh;
}
.container { max-width: 1100px; margin: 0 auto; padding: 24px 20px; }
.top-bar { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; }
.top-bar .left { flex: 1; }
h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 4px; }
h1 span { color: var(--accent); }
.subtitle { color: var(--text2); font-size: 0.85rem; }
.lang-toggle {
  padding: 5px 14px; border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface2); color: var(--text2); cursor: pointer;
  font-size: 0.82rem; font-weight: 500; white-space: nowrap;
  transition: background 0.15s, border-color 0.15s; flex-shrink: 0; margin-top: 2px;
}
.lang-toggle:hover { background: var(--border); border-color: var(--accent); color: var(--text); }

/* Input area */
.input-panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 20px; margin-bottom: 20px;
}
.input-panel label { font-size: 0.85rem; color: var(--text2); display: block; margin-bottom: 8px; }
.materials-box {
  width: 100%; min-height: 180px; max-height: 400px;
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 6px; padding: 12px 14px;
  color: var(--text); font-size: 0.9rem; font-family: inherit;
  line-height: 1.6; resize: vertical;
}
.materials-box:focus { outline: none; border-color: var(--accent); }
.materials-box::placeholder { color: #555872; }
.hint { font-size: 0.75rem; color: var(--text2); margin-top: 6px; }
.btn-row { display: flex; gap: 10px; margin-top: 14px; align-items: center; }
.btn {
  padding: 9px 22px; border: none; border-radius: 6px;
  font-size: 0.88rem; font-weight: 500; cursor: pointer;
  transition: background 0.15s;
}
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent2); }
.btn-primary:disabled { background: #3a4060; color: #666; cursor: not-allowed; }
.btn-secondary { background: var(--surface2); color: var(--text2); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--border); }
.status { font-size: 0.82rem; color: var(--text2); }

/* Results */
.results { display: none; }
.results.visible { display: block; }

.section {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); margin-bottom: 14px; overflow: hidden;
}
.section-header {
  padding: 12px 16px; cursor: pointer; display: flex;
  justify-content: space-between; align-items: center;
  user-select: none; transition: background 0.1s;
}
.section-header:hover { background: var(--surface2); }
.section-header h3 { font-size: 0.92rem; font-weight: 500; }
.section-header .tag {
  font-size: 0.7rem; padding: 2px 8px; border-radius: 10px;
  background: var(--surface2); color: var(--text2);
}
.section-body { padding: 0 16px 16px; display: none; }
.section.open .section-body { display: block; }
.section-header .arrow { transition: transform 0.2s; color: var(--text2); }
.section.open .section-header .arrow { transform: rotate(90deg); }

/* Summary banner */
.summary-card {
  background: linear-gradient(135deg, #1e2235, #252a3e);
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 20px; margin-bottom: 14px;
}
.summary-card .conclusion { font-size: 1.05rem; font-weight: 500; margin-bottom: 10px; }
.summary-card .summary-text { color: var(--text2); font-size: 0.88rem; }
.confidence-badge {
  display: inline-block; padding: 3px 10px; border-radius: 10px;
  font-size: 0.72rem; font-weight: 600; margin-top: 8px;
}
.conf-high { background: rgba(76,175,130,0.15); color: var(--green); }
.conf-medium { background: rgba(232,164,74,0.15); color: var(--orange); }
.conf-low { background: rgba(224,85,85,0.15); color: var(--red); }

/* Big Five */
.big-five-grid {
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;
  margin: 12px 0;
}
.trait-card {
  background: var(--surface2); border-radius: 6px; padding: 12px;
  text-align: center;
}
.trait-card .trait-name { font-size: 0.72rem; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }
.trait-card .trait-val { font-size: 1.4rem; font-weight: 600; margin: 4px 0; }
.trait-card .trait-bar { height: 4px; border-radius: 2px; background: var(--border); margin-top: 6px; }
.trait-card .trait-bar-fill { height: 100%; border-radius: 2px; background: var(--accent); }

/* Key-value list */
.kv-list { list-style: none; }
.kv-list li {
  padding: 6px 0; border-bottom: 1px solid var(--border);
  display: flex; justify-content: space-between; font-size: 0.85rem;
}
.kv-list li:last-child { border-bottom: none; }
.kv-list .kv-key { color: var(--text2); }
.kv-list .kv-val { font-weight: 500; text-align: right; max-width: 60%; }

/* Defense / prediction cards */
.card-list { display: flex; flex-direction: column; gap: 8px; }
.mini-card {
  background: var(--surface2); border-radius: 6px; padding: 10px 14px;
  font-size: 0.84rem;
}
.mini-card .mc-title { font-weight: 500; margin-bottom: 3px; }
.mini-card .mc-detail { color: var(--text2); font-size: 0.78rem; }

/* Loop diagram */
.loop-card {
  background: var(--surface2); border-radius: 6px; padding: 12px 14px;
  margin-bottom: 8px;
}
.loop-card .loop-name { font-weight: 500; font-size: 0.88rem; margin-bottom: 4px; }
.loop-chain { font-size: 0.78rem; color: var(--accent); margin-bottom: 4px; font-family: monospace; }
.loop-desc { font-size: 0.8rem; color: var(--text2); }
.loop-badge {
  display: inline-block; font-size: 0.68rem; padding: 1px 6px;
  border-radius: 8px; margin-left: 6px;
}
.loop-reinforcing { background: rgba(224,85,85,0.15); color: var(--red); }
.loop-balancing { background: rgba(76,175,130,0.15); color: var(--green); }

/* Uncertainty */
.missing-list { list-style: disc; padding-left: 20px; }
.missing-list li { font-size: 0.84rem; color: var(--text2); padding: 2px 0; }

/* JSON view */
.json-toggle { margin-top: 14px; }
.json-view {
  display: none; background: var(--surface2); border-radius: 6px;
  padding: 14px; margin-top: 8px; overflow-x: auto;
  max-height: 500px; font-size: 0.78rem; font-family: "Cascadia Code", "Fira Code", monospace;
  color: var(--text2); white-space: pre-wrap; word-break: break-all;
}
.json-view.visible { display: block; }

.gt-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.gt-card {
  background: var(--surface2);
  border-radius: 6px;
  padding: 12px 14px;
}
.gt-path {
  font-size: 0.72rem;
  color: var(--accent);
  margin-bottom: 6px;
  font-family: "Cascadia Code", "Fira Code", monospace;
  word-break: break-all;
}
.gt-value {
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 8px;
}
.gt-subtitle {
  font-size: 0.76rem;
  color: var(--text2);
  margin: 8px 0 4px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}
.gt-body {
  font-size: 0.8rem;
  color: var(--text2);
  white-space: pre-wrap;
}
.gt-list {
  list-style: disc;
  padding-left: 18px;
  color: var(--text2);
}
.gt-list li {
  font-size: 0.8rem;
  margin-bottom: 4px;
}
.gt-provenance {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.prov-card {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  background: rgba(255,255,255,0.02);
}
.prov-head {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: center;
  margin-bottom: 4px;
}
.prov-kind {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--accent);
  font-family: "Cascadia Code", "Fira Code", monospace;
}
.prov-id {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text);
  word-break: break-all;
}
.prov-meta,
.prov-segments,
.prov-motifs {
  font-size: 0.76rem;
  color: var(--text2);
  margin-top: 3px;
  word-break: break-word;
}
.evidence-card {
  background: var(--surface2);
  border-radius: 6px;
  padding: 12px 14px;
  margin-bottom: 8px;
}
.evidence-card .excerpt {
  font-size: 0.84rem;
  margin-bottom: 8px;
}
.evidence-card .meta {
  font-size: 0.76rem;
  color: var(--text2);
}

@media (max-width: 640px) {
  .big-five-grid { grid-template-columns: repeat(3, 1fr); }
  .gt-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="container">
  <div class="top-bar">
    <div class="left">
      <h1><span>Segmentum</span> <span id="titleText">Personality Analyzer</span></h1>
      <p class="subtitle" id="subtitleText"></p>
    </div>
    <button class="lang-toggle" id="langBtn" onclick="toggleLang()">EN / CN</button>
  </div>

  <div class="input-panel">
    <label for="materials" id="labelText"></label>
    <textarea id="materials" class="materials-box"></textarea>
    <p class="hint" id="hintText"></p>
    <div class="btn-row">
      <button id="analyzeBtn" class="btn btn-primary" onclick="runAnalysis()"></button>
      <button id="exampleBtn" class="btn btn-secondary" onclick="loadExample()"></button>
      <button id="gtExampleBtn" class="btn btn-secondary" onclick="loadGroundTruthExample()"></button>
      <button id="exportBtn" class="btn btn-secondary" onclick="exportGroundTruthJsonl()"></button>
      <span id="status" class="status"></span>
    </div>
  </div>

  <div id="results" class="results"></div>
</div>

<script>
/* ------------------------------------------------------------------ */
/*  i18n                                                               */
/* ------------------------------------------------------------------ */
const I = {
  en: {
    title: 'Personality Analyzer',
    subtitle: 'Free Energy Principle / Active Inference inverse personality inference',
    label: 'Input Materials',
    placeholder: 'Paste text materials here, one segment per line.\\n\\nExamples:\\nI explored a new trail through the valley, mapping unfamiliar terrain.\\nA friend helped me when I was lost. They shared food and stayed nearby.\\nI was excluded from the group. They rejected me and trust was broken.\\n\\nSupports: conversations, diary entries, behavioral descriptions, biographical fragments.\\nSupports Chinese / English / mixed.',
    hint: 'Each non-empty line is treated as one material segment. More segments = higher confidence.',
    analyze: 'Analyze',
    example: 'Load Example',
    gtExample: 'Load Ground Truth Example',
    exportJsonl: 'Export JSONL',
    analyzing: (n) => `Analyzing ${n} segment(s)...`,
    done: 'Done.',
    error: (m) => `Error: ${m}`,
    alertEmpty: 'Please enter at least one text segment.',
    overallConf: 'Overall confidence',
    // section titles
    secBigFive: 'Big Five Personality Traits',
    secPriors: 'Core Priors',
    secPriorsTag: 'self / others / world',
    secValues: 'Value Hierarchy',
    secValuesTag: 'ranked',
    secCog: 'Cognitive Style',
    secAffect: 'Affective Dynamics',
    secDefense: 'Defense Mechanisms',
    secSocial: 'Social Orientation',
    secLoops: 'Feedback Loops',
    secDev: 'Developmental Inferences',
    secStab: 'Stability Analysis',
    secPred: 'Behavioral Predictions',
    secUnc: 'Uncertainty & Gaps',
    secVia: 'VIA Character Strengths',
    secEvidence: 'Evidence Trace',
    secGT: 'Ground Truth View',
    secViaTag: '24 strengths',
    secEvidenceTag: 'source materials',
    secGTTag: 'confidence / evidence / reasoning',
    toggleJson: 'Toggle Raw JSON',
    // sub labels
    detected: 'detected',
    noneDetected: 'None detected.',
    insufficientData: 'Insufficient data.',
    none: 'None.',
    noPredictions: 'No predictions.',
    stableCore: 'Stable Core',
    fragilePoints: 'Fragile Points',
    plasticPoints: 'Plastic Points',
    missingEvidence: 'Missing Evidence',
    unresolvableQ: 'Unresolvable Questions',
    baselineArousal: 'Baseline Arousal',
    recoverySpeed: 'Recovery Speed',
    emotionWeights: 'Emotion Channel Weights',
    evidence: 'Evidence',
    reasoning: 'Reasoning',
    provenance: 'Provenance',
    sourceSegments: 'Segments',
    motifs: 'Motifs',
    supportCount: 'Support',
    schemaLinks: 'Schemas',
    sourceIndex: 'Source',
    category: 'Category',
    appraisal: 'Appraisal',
    noEvidence: 'No evidence attached.',
    noGroundTruth: 'No ConfidenceRated outputs available.',
    // Big Five
    traitOpenness: 'Openness',
    traitConsc: 'Consc.',
    traitExtra: 'Extraversion',
    traitAgree: 'Agreeable.',
    traitNeuro: 'Neuroticism',
    // Core Priors
    selfWorth: 'Self Worth',
    selfEfficacy: 'Self Efficacy',
    otherReliability: 'Other Reliability',
    otherPredictability: 'Other Predictability',
    worldSafety: 'World Safety',
    worldFairness: 'World Fairness',
    // Cognitive
    abstractConcrete: 'Abstract vs Concrete',
    globalDetail: 'Global vs Detail',
    causalAttrib: 'Causal Attribution',
    reflectiveDepth: 'Reflective Depth',
    coherenceNeed: 'Coherence Need',
    ambiguityTol: 'Ambiguity Tolerance',
    // defense
    target: 'Target',
    benefit: 'Benefit',
    cost: 'Cost',
    // predictions
    if_: 'IF',
    then_: 'THEN',
    // social orientation
    compete: 'compete', cooperate: 'cooperate', attach: 'attach', avoid: 'avoid',
    please: 'please', dominate: 'dominate', observe: 'observe',
    // values
    survival:'survival', safety:'safety', control:'control', dignity:'dignity',
    relation:'relation', achievement:'achievement', freedom:'freedom',
    truth:'truth', meaning:'meaning', contribution:'contribution',
    // emotions
    shame:'shame', anger:'anger', anxiety:'anxiety', sadness:'sadness', void_:'void', disgust:'disgust',
    // report
    genReport: 'Generate Report',
    rptTitle: 'Personality Analysis Report',
    rptSubtitle: 'Segmentum \u00b7 FEP/Active Inference Personality Generative Model',
    rptGenTime: 'Generated',
    rptMaterialCount: 'Material segments analyzed',
    rptSummary: 'Summary',
    rptConclusion: 'Conclusion',
    rptBigFiveDesc: 'Personality trait scores on a 0\u2013100 scale. Scores above 65 are considered high; below 35 are considered low.',
    rptPriorsDesc: 'Fundamental beliefs about the self, others, and the world, inferred from appraisal patterns.',
    rptValuesDesc: 'Value dimensions ranked by inferred importance, derived from behavioral and linguistic evidence.',
    rptCogDesc: 'Characteristic patterns of information processing and reasoning.',
    rptAffectDesc: 'Emotional processing baseline and dominant emotion channels.',
    rptDefenseDesc: 'Defense mechanisms inferred from personality profile and appraisal patterns.',
    rptSocialDesc: 'Interpersonal strategy weights inferred from Big Five traits.',
    rptLoopsDesc: 'Self-reinforcing or balancing feedback dynamics within the personality system.',
    rptDevDesc: 'Hypothetical developmental history inferred from current personality structure.',
    rptStabDesc: 'Which traits are crystallized (stable), which are defended (fragile), and which are still forming (plastic).',
    rptPredDesc: 'Expected behavioral responses in specific scenarios.',
    rptUncDesc: 'Information gaps and questions that cannot be resolved from the available evidence.',
    rptViaDesc: 'VIA Institute 24 character strengths projected from personality parameters.',
    rptDisclaimer: 'This report is generated by an algorithmic model based on the Free Energy Principle. It is not a clinical psychological assessment. The analysis is limited by the quality and quantity of input materials. All inferences should be interpreted as hypotheses, not diagnoses.',
    rptFramework: 'Analytical Framework',
    rptFrameworkText: 'This analysis uses Karl Friston\\'s Free Energy Principle (FEP) and Active Inference as its theoretical foundation. The personality is modeled as a generative system that continuously predicts sensory input and minimizes prediction error. Personality traits, defense mechanisms, and behavioral strategies are understood as stable patterns of error-reduction that the system has learned over its developmental history.',
  },
  zh: {
    title: '人格分析器',
    subtitle: '基于自由能原理 / 主动推理的逆向人格推断',
    label: '输入材料',
    placeholder: '在此粘贴文本材料，每行一个片段。\\n\\n示例：\\n我探索了一条穿越山谷的新小径，绘制了陌生地形的地图。\\n一个朋友在我迷路时帮助了我，他们分享了食物，一直陪在我身边。\\n我被群体排斥了，他们拒绝了我，我感到被抛弃和羞辱，信任破裂了。\\n\\n支持：对话、日记、行为描述、传记片段。\\n支持中文 / 英文 / 混合。',
    hint: '每个非空行视为一个材料片段。片段越多，置信度越高。',
    analyze: '开始分析',
    example: '加载示例',
    analyzing: (n) => `正在分析 ${n} 个片段...`,
    done: '完成。',
    error: (m) => `错误: ${m}`,
    alertEmpty: '请输入至少一条文本。',
    overallConf: '总体置信度',
    secBigFive: '大五人格特质',
    secPriors: '核心先验信念',
    secPriorsTag: '自我 / 他人 / 世界',
    secValues: '价值层级',
    secValuesTag: '排序',
    secCog: '认知风格',
    secAffect: '情感动力学',
    secDefense: '防御机制',
    secSocial: '社交取向',
    secLoops: '反馈环路',
    secDev: '发展史推断',
    secStab: '稳定性分析',
    secPred: '行为预测',
    secUnc: '不确定性与缺口',
    secVia: 'VIA 品格优势',
    secViaTag: '24 项优势',
    toggleJson: '切换原始 JSON',
    detected: '已检出',
    noneDetected: '未检出。',
    insufficientData: '数据不足。',
    none: '无。',
    noPredictions: '无预测。',
    stableCore: '稳定核心',
    fragilePoints: '脆弱点',
    plasticPoints: '可塑点',
    missingEvidence: '缺失证据',
    unresolvableQ: '不可解问题',
    baselineArousal: '基线唤醒度',
    recoverySpeed: '恢复速度',
    emotionWeights: '情绪通道权重',
    traitOpenness: '开放性',
    traitConsc: '尽责性',
    traitExtra: '外向性',
    traitAgree: '宜人性',
    traitNeuro: '神经质',
    selfWorth: '自我价值感',
    selfEfficacy: '自我效能感',
    otherReliability: '他人可靠性',
    otherPredictability: '他人可预测性',
    worldSafety: '世界安全感',
    worldFairness: '世界公平感',
    abstractConcrete: '抽象 vs 具象',
    globalDetail: '全局 vs 细节',
    causalAttrib: '归因倾向',
    reflectiveDepth: '反思深度',
    coherenceNeed: '一致性需求',
    ambiguityTol: '模糊容忍度',
    target: '目标误差',
    benefit: '短期收益',
    cost: '长期代价',
    if_: '若',
    then_: '则',
    compete:'竞争', cooperate:'合作', attach:'依恋', avoid:'回避',
    please:'讨好', dominate:'支配', observe:'观察',
    survival:'生存', safety:'安全', control:'控制', dignity:'尊严',
    relation:'关系', achievement:'成就', freedom:'自由',
    truth:'真理', meaning:'意义', contribution:'贡献',
    shame:'羞耻', anger:'愤怒', anxiety:'焦虑', sadness:'悲伤', void_:'虚无', disgust:'厌恶',
    genReport: '生成报告',
    rptTitle: '人格分析报告',
    rptSubtitle: 'Segmentum \u00b7 基于自由能原理/主动推理的人格生成模型',
    rptGenTime: '生成时间',
    rptMaterialCount: '分析材料片段数',
    rptSummary: '总述',
    rptConclusion: '结论',
    rptBigFiveDesc: '人格特质分数（0\u2013100）。65 以上为高，35 以下为低。',
    rptPriorsDesc: '关于自我、他人和世界的基本信念，从评估模式中推断。',
    rptValuesDesc: '按推断重要性排列的价值维度，来源于行为和语言证据。',
    rptCogDesc: '信息处理和推理的特征模式。',
    rptAffectDesc: '情绪处理基线和主导情绪通道。',
    rptDefenseDesc: '从人格特征和评估模式推断的防御机制。',
    rptSocialDesc: '从大五人格推断的人际策略权重。',
    rptLoopsDesc: '人格系统内的自我强化或平衡反馈动力学。',
    rptDevDesc: '从当前人格结构推断的假设性发展史。',
    rptStabDesc: '哪些特质已结晶（稳定），哪些被防御（脆弱），哪些仍在形成（可塑）。',
    rptPredDesc: '在特定情境下的预期行为反应。',
    rptUncDesc: '信息缺口和无法从现有证据中解决的问题。',
    rptViaDesc: 'VIA 24 项品格优势，由人格参数投影得出。',
    rptDisclaimer: '本报告由基于自由能原理的算法模型生成，不构成临床心理评估。分析结果受输入材料的质量和数量限制。所有推断应被视为假设，而非诊断。',
    rptFramework: '分析框架',
    rptFrameworkText: '本分析以 Karl Friston 的自由能原理（FEP）和主动推理为理论基础。人格被建模为一个持续预测感觉输入并最小化预测误差的生成系统。人格特质、防御机制和行为策略被理解为系统在发展历程中习得的稳定误差缩减模式。',
  },
};

let lang = 'zh';
function t(key) { return I[lang][key] ?? I['en'][key] ?? key; }
function toggleLang() {
  lang = lang === 'zh' ? 'en' : 'zh';
  document.getElementById('langBtn').textContent = lang === 'zh' ? 'EN / CN' : 'CN / EN';
  applyLang();
  if (_lastData) renderResults(_lastData);
}
function applyLang() {
  document.getElementById('titleText').textContent = t('title');
  document.getElementById('subtitleText').textContent = t('subtitle');
  document.getElementById('labelText').textContent = t('label');
  document.getElementById('materials').placeholder = t('placeholder');
  document.getElementById('hintText').textContent = t('hint');
  document.getElementById('analyzeBtn').textContent = t('analyze');
  document.getElementById('exampleBtn').textContent = t('example');
  document.getElementById('gtExampleBtn').textContent = t('gtExample');
  document.getElementById('exportBtn').textContent = t('exportJsonl');
}
applyLang();

let _lastData = null;

const EXAMPLE_TEXT = `I explored a new trail through the valley, mapping unfamiliar terrain. The signals were novel and I adapted quickly.
A friend helped me when I was lost. They shared food and stayed nearby until I felt safe. I trusted them completely.
I was excluded from the group. They rejected me and I felt abandoned and humiliated. Trust was broken.
\\u7b2c\\u4e8c\\u5929\\uff0cagent\\u6628\\u5929\\u8def\\u8fc7\\u6cb3\\u8fb9\\uff0c\\u88ab\\u4e00\\u53ea\\u9cc4\\u9c7c\\u653b\\u51fb\\u4e86\\uff0c\\u53d7\\u4f24\\u4e86\\u3002
I found berries and shared a meal with my group. Safe resources.`;

const GROUND_TRUTH_EXAMPLE_TEXT = `Morgan spent the entire weekend building a detailed spreadsheet
to track every household expense down to the cent. When a friend
suggested a spontaneous road trip, Morgan declined, saying they
needed to finish organizing the garage according to a color-coded
system they had designed.`;

function loadExample() {
  document.getElementById('materials').value = EXAMPLE_TEXT;
}

function loadGroundTruthExample() {
  document.getElementById('materials').value = GROUND_TRUTH_EXAMPLE_TEXT;
}

async function runAnalysis() {
  const raw = document.getElementById('materials').value.trim();
  if (!raw) { alert(t('alertEmpty')); return; }
  const materials = raw.split('\\n').map(s => s.trim()).filter(Boolean);
  const btn = document.getElementById('analyzeBtn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = t('analyzing')(materials.length);

  try {
    const resp = await fetch('/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({materials}),
    });
    if (!resp.ok) { throw new Error('HTTP ' + resp.status); }
    const data = await resp.json();
    _lastData = data;
    renderResults(data);
    status.textContent = t('done');
  } catch(e) {
    status.textContent = t('error')(e.message);
  } finally {
    btn.disabled = false;
  }
}

async function exportGroundTruthJsonl() {
  const raw = document.getElementById('materials').value.trim();
  if (!raw) { alert(t('alertEmpty')); return; }
  const materials = raw.split('\\n').map(s => s.trim()).filter(Boolean);
  const status = document.getElementById('status');
  status.textContent = t('analyzing')(materials.length);
  try {
    const res = await fetch('/analyze/ground-truth/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ materials })
    });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `segmentum-ground-truth-${new Date().toISOString().slice(0,19).replace(/[:T]/g, '-')}.jsonl`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    status.textContent = t('done');
  } catch (err) {
    status.textContent = t('error')(String(err));
  }
}

/* ------------------------------------------------------------------ */
/*  Rendering helpers                                                  */
/* ------------------------------------------------------------------ */

function confClass(c) {
  if (typeof c === 'string') {
    if (c === 'high') return 'conf-high';
    if (c === 'medium') return 'conf-medium';
    return 'conf-low';
  }
  if (c >= 0.6) return 'conf-high';
  if (c >= 0.4) return 'conf-medium';
  return 'conf-low';
}

function confLabel(c) {
  if (typeof c === 'number') return (c * 100).toFixed(0) + '%';
  return c;
}

function fmtVal(v) {
  if (typeof v === 'number') return v.toFixed(3);
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function traitColor(v) {
  if (v >= 0.65) return 'var(--green)';
  if (v <= 0.35) return 'var(--red)';
  return 'var(--accent)';
}

const _socialMap = {compete:'compete',cooperate:'cooperate',attach:'attach',avoid:'avoid',please:'please',dominate:'dominate',observe:'observe'};
const _valueMap = {survival:'survival',safety:'safety',control:'control',dignity:'dignity',relation:'relation',achievement:'achievement',freedom:'freedom',truth:'truth',meaning:'meaning',contribution:'contribution'};
const _emotionMap = {shame:'shame',anger:'anger',anxiety:'anxiety',sadness:'sadness','void':'void_',disgust:'disgust'};

function tKey(key, map) { const k = map[key]; return k ? t(k) : key; }

function section(id, title, tag, bodyHtml) {
  return `<div class="section" id="sec-${id}">
    <div class="section-header" onclick="this.parentElement.classList.toggle('open')">
      <h3><span class="arrow">&#9654;</span> ${title}</h3>
      ${tag ? '<span class=\\"tag\\">' + tag + '</span>' : ''}
    </div>
    <div class="section-body">${bodyHtml}</div>
  </div>`;
}

function renderBigFive(b5) {
  const names = {openness:t('traitOpenness'),conscientiousness:t('traitConsc'),extraversion:t('traitExtra'),agreeableness:t('traitAgree'),neuroticism:t('traitNeuro')};
  let h = '<div class="big-five-grid">';
  for (const [k,label] of Object.entries(names)) {
    const v = b5[k] || 0.5;
    const pct = (v*100).toFixed(0);
    h += `<div class="trait-card">
      <div class="trait-name">${label}</div>
      <div class="trait-val" style="color:${traitColor(v)}">${pct}</div>
      <div class="trait-bar"><div class="trait-bar-fill" style="width:${pct}%;background:${traitColor(v)}"></div></div>
    </div>`;
  }
  return h + '</div>';
}

function renderCorePriors(cp) {
  const labels = {self_worth:t('selfWorth'),self_efficacy:t('selfEfficacy'),other_reliability:t('otherReliability'),other_predictability:t('otherPredictability'),world_safety:t('worldSafety'),world_fairness:t('worldFairness')};
  let h = '<ul class="kv-list">';
  for (const [k,label] of Object.entries(labels)) {
    const item = cp[k];
    if (!item) continue;
    h += `<li><span class="kv-key">${label}</span><span class="kv-val">${fmtVal(item.value)} <span class="confidence-badge ${confClass(item.confidence)}">${item.confidence}</span></span></li>`;
  }
  return h + '</ul>';
}

function renderCogStyle(cs) {
  const labels = {abstract_vs_concrete:t('abstractConcrete'),global_vs_detail:t('globalDetail'),causal_attribution_tendency:t('causalAttrib'),reflective_depth:t('reflectiveDepth'),coherence_need:t('coherenceNeed'),ambiguity_tolerance:t('ambiguityTol')};
  let h = '<ul class="kv-list">';
  for (const [k,label] of Object.entries(labels)) {
    const item = cs[k];
    if (!item) continue;
    h += `<li><span class="kv-key">${label}</span><span class="kv-val">${fmtVal(item.value)}</span></li>`;
  }
  return h + '</ul>';
}

function renderSocialOrientation(so) {
  const w = so.orientation_weights || {};
  const entries = Object.entries(w).sort((a,b) => (b[1].value||0)-(a[1].value||0));
  let h = '<ul class="kv-list">';
  for (const [k,cr] of entries) {
    const pct = ((cr.value||0)*100).toFixed(0);
    h += `<li><span class="kv-key">${tKey(k,_socialMap)}</span><span class="kv-val">${pct}%</span></li>`;
  }
  return h + '</ul>';
}

function renderValueHierarchy(vh) {
  const ranked = vh.ranked_values || [];
  let h = '<div class="card-list">';
  for (let i = 0; i < ranked.length; i++) {
    const item = ranked[i];
    const name = item.name || '?';
    const val = item.value != null ? fmtVal(item.value) : '?';
    h += `<div class="mini-card"><span class="mc-title">#${i+1} ${tKey(name,_valueMap)}</span> <span style="float:right">${val}</span></div>`;
  }
  return h + '</div>';
}

function renderDefenses(dm) {
  const mechs = dm.mechanisms || [];
  if (!mechs.length) return `<p style="color:var(--text2)">${t('noneDetected')}</p>`;
  let h = '<div class="card-list">';
  for (const m of mechs) {
    h += `<div class="mini-card">
      <div class="mc-title">${m.name} <span class="confidence-badge ${confClass(m.confidence)}">${m.confidence}</span></div>
      <div class="mc-detail">${t('target')}: ${m.target_error}</div>
      <div class="mc-detail">${t('benefit')}: ${m.short_term_benefit} | ${t('cost')}: ${m.long_term_cost}</div>
    </div>`;
  }
  return h + '</div>';
}

function renderAffect(ad) {
  let h = '<ul class="kv-list">';
  h += `<li><span class="kv-key">${t('baselineArousal')}</span><span class="kv-val">${fmtVal(ad.baseline_arousal?.value)}</span></li>`;
  h += `<li><span class="kv-key">${t('recoverySpeed')}</span><span class="kv-val">${fmtVal(ad.recovery_speed?.value)}</span></li>`;
  h += '</ul>';
  const w = ad.emotion_channel_weights || {};
  const sorted = Object.entries(w).sort((a,b)=>b[1]-a[1]);
  h += `<div style="margin-top:10px"><div style="font-size:0.78rem;color:var(--text2);margin-bottom:6px">${t('emotionWeights')}</div>`;
  for (const [name,val] of sorted) {
    const pct = (val*100).toFixed(0);
    h += `<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
      <span style="width:60px;font-size:0.8rem;color:var(--text2)">${tKey(name,_emotionMap)}</span>
      <div style="flex:1;height:6px;background:var(--border);border-radius:3px">
        <div style="width:${Math.min(100,pct*2)}%;height:100%;border-radius:3px;background:var(--accent)"></div>
      </div>
      <span style="font-size:0.78rem;width:36px;text-align:right">${pct}%</span>
    </div>`;
  }
  return h + '</div>';
}

function renderLoops(loops) {
  if (!loops.length) return `<p style="color:var(--text2)">${t('noneDetected')}</p>`;
  let h = '';
  for (const l of loops) {
    const badge = l.valence === 'reinforcing' ? 'loop-reinforcing' : 'loop-balancing';
    h += `<div class="loop-card">
      <div class="loop-name">${l.name} <span class="loop-badge ${badge}">${l.valence}</span></div>
      <div class="loop-chain">${(l.components||[]).join(' \\u2192 ')}</div>
      <div class="loop-desc">${l.description}</div>
    </div>`;
  }
  return h;
}

function renderCRList(items, fallback) {
  if (!items || !items.length) return `<p style="color:var(--text2)">${fallback}</p>`;
  let h = '<div class="card-list">';
  for (const cr of items) {
    h += `<div class="mini-card">
      <div class="mc-title">${typeof cr.value === 'string' ? cr.value : fmtVal(cr.value)} <span class="confidence-badge ${confClass(cr.confidence)}">${cr.confidence}</span></div>
      ${cr.reasoning ? '<div class="mc-detail">' + cr.reasoning + '</div>' : ''}
    </div>`;
  }
  return h + '</div>';
}

function renderPredictions(preds) {
  if (!preds.length) return `<p style="color:var(--text2)">${t('noPredictions')}</p>`;
  let h = '<div class="card-list">';
  for (const p of preds) {
    h += `<div class="mini-card">
      <div class="mc-title">${t('if_')}: ${p.scenario} <span class="confidence-badge ${confClass(p.confidence)}">${p.confidence}</span></div>
      <div class="mc-detail">${t('then_')}: ${p.predicted_behavior}</div>
    </div>`;
  }
  return h + '</div>';
}

function renderEvidenceList(items) {
  if (!items || !items.length) return `<p style="color:var(--text2)">${t('insufficientData')}</p>`;
  let h = '';
  for (const item of items) {
    const appraisal = Object.entries(item.appraisal_relevance || {})
      .map(([k, v]) => `${escapeHtml(k)}=${fmtVal(v)}`)
      .join(', ');
    h += `<div class="evidence-card">
      <div class="excerpt">${escapeHtml(item.excerpt || '')}</div>
      <div class="meta">${t('sourceIndex')}: ${item.source_index ?? 0} | ${t('category')}: ${escapeHtml(item.category || '')}</div>
      <div class="meta">${t('appraisal')}: ${escapeHtml(appraisal || '-')}</div>
    </div>`;
  }
  return h;
}

function pushCREntry(entries, path, value) {
  if (!value || typeof value !== 'object') return;
  if (!Object.prototype.hasOwnProperty.call(value, 'confidence')) return;
  entries.push({ path, value });
}

function collectGroundTruthEntries(data) {
  const entries = [];

  const cp = data.core_priors || {};
  for (const [k, v] of Object.entries(cp)) pushCREntry(entries, `core_priors.${k}`, v);

  const vh = (data.value_hierarchy || {}).ranked_values || [];
  for (const item of vh) pushCREntry(entries, `value_hierarchy.${item.name}`, item);

  const cs = data.cognitive_style || {};
  for (const [k, v] of Object.entries(cs)) pushCREntry(entries, `cognitive_style.${k}`, v);

  const pa = data.precision_allocation || {};
  for (const item of (pa.hypersensitive_channels || [])) pushCREntry(entries, 'precision_allocation.hypersensitive_channels', item);
  for (const item of (pa.blind_spots || [])) pushCREntry(entries, 'precision_allocation.blind_spots', item);
  pushCREntry(entries, 'precision_allocation.internal_vs_external', pa.internal_vs_external);
  pushCREntry(entries, 'precision_allocation.immediate_vs_narrative', pa.immediate_vs_narrative);

  const ad = data.affective_dynamics || {};
  pushCREntry(entries, 'affective_dynamics.baseline_arousal', ad.baseline_arousal);
  pushCREntry(entries, 'affective_dynamics.recovery_speed', ad.recovery_speed);
  for (const item of (ad.dominant_emotions || [])) pushCREntry(entries, 'affective_dynamics.dominant_emotions', item);

  const so = (data.social_orientation || {}).orientation_weights || {};
  for (const [k, v] of Object.entries(so)) pushCREntry(entries, `social_orientation.${k}`, v);

  const sm = data.self_model_profile || {};
  pushCREntry(entries, 'self_model_profile.self_narrative', sm.self_narrative);
  for (const item of (sm.identity_consistency_needs || [])) pushCREntry(entries, 'self_model_profile.identity_consistency_needs', item);
  for (const item of (sm.identity_threats || [])) pushCREntry(entries, 'self_model_profile.identity_threats', item);

  const om = data.other_model_profile || {};
  for (const [k, v] of Object.entries(om)) pushCREntry(entries, `other_model_profile.${k}`, v);

  const ts = data.temporal_structure || {};
  for (const [k, v] of Object.entries(ts)) pushCREntry(entries, `temporal_structure.${k}`, v);

  const sp = data.strategy_profile || {};
  for (const item of (sp.preferred_strategies || [])) pushCREntry(entries, 'strategy_profile.preferred_strategies', item);
  for (const [k, v] of Object.entries(sp.cost_analysis || {})) pushCREntry(entries, `strategy_profile.cost_analysis.${k}`, v);
  for (const item of (sp.blocked_strategies || [])) pushCREntry(entries, 'strategy_profile.blocked_strategies', item);

  for (const item of (data.developmental_inferences || [])) pushCREntry(entries, 'developmental_inferences', item);
  for (const item of (data.stable_core || [])) pushCREntry(entries, 'stable_core', item);
  for (const item of (data.fragile_points || [])) pushCREntry(entries, 'fragile_points', item);
  for (const item of (data.plastic_points || [])) pushCREntry(entries, 'plastic_points', item);

  return entries;
}

function renderGroundTruth(entries) {
  if (!entries.length) return `<p style="color:var(--text2)">${t('noGroundTruth')}</p>`;
  const renderProvenance = (details) => {
    if (!details || !details.length) return '';
    const cards = details.map((detail) => {
      if (detail.kind === 'schema') {
        const motifs = (detail.motif_signature || []).map(item => escapeHtml(String(item))).join(', ');
        const episodeIds = (detail.supporting_episode_ids || []).map(item => escapeHtml(String(item))).join(', ');
        return `<div class="prov-card">
          <div class="prov-head">
            <div class="prov-kind">schema</div>
            <div class="prov-id">${escapeHtml(detail.schema_id || '')}</div>
          </div>
          <div class="prov-meta">${t('supportCount')}: ${escapeHtml(fmtVal(detail.support_count ?? 0))} | direction: ${escapeHtml(detail.dominant_direction || '-')}</div>
          <div class="prov-motifs">${t('motifs')}: ${motifs || '-'}</div>
          <div class="prov-segments">episodes: ${episodeIds || '-'}</div>
        </div>`;
      }
      const segments = (detail.supporting_segments || []).map(item => escapeHtml(String(item))).join(' | ');
      const schemas = (detail.matched_schema_ids || []).map(item => escapeHtml(String(item))).join(', ');
      const appraisal = Object.entries(detail.appraisal_relevance || {})
        .map(([k, v]) => `${escapeHtml(k)}=${escapeHtml(fmtVal(v))}`)
        .join(', ');
      return `<div class="prov-card">
        <div class="prov-head">
          <div class="prov-kind">episode</div>
          <div class="prov-id">${escapeHtml(detail.episode_id || '')}</div>
        </div>
        <div class="prov-meta">${t('category')}: ${escapeHtml(detail.category || '-')} | event: ${escapeHtml(detail.compiled_event_type || '-')} | outcome: ${escapeHtml(detail.predicted_outcome || '-')}</div>
        <div class="prov-meta">${t('sourceIndex')}: ${escapeHtml(fmtVal(detail.source_index ?? 0))} | confidence: ${escapeHtml(fmtVal(detail.compiler_confidence ?? 0))}</div>
        <div class="prov-segments">${t('sourceSegments')}: ${segments || '-'}</div>
        <div class="prov-motifs">${t('schemaLinks')}: ${schemas || '-'}${appraisal ? ' | ' + t('appraisal') + ': ' + appraisal : ''}</div>
      </div>`;
    }).join('');
    return `<div class="gt-subtitle">${t('provenance')}</div><div class="gt-provenance">${cards}</div>`;
  };
  let h = '<div class="gt-grid">';
  for (const entry of entries) {
    const cr = entry.value || {};
    const evidenceItems = (cr.evidence || []).length
      ? '<ul class="gt-list">' + cr.evidence.map(item => `<li>${escapeHtml(item)}</li>`).join('') + '</ul>'
      : `<div class="gt-body">${t('noEvidence')}</div>`;
    const provenanceItems = renderProvenance(cr.evidence_details || []);
    h += `<div class="gt-card">
      <div class="gt-path">${escapeHtml(entry.path)}</div>
      <div class="gt-value">${escapeHtml(fmtVal(cr.value))} <span class="confidence-badge ${confClass(cr.confidence)}">${escapeHtml(cr.confidence || 'low')}</span></div>
      <div class="gt-subtitle">${t('reasoning')}</div>
      <div class="gt-body">${escapeHtml(cr.reasoning || '')}</div>
      <div class="gt-subtitle">${t('evidence')}</div>
      ${evidenceItems}
      ${provenanceItems}
    </div>`;
  }
  return h + '</div>';
}

function renderResults(data) {
  const el = document.getElementById('results');
  let h = '';

  // Summary card
  h += `<div class="summary-card">
    <div class="conclusion">${data.one_line_conclusion || ''}</div>
    <div class="summary-text">${data.summary || ''}</div>
    <span class="confidence-badge ${confClass(data.analysis_confidence)}">${t('overallConf')}: ${confLabel(data.analysis_confidence)}</span>
  </div>`;

  h += section('b5', t('secBigFive'), null, renderBigFive(data.big_five || {}));
  h += section('evidence', t('secEvidence'), t('secEvidenceTag'), renderEvidenceList(data.evidence_list || []));
  h += section('priors', t('secPriors'), t('secPriorsTag'), renderCorePriors(data.core_priors || {}));
  h += section('values', t('secValues'), t('secValuesTag'), renderValueHierarchy(data.value_hierarchy || {}));
  h += section('cog', t('secCog'), null, renderCogStyle(data.cognitive_style || {}));
  h += section('affect', t('secAffect'), null, renderAffect(data.affective_dynamics || {}));

  const nDef = (data.defense_mechanisms?.mechanisms||[]).length;
  h += section('defense', t('secDefense'), nDef + ' ' + t('detected'), renderDefenses(data.defense_mechanisms || {}));

  h += section('social', t('secSocial'), null, renderSocialOrientation(data.social_orientation || {}));
  h += section('loops', t('secLoops'), null, renderLoops(data.feedback_loops || []));
  h += section('dev', t('secDev'), null, renderCRList(data.developmental_inferences, t('insufficientData')));

  let stabHtml = `<h4 style="font-size:0.82rem;color:var(--text2);margin:8px 0 4px">${t('stableCore')}</h4>`;
  stabHtml += renderCRList(data.stable_core, t('none'));
  stabHtml += `<h4 style="font-size:0.82rem;color:var(--text2);margin:12px 0 4px">${t('fragilePoints')}</h4>`;
  stabHtml += renderCRList(data.fragile_points, t('none'));
  stabHtml += `<h4 style="font-size:0.82rem;color:var(--text2);margin:12px 0 4px">${t('plasticPoints')}</h4>`;
  stabHtml += renderCRList(data.plastic_points, t('none'));
  h += section('stab', t('secStab'), null, stabHtml);

  h += section('pred', t('secPred'), null, renderPredictions(data.behavioral_predictions || []));

  let uncHtml = `<h4 style="font-size:0.82rem;color:var(--text2);margin:8px 0 4px">${t('missingEvidence')}</h4>`;
  uncHtml += '<ul class="missing-list">' + (data.missing_evidence||[]).map(s=>'<li>'+s+'</li>').join('') + '</ul>';
  uncHtml += `<h4 style="font-size:0.82rem;color:var(--text2);margin:12px 0 4px">${t('unresolvableQ')}</h4>`;
  uncHtml += '<ul class="missing-list">' + (data.unresolvable_questions||[]).map(s=>'<li>'+s+'</li>').join('') + '</ul>';
  h += section('unc', t('secUnc'), null, uncHtml);

  const via = data.via_strengths || {};
  const viaSorted = Object.entries(via).sort((a,b)=>b[1]-a[1]);
  let viaHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px">';
  for (const [name,val] of viaSorted) {
    const pct = (val*100).toFixed(0);
    viaHtml += `<div class="mini-card" style="padding:8px 10px;text-align:center">
      <div style="font-size:0.72rem;color:var(--text2)">${name.replace(/_/g,' ')}</div>
      <div style="font-size:1.1rem;font-weight:600;color:${traitColor(val)}">${pct}</div>
    </div>`;
  }
  viaHtml += '</div>';
  h += section('via', t('secVia'), t('secViaTag'), viaHtml);
  h += section('gt', t('secGT'), t('secGTTag'), renderGroundTruth(collectGroundTruthEntries(data)));

  h += `<div class="json-toggle" style="display:flex;gap:10px;flex-wrap:wrap">
    <button class="btn btn-primary" onclick="generateReport()">${t('genReport')}</button>
    <button class="btn btn-secondary" onclick="document.getElementById('jsonraw').classList.toggle('visible')">${t('toggleJson')}</button>
    <div id="jsonraw" class="json-view" style="width:100%">${JSON.stringify(data, null, 2)}</div>
  </div>`;

  el.innerHTML = h;
  el.classList.add('visible');
  document.getElementById('sec-b5')?.classList.add('open');
}

/* ------------------------------------------------------------------ */
/*  Report generation                                                  */
/* ------------------------------------------------------------------ */

function generateReport() {
  if (!_lastData) return;
  const d = _lastData;
  const now = new Date().toLocaleString(lang === 'zh' ? 'zh-CN' : 'en-US');
  const b5 = d.big_five || {};
  const traitNames = {openness:t('traitOpenness'),conscientiousness:t('traitConsc'),extraversion:t('traitExtra'),agreeableness:t('traitAgree'),neuroticism:t('traitNeuro')};
  const priorNames = {self_worth:t('selfWorth'),self_efficacy:t('selfEfficacy'),other_reliability:t('otherReliability'),other_predictability:t('otherPredictability'),world_safety:t('worldSafety'),world_fairness:t('worldFairness')};
  const cogNames = {abstract_vs_concrete:t('abstractConcrete'),global_vs_detail:t('globalDetail'),causal_attribution_tendency:t('causalAttrib'),reflective_depth:t('reflectiveDepth'),coherence_need:t('coherenceNeed'),ambiguity_tolerance:t('ambiguityTol')};

  function confDot(c) {
    if (c==='high') return '\\u{1f7e2}';
    if (c==='medium') return '\\u{1f7e1}';
    return '\\u{1f534}';
  }
  function pct(v) { return typeof v==='number' ? (v*100).toFixed(0) : v; }
  function barSvg(val, color) {
    const w = Math.max(2, Math.min(100, Math.round(val*100)));
    return `<svg width="120" height="10" style="vertical-align:middle"><rect width="120" height="10" rx="5" fill="#e8e8ef"/><rect width="${w*1.2}" height="10" rx="5" fill="${color}"/></svg>`;
  }
  function trColor(v) { return v>=0.65?'#2e8b57':v<=0.35?'#c0392b':'#3a6abf'; }

  // Big Five table rows
  let b5Rows = '';
  for (const [k,label] of Object.entries(traitNames)) {
    const v = b5[k]||0.5;
    const p = pct(v);
    const c = trColor(v);
    b5Rows += `<tr><td style="font-weight:500">${label}</td><td style="text-align:center"><span style="font-size:1.3em;font-weight:700;color:${c}">${p}</span></td><td>${barSvg(v,c)}</td><td style="color:#888;font-size:0.85em">${v>0.65?(lang==='zh'?'\\u9ad8':'High'):v<0.35?(lang==='zh'?'\\u4f4e':'Low'):(lang==='zh'?'\\u4e2d\\u7b49':'Moderate')}</td></tr>`;
  }

  // Core Priors rows
  let priorsRows = '';
  const cp = d.core_priors || {};
  for (const [k,label] of Object.entries(priorNames)) {
    const item = cp[k];
    if (!item) continue;
    priorsRows += `<tr><td>${label}</td><td style="text-align:center;font-weight:600">${fmtVal(item.value)}</td><td>${confDot(item.confidence)} ${item.confidence}</td></tr>`;
  }

  // Value hierarchy rows
  let valRows = '';
  const ranked = (d.value_hierarchy||{}).ranked_values || [];
  for (let i=0;i<ranked.length;i++) {
    const item = ranked[i];
    const name = item.name||'?';
    valRows += `<tr><td style="text-align:center;font-weight:600">#${i+1}</td><td>${tKey(name,_valueMap)}</td><td style="text-align:center">${fmtVal(item.value)}</td></tr>`;
  }

  // Cognitive style rows
  let cogRows = '';
  const cs = d.cognitive_style || {};
  for (const [k,label] of Object.entries(cogNames)) {
    const item = cs[k];
    if (!item) continue;
    cogRows += `<tr><td>${label}</td><td style="text-align:center;font-weight:500">${fmtVal(item.value)}</td></tr>`;
  }

  // Affective dynamics
  const ad = d.affective_dynamics || {};
  const ew = ad.emotion_channel_weights || {};
  const ewSorted = Object.entries(ew).sort((a,b)=>b[1]-a[1]);
  let emotionBars = '';
  for (const [name,val] of ewSorted) {
    const p = (val*100).toFixed(0);
    emotionBars += `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px"><span style="width:60px;font-size:0.9em;color:#666">${tKey(name,_emotionMap)}</span><div style="flex:1;height:8px;background:#e8e8ef;border-radius:4px"><div style="width:${Math.min(100,p*2)}%;height:100%;border-radius:4px;background:#5b7fc7"></div></div><span style="width:36px;text-align:right;font-size:0.85em">${p}%</span></div>`;
  }

  // Defense mechanisms
  const mechs = (d.defense_mechanisms||{}).mechanisms || [];
  let defRows = '';
  for (const m of mechs) {
    defRows += `<tr><td style="font-weight:500">${m.name} ${confDot(m.confidence)}</td><td>${m.target_error}</td><td style="color:#2e8b57">${m.short_term_benefit}</td><td style="color:#c0392b">${m.long_term_cost}</td></tr>`;
  }

  // Social orientation rows
  const so = d.social_orientation||{};
  const sow = so.orientation_weights || {};
  const soSorted = Object.entries(sow).sort((a,b)=>(b[1].value||0)-(a[1].value||0));
  let soRows = '';
  for (const [k,cr] of soSorted) {
    const p = ((cr.value||0)*100).toFixed(0);
    soRows += `<tr><td>${tKey(k,_socialMap)}</td><td style="text-align:center;font-weight:500">${p}%</td><td>${barSvg(cr.value||0,'#5b7fc7')}</td></tr>`;
  }

  // Feedback loops
  let loopBlocks = '';
  for (const l of (d.feedback_loops||[])) {
    const badge = l.valence==='reinforcing' ? '<span style="color:#c0392b;font-weight:600">\\u21bb '+l.valence+'</span>' : '<span style="color:#2e8b57;font-weight:600">\\u21cb '+l.valence+'</span>';
    loopBlocks += `<div style="background:#f8f9fb;border-left:4px solid ${l.valence==='reinforcing'?'#e88':'#6b6'};padding:14px 18px;border-radius:6px;margin-bottom:12px">
      <div style="font-weight:600;margin-bottom:4px">${l.name} ${badge}</div>
      <div style="font-family:monospace;color:#5b7fc7;font-size:0.88em;margin-bottom:6px">${(l.components||[]).join(' \\u2192 ')}</div>
      <div style="color:#555;font-size:0.9em">${l.description}</div>
    </div>`;
  }

  // Development inferences
  let devBlocks = '';
  for (const cr of (d.developmental_inferences||[])) {
    devBlocks += `<div style="border-left:3px solid #ccc;padding:8px 14px;margin-bottom:8px;color:#444"><span>${confDot(cr.confidence)}</span> ${typeof cr.value==='string'?cr.value:fmtVal(cr.value)}<div style="font-size:0.82em;color:#999;margin-top:2px">${cr.reasoning||''}</div></div>`;
  }

  // Stability
  function stabList(items) {
    if (!items||!items.length) return `<p style="color:#999">${t('none')}</p>`;
    let h='';
    for (const cr of items) h += `<li style="margin-bottom:4px">${confDot(cr.confidence)} ${typeof cr.value==='string'?cr.value:fmtVal(cr.value)}</li>`;
    return '<ul style="padding-left:20px">'+h+'</ul>';
  }

  // Predictions
  let predBlocks = '';
  for (const p of (d.behavioral_predictions||[])) {
    predBlocks += `<div style="background:#f8f9fb;padding:12px 16px;border-radius:6px;margin-bottom:8px">
      <div style="font-weight:500;margin-bottom:3px">${confDot(p.confidence)} ${t('if_')}: ${p.scenario}</div>
      <div style="color:#555">${t('then_')}: ${p.predicted_behavior}</div>
    </div>`;
  }

  // Uncertainty
  let missingHtml = '<ul style="padding-left:20px;color:#666">'+(d.missing_evidence||[]).map(s=>'<li>'+s+'</li>').join('')+'</ul>';
  let unresHtml = '<ul style="padding-left:20px;color:#666">'+(d.unresolvable_questions||[]).map(s=>'<li>'+s+'</li>').join('')+'</ul>';

  // VIA strengths
  const via = d.via_strengths || {};
  const viaSorted = Object.entries(via).sort((a,b)=>b[1]-a[1]);
  let viaGrid = '';
  for (const [name,val] of viaSorted) {
    const p = (val*100).toFixed(0);
    const c = trColor(val);
    viaGrid += `<div style="background:#f8f9fb;border-radius:6px;padding:8px;text-align:center"><div style="font-size:0.78em;color:#888">${name.replace(/_/g,' ')}</div><div style="font-size:1.2em;font-weight:700;color:${c}">${p}</div></div>`;
  }

  // Temporal structure
  const ts = d.temporal_structure || {};
  let tsHtml = '';
  if (ts.past_trauma_weight) {
    const labels = {past_trauma_weight:lang==='zh'?'\\u8fc7\\u53bb\\u521b\\u4f24\\u6743\\u91cd':'Past Trauma Weight',present_pressure_weight:lang==='zh'?'\\u5f53\\u524d\\u538b\\u529b\\u6743\\u91cd':'Present Pressure Weight',future_imagination_weight:lang==='zh'?'\\u672a\\u6765\\u60f3\\u8c61\\u6743\\u91cd':'Future Imagination Weight'};
    tsHtml = '<table style="width:100%;border-collapse:collapse">';
    for (const [k,label] of Object.entries(labels)) {
      const item = ts[k];
      if (!item) continue;
      tsHtml += `<tr><td style="padding:6px 0;border-bottom:1px solid #eee">${label}</td><td style="text-align:center;font-weight:500;padding:6px 0;border-bottom:1px solid #eee">${pct(item.value)}%</td></tr>`;
    }
    tsHtml += '</table>';
  }

  const html = `<!DOCTYPE html>
<html lang="${lang==='zh'?'zh-CN':'en'}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${t('rptTitle')}</title>
<style>
@page { margin: 20mm 18mm; size: A4; }
@media print { .no-print { display: none !important; } body { font-size: 11pt; } }
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: "Noto Sans SC", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color:#222; line-height:1.7; background:#fff; }
.page { max-width:820px; margin:0 auto; padding:40px 36px; }
/* Header */
.header { text-align:center; padding-bottom:28px; border-bottom:2px solid #222; margin-bottom:32px; }
.header h1 { font-size:1.7rem; font-weight:700; letter-spacing:1px; margin-bottom:4px; }
.header .sub { color:#666; font-size:0.88rem; }
.header .meta { margin-top:12px; font-size:0.82rem; color:#999; }
.header .meta span { margin:0 10px; }
/* Sections */
.sec { margin-bottom:32px; page-break-inside:avoid; }
.sec h2 { font-size:1.15rem; font-weight:600; color:#222; margin-bottom:6px; padding-bottom:6px; border-bottom:1px solid #ddd; }
.sec .sec-desc { font-size:0.84rem; color:#888; margin-bottom:14px; }
/* Tables */
table { width:100%; border-collapse:collapse; }
th, td { padding:8px 10px; text-align:left; border-bottom:1px solid #eee; font-size:0.9rem; }
th { font-weight:600; color:#555; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.3px; }
/* Summary box */
.summary-box { background:#f4f6fa; border-radius:8px; padding:20px 24px; margin-bottom:14px; }
.summary-box .conclusion { font-size:1.1rem; font-weight:600; margin-bottom:8px; color:#2a2a2a; }
.summary-box .text { color:#555; font-size:0.92rem; }
.conf-pill { display:inline-block; padding:2px 10px; border-radius:10px; font-size:0.76rem; font-weight:600; }
.conf-pill.high { background:#e6f5ed; color:#2e8b57; }
.conf-pill.medium { background:#fef5e7; color:#c8870a; }
.conf-pill.low { background:#fde8e8; color:#c0392b; }
/* Framework */
.framework { background:#f9fafb; border:1px solid #e8e8ef; border-radius:8px; padding:18px 22px; margin-bottom:32px; }
.framework h3 { font-size:0.95rem; font-weight:600; margin-bottom:6px; }
.framework p { font-size:0.88rem; color:#555; }
/* Disclaimer */
.disclaimer { margin-top:40px; padding-top:20px; border-top:1px solid #ddd; font-size:0.8rem; color:#999; text-align:center; }
/* Footer */
.footer { text-align:center; margin-top:24px; font-size:0.78rem; color:#bbb; }
.print-btn { position:fixed; top:16px; right:16px; padding:8px 20px; border:none; border-radius:6px; background:#3a6abf; color:#fff; font-size:0.88rem; cursor:pointer; z-index:100; }
.print-btn:hover { background:#2d5aa0; }
</style>
</head>
<body>
<button class="print-btn no-print" onclick="window.print()">\\u{1f5a8} ${lang==='zh'?'\\u6253\\u5370 / \\u4fdd\\u5b58 PDF':'Print / Save PDF'}</button>
<div class="page">
  <div class="header">
    <h1>${t('rptTitle')}</h1>
    <div class="sub">${t('rptSubtitle')}</div>
    <div class="meta">
      <span>${t('rptGenTime')}: ${now}</span>
      <span>|</span>
      <span>${t('rptMaterialCount')}: ${(d.evidence_list||[]).length}</span>
      <span>|</span>
      <span>${t('overallConf')}: <span class="conf-pill ${d.analysis_confidence>=0.6?'high':d.analysis_confidence>=0.4?'medium':'low'}">${(d.analysis_confidence*100).toFixed(0)}%</span></span>
    </div>
  </div>

  <div class="framework">
    <h3>${t('rptFramework')}</h3>
    <p>${t('rptFrameworkText')}</p>
  </div>

  <div class="sec">
    <h2>1. ${t('rptSummary')}</h2>
    <div class="summary-box">
      <div class="conclusion">${d.one_line_conclusion||''}</div>
      <div class="text">${d.summary||''}</div>
    </div>
  </div>

  <div class="sec">
    <h2>2. ${t('secBigFive')}</h2>
    <p class="sec-desc">${t('rptBigFiveDesc')}</p>
    <table>
      <tr><th>${lang==='zh'?'\\u7279\\u8d28':'Trait'}</th><th style="text-align:center">${lang==='zh'?'\\u5206\\u6570':'Score'}</th><th></th><th>${lang==='zh'?'\\u6c34\\u5e73':'Level'}</th></tr>
      ${b5Rows}
    </table>
  </div>

  <div class="sec">
    <h2>3. ${t('secPriors')}</h2>
    <p class="sec-desc">${t('rptPriorsDesc')}</p>
    <table>
      <tr><th>${lang==='zh'?'\\u4fe1\\u5ff5\\u7ef4\\u5ea6':'Dimension'}</th><th style="text-align:center">${lang==='zh'?'\\u503c':'Value'}</th><th>${lang==='zh'?'\\u7f6e\\u4fe1\\u5ea6':'Confidence'}</th></tr>
      ${priorsRows}
    </table>
  </div>

  <div class="sec">
    <h2>4. ${t('secValues')}</h2>
    <p class="sec-desc">${t('rptValuesDesc')}</p>
    <table>
      <tr><th style="text-align:center">${lang==='zh'?'\\u6392\\u540d':'Rank'}</th><th>${lang==='zh'?'\\u4ef7\\u503c\\u7ef4\\u5ea6':'Value'}</th><th style="text-align:center">${lang==='zh'?'\\u5206\\u6570':'Score'}</th></tr>
      ${valRows}
    </table>
  </div>

  <div class="sec">
    <h2>5. ${t('secCog')}</h2>
    <p class="sec-desc">${t('rptCogDesc')}</p>
    <table>
      <tr><th>${lang==='zh'?'\\u7ef4\\u5ea6':'Dimension'}</th><th style="text-align:center">${lang==='zh'?'\\u503c':'Value'}</th></tr>
      ${cogRows}
    </table>
  </div>

  <div class="sec">
    <h2>6. ${t('secAffect')}</h2>
    <p class="sec-desc">${t('rptAffectDesc')}</p>
    <table>
      <tr><td style="font-weight:500">${t('baselineArousal')}</td><td style="text-align:center;font-weight:600">${fmtVal(ad.baseline_arousal?.value)}</td></tr>
      <tr><td style="font-weight:500">${t('recoverySpeed')}</td><td style="text-align:center;font-weight:600">${fmtVal(ad.recovery_speed?.value)}</td></tr>
    </table>
    <div style="margin-top:14px"><div style="font-weight:500;margin-bottom:8px;font-size:0.9em;color:#555">${t('emotionWeights')}</div>${emotionBars}</div>
  </div>

  ${tsHtml ? `<div class="sec"><h2>7. ${lang==='zh'?'\\u65f6\\u95f4\\u7ed3\\u6784':'Temporal Structure'}</h2>${tsHtml}</div>` : ''}

  <div class="sec">
    <h2>8. ${t('secDefense')}</h2>
    <p class="sec-desc">${t('rptDefenseDesc')}</p>
    ${mechs.length ? `<table>
      <tr><th>${lang==='zh'?'\\u673a\\u5236':'Mechanism'}</th><th>${t('target')}</th><th>${t('benefit')}</th><th>${t('cost')}</th></tr>
      ${defRows}
    </table>` : `<p style="color:#999">${t('noneDetected')}</p>`}
  </div>

  <div class="sec">
    <h2>9. ${t('secSocial')}</h2>
    <p class="sec-desc">${t('rptSocialDesc')}</p>
    <table>
      <tr><th>${lang==='zh'?'\\u53d6\\u5411':'Orientation'}</th><th style="text-align:center">${lang==='zh'?'\\u6743\\u91cd':'Weight'}</th><th></th></tr>
      ${soRows}
    </table>
  </div>

  <div class="sec">
    <h2>10. ${t('secLoops')}</h2>
    <p class="sec-desc">${t('rptLoopsDesc')}</p>
    ${loopBlocks || `<p style="color:#999">${t('noneDetected')}</p>`}
  </div>

  <div class="sec">
    <h2>11. ${t('secDev')}</h2>
    <p class="sec-desc">${t('rptDevDesc')}</p>
    ${devBlocks || `<p style="color:#999">${t('insufficientData')}</p>`}
  </div>

  <div class="sec">
    <h2>12. ${t('secStab')}</h2>
    <p class="sec-desc">${t('rptStabDesc')}</p>
    <h3 style="font-size:0.92rem;margin:10px 0 4px">${t('stableCore')}</h3>
    ${stabList(d.stable_core)}
    <h3 style="font-size:0.92rem;margin:14px 0 4px">${t('fragilePoints')}</h3>
    ${stabList(d.fragile_points)}
    <h3 style="font-size:0.92rem;margin:14px 0 4px">${t('plasticPoints')}</h3>
    ${stabList(d.plastic_points)}
  </div>

  <div class="sec">
    <h2>13. ${t('secPred')}</h2>
    <p class="sec-desc">${t('rptPredDesc')}</p>
    ${predBlocks || `<p style="color:#999">${t('noPredictions')}</p>`}
  </div>

  <div class="sec">
    <h2>14. ${t('secUnc')}</h2>
    <p class="sec-desc">${t('rptUncDesc')}</p>
    <h3 style="font-size:0.92rem;margin:8px 0 4px">${t('missingEvidence')}</h3>
    ${missingHtml}
    <h3 style="font-size:0.92rem;margin:14px 0 4px">${t('unresolvableQ')}</h3>
    ${unresHtml}
  </div>

  <div class="sec">
    <h2>15. ${t('secVia')}</h2>
    <p class="sec-desc">${t('rptViaDesc')}</p>
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:6px">
      ${viaGrid}
    </div>
  </div>

  <div class="sec">
    <h2>${t('rptConclusion')}</h2>
    <div class="summary-box">
      <div class="conclusion">${d.one_line_conclusion||''}</div>
    </div>
  </div>

  <div class="disclaimer">${t('rptDisclaimer')}</div>
  <div class="footer">Segmentum Personality Analyzer &middot; Free Energy Principle / Active Inference &middot; ${now}</div>
</div>
</body>
</html>`;

  const w = window.open('', '_blank');
  w.document.write(html);
  w.document.close();
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the Web UI."""
    return _HTML_PAGE


# ---------------------------------------------------------------------------
# API Endpoints (used by the UI via fetch, also usable directly)
# ---------------------------------------------------------------------------


def _build_analyzer(request: AnalysisRequest) -> PersonalityAnalyzer:
    llm_cfg = request.llm_config if request.llm_enhanced else None
    return PersonalityAnalyzer(llm_config=llm_cfg)


def _build_ground_truth_payload(request: AnalysisRequest) -> dict[str, Any]:
    analyzer = _build_analyzer(request)
    result = analyzer.analyze(request.materials, metadata=request.metadata)
    payload = result.to_dict()

    def _collect_confidence_rated(value: Any, path: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if "confidence" in value and "value" in value:
                items.append({
                    "path": path,
                    "value": value.get("value"),
                    "confidence": value.get("confidence"),
                    "evidence": list(value.get("evidence", [])),
                    "evidence_details": list(value.get("evidence_details", [])),
                    "reasoning": value.get("reasoning", ""),
                })
            else:
                for key, nested in value.items():
                    child_path = f"{path}.{key}" if path else str(key)
                    items.extend(_collect_confidence_rated(nested, child_path))
        elif isinstance(value, list):
            for idx, nested in enumerate(value):
                child_path = f"{path}[{idx}]"
                items.extend(_collect_confidence_rated(nested, child_path))
        return items

    return {
        "materials": list(request.materials),
        "analysis_confidence": payload.get("analysis_confidence", 0.0),
        "ground_truth": _collect_confidence_rated(payload, ""),
        "missing_evidence": list(payload.get("missing_evidence", [])),
        "unresolvable_questions": list(payload.get("unresolvable_questions", [])),
        "raw_analysis": payload,
    }


@app.post("/analyze")
def analyze_personality(request: AnalysisRequest) -> dict[str, Any]:
    """Full 10-step personality analysis from text materials."""
    analyzer = _build_analyzer(request)
    result = analyzer.analyze(request.materials, metadata=request.metadata)
    return result.to_dict()


@app.post("/analyze/ground-truth")
def analyze_ground_truth(request: AnalysisRequest) -> dict[str, Any]:
    """Ground-truth oriented view for benchmark generation workflows."""
    return _build_ground_truth_payload(request)


@app.post("/analyze/ground-truth/export", response_class=PlainTextResponse)
def export_ground_truth_jsonl(request: AnalysisRequest) -> str:
    """Export benchmark-ready ground truth as a single JSONL record."""
    payload = _build_ground_truth_payload(request)
    export_record = {
        "materials": payload["materials"],
        "analysis_confidence": payload["analysis_confidence"],
        "ground_truth": payload["ground_truth"],
        "missing_evidence": payload["missing_evidence"],
        "unresolvable_questions": payload["unresolvable_questions"],
        "raw_analysis": payload["raw_analysis"],
    }
    return json.dumps(export_record, ensure_ascii=False) + "\n"


@app.post("/analyze/evidence")
def extract_evidence(request: AnalysisRequest) -> dict[str, Any]:
    """Step 1 only: evidence extraction from materials."""
    analyzer = _build_analyzer(request)
    evidence, _, appraisals, signals, semantic_schemas = analyzer._extract_evidence(request.materials)
    agg = analyzer._aggregate_appraisals(appraisals)
    big_five = analyzer._aggregate_big_five(signals)
    return {
        "evidence": [e.to_dict() for e in evidence],
        "aggregate_appraisal": agg,
        "big_five": big_five,
        "evidence_count": len(evidence),
        "semantic_schemas": semantic_schemas,
    }


@app.post("/analyze/parameters")
def infer_parameters(request: AnalysisRequest) -> dict[str, Any]:
    """Steps 1-3: evidence + hypothesis + parameter space."""
    analyzer = _build_analyzer(request)
    evidence, _, appraisals, signals, semantic_schemas = analyzer._extract_evidence(request.materials)
    agg = analyzer._aggregate_appraisals(appraisals)
    big_five = analyzer._aggregate_big_five(signals)
    conf = "medium" if len(request.materials) >= 2 else "low"
    hypothesis = analyzer._build_predictive_hypothesis(evidence, agg, big_five, semantic_schemas)
    core_priors = analyzer._infer_core_priors(agg, big_five, conf, evidence, semantic_schemas)
    cognitive_style = analyzer._infer_cognitive_style(agg, big_five, conf, evidence, semantic_schemas)
    affective = analyzer._infer_affective_dynamics(agg, big_five, conf, evidence, semantic_schemas)
    social = analyzer._infer_social_orientation(big_five, conf, evidence, semantic_schemas)
    precision = analyzer._infer_precision_allocation(agg, big_five, conf, evidence, semantic_schemas)
    temporal = analyzer._infer_temporal_structure(agg, big_five, conf, evidence, semantic_schemas)
    value_hierarchy = analyzer._infer_value_hierarchy(agg, big_five, conf, evidence, semantic_schemas)

    return {
        "evidence_count": len(evidence),
        "big_five": big_five,
        "semantic_schemas": semantic_schemas,
        "hypothesis": hypothesis.to_dict(),
        "core_priors": core_priors.to_dict(),
        "cognitive_style": cognitive_style.to_dict(),
        "affective_dynamics": affective.to_dict(),
        "social_orientation": social.to_dict(),
        "precision_allocation": precision.to_dict(),
        "temporal_structure": temporal.to_dict(),
        "value_hierarchy": value_hierarchy.to_dict(),
    }


@app.post("/analyze/simulate")
def simulate_personality(request: SimulationRequest) -> dict[str, Any]:
    """Forward simulation: given inferred personality, run N cycles."""
    try:
        from .narrative_initialization import NarrativeInitializer
        from .narrative_types import NarrativeEpisode
        from .runtime import SegmentRuntime
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"Simulation requires full segmentum installation: {exc}",
        )

    big_five = request.personality.get("big_five", {})
    if not big_five:
        raise HTTPException(
            status_code=422,
            detail="personality.big_five is required for simulation",
        )

    runtime = SegmentRuntime.load_or_create(state_path=None, trace_path=None)
    agent = runtime.agent

    pp = agent.self_model.personality
    for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        if trait in big_five:
            setattr(pp, trait, float(big_five[trait]))

    scenario_episode = NarrativeEpisode(
        episode_id="sim-scenario",
        timestamp=0,
        source="simulation",
        raw_text=request.scenario,
        tags=["simulation_scenario"],
    )

    initializer = NarrativeInitializer()
    init_result = initializer.initialize_agent(agent, [scenario_episode])

    cycle_traces: list[dict[str, Any]] = []
    for cycle in range(min(request.cycles, 500)):
        try:
            tick_result = runtime.tick()
            cycle_traces.append({
                "cycle": cycle,
                "action": tick_result.get("chosen_action", "unknown") if isinstance(tick_result, dict) else "completed",
                "surprise": tick_result.get("surprise", 0.0) if isinstance(tick_result, dict) else 0.0,
            })
        except Exception:
            break

    return {
        "cycles_completed": len(cycle_traces),
        "initialization": {
            "personality": {t: getattr(pp, t) for t in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")},
            "policy_distribution": init_result.policy_distribution,
        },
        "trace_summary": cycle_traces[-5:] if cycle_traces else [],
        "final_state": {
            "personality": {t: getattr(pp, t) for t in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")},
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "segmentum-personality-analyzer"}
