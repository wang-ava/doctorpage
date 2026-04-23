const questionInput = document.getElementById("questionInput");
const apiKeyInput = document.getElementById("apiKeyInput");
const apiKeyStatus = document.getElementById("apiKeyStatus");
const modelInput = document.getElementById("modelInput");
const modelStatus = document.getElementById("modelStatus");
const modelNote = document.getElementById("modelNote");
const modelSuggestions = document.getElementById("modelSuggestions");
const modelPicker = document.getElementById("modelPicker");
const imageInput = document.getElementById("imageInput");
const imagePickerButton = document.getElementById("imagePickerButton");
const imagePickerStatus = document.getElementById("imagePickerStatus");
const imagePreview = document.getElementById("imagePreview");
const submitButton = document.getElementById("submitButton");
const resultsShell = document.getElementById("resultsShell");
const advancedReview = document.getElementById("advancedReview");
const pipelineBadge = document.getElementById("pipelineBadge");
const activityFeed = document.getElementById("activityFeed");
const activityItemTemplate = document.getElementById("activityItemTemplate");

const languageBadge = document.getElementById("languageBadge");
const languageRationale = document.getElementById("languageRationale");
const translationState = document.getElementById("translationState");
const translatedQuestion = document.getElementById("translatedQuestion");
const backTranslationState = document.getElementById("backTranslationState");
const translatedAnswer = document.getElementById("translatedAnswer");

const answerState = document.getElementById("answerState");
const englishAnswer = document.getElementById("englishAnswer");
const statementConfidenceState = document.getElementById("statementConfidenceState");
const statementConfidenceSummary = document.getElementById("statementConfidenceSummary");
const statementConfidenceList = document.getElementById("statementConfidenceList");
const logprobState = document.getElementById("logprobState");
const metricAvgProb = document.getElementById("metricAvgProb");
const metricPerplexity = document.getElementById("metricPerplexity");
const metricLowConfidence = document.getElementById("metricLowConfidence");
const metricTotalTokens = document.getElementById("metricTotalTokens");
const tokenStream = document.getElementById("tokenStream");
const tokenLegend = document.getElementById("tokenLegend");
const pipelineAnalyticsState = document.getElementById("pipelineAnalyticsState");
const pipelineFigure = document.getElementById("pipelineFigure");
const perplexityState = document.getElementById("perplexityState");
const perplexityFigure = document.getElementById("perplexityFigure");
const perplexitySummary = document.getElementById("perplexitySummary");
const logicBreakState = document.getElementById("logicBreakState");
const logicBreakCount = document.getElementById("logicBreakCount");
const logicBreakRate = document.getElementById("logicBreakRate");
const logicBreakList = document.getElementById("logicBreakList");
const mutationState = document.getElementById("mutationState");
const mutationCount = document.getElementById("mutationCount");
const mutationRate = document.getElementById("mutationRate");
const mutationFigure = document.getElementById("mutationFigure");
const mutationSummary = document.getElementById("mutationSummary");

const stepChips = {
  detect_language: document.querySelector('[data-step="detect_language"]'),
  translate_input: document.querySelector('[data-step="translate_input"]'),
  answer_in_english: document.querySelector('[data-step="answer_in_english"]'),
  translate_answer_back: document.querySelector('[data-step="translate_answer_back"]'),
};

let selectedImages = [];
let translatedQuestionBuffer = "";
let englishAnswerBuffer = "";
let translatedAnswerBuffer = "";
let isSubmitting = false;
let requiresUserApiKey = true;
const apiKeyStorageKey = "doctorWebSavedOpenRouterApiKey";
const modelStorageKey = "doctorWebSelectedModel";
let defaultModelName = "openai/gpt-4o";
let tokenAnalyticsModel = "openai/gpt-4o";
let currentRunModel = "openai/gpt-4o";
let openRouterModels = [];
let modelSearchAbortController = null;

const placeholderApiKeys = new Set([
  "your-api-key-here",
  "your-openrouter-api-key",
  "your key here",
]);

imagePickerButton.addEventListener("click", () => {
  imageInput.click();
});

if (apiKeyInput) {
  apiKeyInput.addEventListener("input", () => {
    persistApiKeyValue();
    syncApiKeyState();
  });
}

if (modelInput) {
  modelInput.addEventListener("input", () => {
    persistModelValue();
    syncModelState();
    scheduleModelSearch();
  });
  modelInput.addEventListener("focus", () => {
    renderModelPicker(filterModels(modelInput.value));
    modelPicker?.classList.add("is-open");
  });
  modelInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      modelPicker?.classList.remove("is-open");
    }
  });
}

document.addEventListener("click", (event) => {
  if (!modelInput || !modelPicker) {
    return;
  }
  if (event.target === modelInput || modelPicker.contains(event.target)) {
    return;
  }
  modelPicker.classList.remove("is-open");
});

imageInput.addEventListener("change", async (event) => {
  const files = Array.from(event.target.files || []);
  selectedImages = await Promise.all(files.map(fileToPayload));
  updateImagePickerStatus(files);
  renderImagePreview();
});

submitButton.addEventListener("click", runConsultation);

boot();

async function boot() {
  restorePersistedApiKey();
  restorePersistedModel();
  await refreshQuotaState();
  syncApiKeyState();
  syncModelState();
  hideResultsShell();
  renderIdleProcessingLog();
}

function restorePersistedApiKey() {
  if (!apiKeyInput) {
    return;
  }
  try {
    const savedValue = localStorage.getItem(apiKeyStorageKey);
    if (savedValue) {
      apiKeyInput.value = savedValue;
    }
  } catch (error) {
    // Ignore storage failures and continue without persistence.
  }
}

function persistApiKeyValue() {
  if (!apiKeyInput) {
    return;
  }
  try {
    const value = apiKeyInput.value.trim();
    if (!value) {
      localStorage.removeItem(apiKeyStorageKey);
      return;
    }
    localStorage.setItem(apiKeyStorageKey, value);
  } catch (error) {
    // Ignore storage failures and continue normally.
  }
}

function restorePersistedModel() {
  if (!modelInput) {
    return;
  }
  try {
    const savedValue = localStorage.getItem(modelStorageKey);
    if (savedValue) {
      modelInput.value = savedValue;
    }
  } catch (error) {
    // Ignore storage failures and continue without persistence.
  }
}

function persistModelValue() {
  if (!modelInput) {
    return;
  }
  try {
    const value = modelInput.value.trim();
    if (!value) {
      localStorage.removeItem(modelStorageKey);
      return;
    }
    localStorage.setItem(modelStorageKey, value);
  } catch (error) {
    // Ignore storage failures and continue normally.
  }
}

function getSelectedModelValue() {
  return modelInput?.value.trim() || defaultModelName;
}

function populateModelSuggestions(models = []) {
  if (!modelSuggestions) {
    return;
  }
  const values = Array.from(
    new Set(
      [defaultModelName, ...(models || [])]
        .map((model) => (typeof model === "string" ? model : model?.id))
        .filter(Boolean)
    )
  );
  modelSuggestions.replaceChildren(
    ...values.map((value) => {
      const option = document.createElement("option");
      option.value = value;
      return option;
    })
  );
}

function normalizeModelOption(model) {
  if (typeof model === "string") {
    return {
      id: model,
      name: model,
      provider: model.split("/", 1)[0] || "",
      description: "",
      context_length: null,
      input_modalities: [],
      output_modalities: [],
      supports_token_analytics: model === tokenAnalyticsModel,
    };
  }
  return {
    id: model?.id || "",
    name: model?.name || model?.id || "",
    provider: model?.provider || (model?.id?.split("/", 1)[0] || ""),
    description: model?.description || "",
    context_length: model?.context_length ?? null,
    input_modalities: model?.input_modalities || [],
    output_modalities: model?.output_modalities || [],
    supports_token_analytics: Boolean(model?.supports_token_analytics || model?.id === tokenAnalyticsModel),
  };
}

function setOpenRouterModels(models = []) {
  const byId = new Map();
  for (const model of [defaultModelName, ...(models || [])]) {
    const normalized = normalizeModelOption(model);
    if (normalized.id) {
      byId.set(normalized.id, normalized);
    }
  }
  openRouterModels = Array.from(byId.values());
  populateModelSuggestions(openRouterModels);
  renderModelPicker(filterModels(modelInput?.value || ""));
}

function filterModels(query) {
  const terms = String(query || "")
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  const source = openRouterModels.length ? openRouterModels : [normalizeModelOption(defaultModelName)];
  if (!terms.length) {
    return source.slice(0, 18);
  }
  return source
    .map((model) => {
      const haystack = `${model.id} ${model.name} ${model.provider} ${model.description}`.toLowerCase();
      const matched = terms.every((term) => haystack.includes(term));
      const firstTerm = terms[0] || "";
      const score =
        (model.id.toLowerCase().startsWith(firstTerm) ? 0 : 2) +
        (model.name.toLowerCase().startsWith(firstTerm) ? 0 : 1) +
        (model.supports_token_analytics ? -1 : 0);
      return { model, matched, score };
    })
    .filter((entry) => entry.matched)
    .sort((a, b) => a.score - b.score || a.model.id.localeCompare(b.model.id))
    .slice(0, 24)
    .map((entry) => entry.model);
}

function renderModelPicker(models = []) {
  if (!modelPicker) {
    return;
  }
  modelPicker.innerHTML = "";
  if (!models.length) {
    const empty = document.createElement("div");
    empty.className = "model-option model-option-empty";
    empty.textContent = "No matching OpenRouter models found.";
    modelPicker.appendChild(empty);
    modelPicker.classList.add("is-open");
    return;
  }

  for (const model of models) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-option";
    button.setAttribute("role", "option");
    const context = typeof model.context_length === "number" ? `${model.context_length.toLocaleString()} context` : "";
    const modalities = [...(model.input_modalities || []), ...(model.output_modalities || [])]
      .filter(Boolean)
      .slice(0, 3)
      .join(" + ");
    button.innerHTML = `
      <span class="model-option-main">
        <strong>${escapeHtml(model.name || model.id)}</strong>
        <code>${escapeHtml(model.id)}</code>
      </span>
      <span class="model-option-meta">
        ${model.supports_token_analytics ? "<mark>Detailed confidence</mark>" : ""}
        ${context ? `<span>${escapeHtml(context)}</span>` : ""}
        ${modalities ? `<span>${escapeHtml(modalities)}</span>` : ""}
      </span>
    `;
    button.addEventListener("click", () => {
      modelInput.value = model.id;
      persistModelValue();
      syncModelState();
      modelPicker.classList.remove("is-open");
    });
    modelPicker.appendChild(button);
  }
  modelPicker.classList.add("is-open");
}

async function loadOpenRouterModels(query = "") {
  if (modelSearchAbortController) {
    modelSearchAbortController.abort();
  }
  modelSearchAbortController = new AbortController();
  const url = query ? `/api/models?q=${encodeURIComponent(query)}` : "/api/models";
  const response = await fetch(url, { signal: modelSearchAbortController.signal });
  if (!response.ok) {
    throw new Error("Unable to load OpenRouter models.");
  }
  const payload = await response.json();
  setOpenRouterModels(payload.models || []);
}

function scheduleModelSearch() {
  if (!modelInput) {
    return;
  }
  renderModelPicker(filterModels(modelInput.value));
  window.clearTimeout(scheduleModelSearch.timer);
  scheduleModelSearch.timer = window.setTimeout(() => {
    loadOpenRouterModels(modelInput.value.trim()).catch(() => {
      // Keep local/cached options if the OpenRouter model list cannot be refreshed.
    });
  }, 180);
}

function syncModelState() {
  const selectedModel = getSelectedModelValue();
  const detailedConfidenceAvailable = selectedModel === tokenAnalyticsModel;
  if (modelStatus) {
    modelStatus.textContent = detailedConfidenceAvailable ? "Detailed confidence available" : "Answer generation only";
  }
  if (modelNote) {
    modelNote.innerHTML = detailedConfidenceAvailable
      ? `Using <code>${escapeHtml(selectedModel)}</code>. This selection supports the detailed statement and token-level confidence review shown below.`
      : `Using <code>${escapeHtml(selectedModel)}</code>. You can still generate the answer with this model, but detailed confidence review in this interface is currently available only when <code>${escapeHtml(tokenAnalyticsModel)}</code> is selected.`;
  }
}

async function refreshQuotaState() {
  try {
    const response = await fetch("/api/meta");
    const meta = await response.json();
    requiresUserApiKey = meta.requires_user_api_key !== false;
    defaultModelName = meta.default_model || meta.model || defaultModelName;
    tokenAnalyticsModel = meta.token_analytics_model || defaultModelName;
    if (questionInput) {
      questionInput.maxLength = meta.max_input_chars;
    }
    if (modelInput && !modelInput.value.trim()) {
      modelInput.value = defaultModelName;
    }
    setOpenRouterModels(meta.models || meta.model_suggestions || []);
    currentRunModel = getSelectedModelValue();
    syncApiKeyState();
    syncModelState();
    loadOpenRouterModels().catch(() => {
      // The initial meta response already contains fallback model options.
    });
  } catch (error) {
    // The page can still work without meta preloading.
  }
}

async function runConsultation() {
  const text = questionInput.value.trim();
  const apiKey = getApiKeyValue();
  const selectedModel = getSelectedModelValue();
  if (requiresUserApiKey && !isUsableApiKey(apiKey)) {
    pushActivity(
      "API key required",
      "Create an OpenRouter account, generate an API key, and paste it into the access field before running an analysis."
    );
    if (apiKeyInput) {
      apiKeyInput.focus();
    }
    return;
  }
  if (!text) {
    pushActivity("Missing input", "Please enter a text question before starting.");
    return;
  }

  revealResultsShell();
  resetOutputs();
  setBusy(true);
  currentRunModel = selectedModel;
  pipelineBadge.textContent = "Running";
  pushActivity("Request started", `The case has been submitted and processing has started with ${selectedModel}.`);

  try {
    let streamHadError = false;
    const response = await fetch("/api/consult/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-OpenRouter-API-Key": apiKey,
      },
      body: JSON.stringify({ text, images: selectedImages, model: selectedModel }),
    });

    if (!response.ok) {
      const message = await readErrorMessage(response);
      throw new Error(message || "Request failed.");
    }

    await consumeNdjsonStream(response.body, (event) => {
      if (event.type === "error") {
        streamHadError = true;
      }
      handleEvent(event);
    });
    if (!streamHadError) {
      pipelineBadge.textContent = "Completed";
    }
  } catch (error) {
    pipelineBadge.textContent = "Failed";
    setAllStepStates("error");
    pushActivity("Request failed", error.message || "Unknown error.");
  } finally {
    setBusy(false);
  }
}

async function consumeNdjsonStream(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n");
    while (boundary >= 0) {
      const line = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 1);
      if (line) {
        onEvent(JSON.parse(line));
      }
      boundary = buffer.indexOf("\n");
    }

    if (done) {
      const trailing = buffer.trim();
      if (trailing) {
        onEvent(JSON.parse(trailing));
      }
      break;
    }
  }
}

function handleEvent(event) {
  switch (event.type) {
    case "status":
      markStepActive(event.stage);
      pushActivity(stageLabel(event.stage), event.message);
      break;
    case "detection":
      markStepDone("detect_language");
      languageBadge.textContent = event.is_english
        ? `English (${event.language_code || "en"})`
        : `${event.language_name} (${event.language_code || "unknown"})`;
      languageRationale.textContent = event.rationale || event.raw || "The model did not provide additional rationale.";
      if (event.is_english) {
        translationState.textContent = "Skipped";
        backTranslationState.textContent = "Waiting";
      }
      break;
    case "translation_skipped":
      markStepDone("translate_input");
      translationState.textContent = "Skipped";
      translatedQuestionBuffer = event.english_text;
      renderMarkdownBlock(translatedQuestion, translatedQuestionBuffer);
      pushActivity("Input translation", event.message);
      break;
    case "translation_chunk":
      markStepActive("translate_input");
      translationState.textContent = "Translating";
      translatedQuestionBuffer += event.text;
      renderMarkdownBlock(translatedQuestion, translatedQuestionBuffer);
      break;
    case "translation_complete":
      markStepDone("translate_input");
      translationState.textContent = "Completed";
      translatedQuestionBuffer = event.text;
      renderMarkdownBlock(translatedQuestion, translatedQuestionBuffer);
      break;
    case "answer_ready":
      markStepActive("answer_in_english");
      currentRunModel = event.model || currentRunModel;
      answerState.textContent = "Streaming";
      logprobState.textContent = event.logprobs_available
        ? "Detailed confidence available"
        : "Detailed confidence unavailable";
      pushActivity("English answer", event.message);
      break;
    case "answer_token":
      answerState.textContent = "Streaming";
      englishAnswerBuffer += event.token;
      renderMarkdownBlock(englishAnswer, englishAnswerBuffer);
      break;
    case "answer_complete":
      markStepDone("answer_in_english");
      answerState.textContent = "Completed";
      englishAnswerBuffer = event.text;
      renderMarkdownBlock(englishAnswer, englishAnswerBuffer);
      renderMetrics(event.metrics);
      renderStatementConfidence(event.analysis?.statement_groups || [], event.analysis?.available, currentRunModel);
      renderTokenDetail(event.analysis?.display_tokens || [], event.analysis?.available, currentRunModel);
      break;
    case "back_translation_skipped":
      markStepDone("translate_answer_back");
      backTranslationState.textContent = "Skipped";
      translatedAnswerBuffer = event.text;
      renderMarkdownBlock(translatedAnswer, translatedAnswerBuffer);
      pushActivity("Answer translation", event.message);
      break;
    case "back_translation_chunk":
      markStepActive("translate_answer_back");
      backTranslationState.textContent = "Translating";
      translatedAnswerBuffer += event.text;
      renderMarkdownBlock(translatedAnswer, translatedAnswerBuffer);
      break;
    case "back_translation_complete":
      markStepDone("translate_answer_back");
      backTranslationState.textContent = "Completed";
      translatedAnswerBuffer = event.text;
      renderMarkdownBlock(translatedAnswer, translatedAnswerBuffer);
      break;
    case "pipeline_analytics":
      if (event.analytics?.available) {
        pipelineAnalyticsState.textContent = event.final ? "Analysis complete" : "Intermediate results available";
      } else {
        pipelineAnalyticsState.textContent =
          currentRunModel === tokenAnalyticsModel
            ? event.final
              ? "Confidence data unavailable"
              : "Waiting for confidence data"
            : `Detailed confidence available with ${tokenAnalyticsModel}`;
      }
      renderPipelineAnalytics(event.analytics || {});
      break;
    case "error":
      pipelineBadge.textContent = "Failed";
      setAllStepStates("error");
      pushActivity("Error", event.message);
      break;
    case "done":
      pushActivity("Completed", "Processing is complete.");
      break;
    default:
      break;
  }
}

function buildTokenNode(event) {
  const token = document.createElement("span");
  token.className = `token-chip ${tokenClass(event.prob)}`;
  token.textContent = event.text;
  token.title = tokenTitle(event);
  return token;
}

function tokenClass(probability) {
  if (typeof probability !== "number") {
    return "none";
  }
  if (probability >= 0.75) {
    return "high";
  }
  if (probability >= 0.4) {
    return "mid";
  }
  return "low";
}

function tokenTitle(event) {
  const lines = [];
  lines.push(`Text: ${JSON.stringify(event.text)}`);
  lines.push(`Probability: ${event.prob_percent || "N/A"}`);
  lines.push(`Lowest token confidence: ${event.min_prob_percent || "N/A"}`);
  lines.push(`Underlying tokens: ${event.token_count ?? "N/A"}`);
  return lines.join("\n");
}

function renderMetrics(metrics = {}) {
  metricAvgProb.textContent = metrics.avg_prob_percent || "N/A";
  metricPerplexity.textContent =
    typeof metrics.perplexity === "number" ? metrics.perplexity.toFixed(3) : "N/A";
  metricLowConfidence.textContent =
    typeof metrics.low_confidence_count === "number"
      ? `${metrics.low_confidence_count} (${((metrics.low_confidence_ratio || 0) * 100).toFixed(1)}%)`
      : "N/A";
  metricTotalTokens.textContent =
    typeof metrics.total_tokens === "number" ? String(metrics.total_tokens) : "N/A";
}

function renderStatementConfidence(groups = [], analysisAvailable, modelName) {
  if (!statementConfidenceState || !statementConfidenceSummary || !statementConfidenceList) {
    return;
  }
  statementConfidenceList.innerHTML = "";

  if (!analysisAvailable) {
    statementConfidenceState.textContent = "Unavailable";
    statementConfidenceSummary.textContent =
      `Detailed statement confidence is currently available only when ${tokenAnalyticsModel} is selected. The answer still ran with ${modelName || currentRunModel}.`;
    statementConfidenceList.innerHTML = `<p class="empty-state">Choose ${escapeHtml(tokenAnalyticsModel)} if you want confidence scores for each full clinical statement.</p>`;
    return;
  }

  if (!groups.length) {
    statementConfidenceState.textContent = "Limited";
    statementConfidenceSummary.textContent =
      "The answer is available, but there was not enough scored text to build the statement-level review.";
    statementConfidenceList.innerHTML = '<p class="empty-state">No statement-level confidence blocks are available for this response.</p>';
    return;
  }

  statementConfidenceState.textContent = "Ready";
  statementConfidenceSummary.textContent =
    "Each row below is a full statement from the English answer. Use it to spot which parts of the recommendation deserve manual review first.";

  for (const group of groups) {
    const item = document.createElement("article");
    const state =
      group.label === "Stable" ? "stable" : group.label === "Review carefully" ? "careful" : "review";
    item.className = "statement-item";
    item.dataset.state = state;
    item.innerHTML = `
      <div class="statement-head">
        <span class="statement-label">${escapeHtml(group.label || "Review")}</span>
        <span class="statement-metrics">Average ${escapeHtml(group.avg_prob_percent || "N/A")} · Lowest phrase ${escapeHtml(group.min_prob_percent || "N/A")}</span>
      </div>
      <p class="statement-text">${escapeHtml(group.text || "")}</p>
      <p class="statement-note">${escapeHtml(group.note || "")}</p>
    `;
    statementConfidenceList.appendChild(item);
  }
}

function renderTokenDetail(groups = [], analysisAvailable, modelName) {
  if (!tokenStream || !tokenLegend) {
    return;
  }
  tokenStream.innerHTML = "";

  if (!analysisAvailable) {
    tokenLegend.innerHTML =
      `Detailed token-level confidence is currently available only when <code>${escapeHtml(tokenAnalyticsModel)}</code> is selected.`;
    tokenStream.innerHTML = `<p class="empty-state">The answer ran with ${escapeHtml(modelName || currentRunModel)}, so word-level confidence detail is not shown for this run.</p>`;
    return;
  }

  tokenLegend.innerHTML =
    "Displayed as word-like chunks for readability. Colors still come from the underlying model tokens, so use this as a fine-detail view rather than the main clinical review.";
  if (!groups.length) {
    tokenStream.innerHTML = '<p class="empty-state">Word-level confidence detail is not available for this response.</p>';
    return;
  }

  for (const group of groups) {
    if (group.text === "\n") {
      tokenStream.appendChild(document.createTextNode("\n"));
      continue;
    }
    tokenStream.appendChild(buildTokenNode(group));
  }
}

function getApiKeyValue() {
  return apiKeyInput ? apiKeyInput.value.trim() : "";
}

function isUsableApiKey(value) {
  const cleaned = String(value || "").trim();
  if (!cleaned) {
    return false;
  }
  return !placeholderApiKeys.has(cleaned.toLowerCase());
}

function syncApiKeyState() {
  const hasApiKey = isUsableApiKey(getApiKeyValue());
  if (apiKeyStatus) {
    apiKeyStatus.textContent = hasApiKey ? "Ready" : "Required";
  }
  syncSubmitButton();
}

function renderPipelineAnalytics(analytics = {}) {
  if (!analytics.available) {
    renderEmptyAnalytics();
    return;
  }

  renderPipelineFigure(analytics);
  renderPerplexityFigure(analytics);
  renderLogicBreakPanel(analytics);
  renderMutationPanel(analytics);
}

function renderPipelineFigure(analytics) {
  const flowPoints = (analytics.flow_points || []).filter(
    (point) => typeof point.mean_prob === "number" && typeof point.global_position === "number"
  );
  const heatmapRows = analytics.heatmap_rows || [];
  if (!flowPoints.length || !heatmapRows.length) {
    pipelineFigure.innerHTML = '<p class="empty-state">There is not enough confidence data to draw this chart yet.</p>';
    return;
  }

  const compactStageLabels = {
    "Input Translation": "Input",
    "English Answer": "Answer",
    "Back-translation": "Return",
  };

  const width = 1600;
  const height = 460;
  const left = 240;
  const right = 44;
  const top = 64;
  const lineHeight = 132;
  const lineBottom = top + lineHeight;
  const heatmapTop = 248;
  const heatmapBottom = 390;
  const usableWidth = width - left - right;
  const rowHeight = Math.max(22, Math.floor((heatmapBottom - heatmapTop) / heatmapRows.length));
  const cellWidth = usableWidth / Math.max((heatmapRows[0]?.bins || []).length, 1);

  const linePoints = flowPoints
    .slice()
    .sort((a, b) => a.global_position - b.global_position)
    .map((point) => ({
      x: left + point.global_position * usableWidth,
      y: lineBottom - point.mean_prob * lineHeight,
      stage: point.label,
      meanProb: point.mean_prob,
    }));

  const linePath = buildSvgPath(linePoints);
  const stageCount = Math.max(analytics.stage_count || 1, 1);
  const stageBands = [];
  const boundaryLines = [];
  const stageLabels = [];
  for (let index = 0; index < stageCount; index += 1) {
    const startX = left + (index / stageCount) * usableWidth;
    const endX = left + ((index + 1) / stageCount) * usableWidth;
    stageBands.push(
      `<rect x="${startX.toFixed(2)}" y="${(top - 18).toFixed(2)}" width="${(endX - startX).toFixed(2)}" height="${(heatmapBottom - top + 34).toFixed(2)}" rx="18" fill="${index % 2 === 0 ? "rgba(255,255,255,0.64)" : "rgba(13,148,136,0.04)"}"></rect>`
    );
    if (index > 0) {
      boundaryLines.push(
        `<line x1="${startX}" y1="${top}" x2="${startX}" y2="${heatmapBottom}" stroke="rgba(146,186,210,0.22)" stroke-dasharray="6 6" />`
      );
    }
    const label = analytics.stages?.filter((stage) => stage.available)[index]?.label || `Stage ${index + 1}`;
    const displayLabel = compactStageLabels[label] || label;
    const labelLines = [displayLabel];
    const longestLineLength = displayLabel.length;
    const stageWidth = endX - startX;
    const pillWidth = Math.min(stageWidth - 30, Math.max(62, longestLineLength * 7.1 + 18));
    const pillHeight = 24;
    const pillY = 14;
    const textY = 30;
    const pillX = ((startX + endX) / 2) - pillWidth / 2;
    const textContent = labelLines
      .map(
        (line, lineIndex) =>
          `<text class="chart-stage-text" x="${((startX + endX) / 2).toFixed(2)}" y="${textY + lineIndex * 12}" text-anchor="middle">${escapeHtml(line)}</text>`
      )
      .join("");
    stageLabels.push(
      `<g>
        <rect class="chart-stage-pill" x="${pillX.toFixed(2)}" y="${pillY}" width="${pillWidth.toFixed(2)}" height="${pillHeight}" rx="12"></rect>
        ${textContent}
      </g>`
    );
  }

  const heatmapRects = heatmapRows.flatMap((row, rowIndex) =>
    (row.bins || []).map((value, columnIndex) => {
      const x = left + columnIndex * cellWidth;
      const y = heatmapTop + rowIndex * rowHeight;
      const color = colorForLogprob(value);
      const stroke = row.logic_break_bins?.includes(columnIndex) ? ' stroke="#ffbf69" stroke-width="2"' : "";
      return `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${cellWidth.toFixed(2)}" height="${(rowHeight - 4).toFixed(2)}" rx="4" fill="${color}"${stroke}></rect>`;
    })
  );

  const rowLabelBadges = heatmapRows
    .map((row, rowIndex) => {
      const y = heatmapTop + rowIndex * rowHeight + rowHeight / 2 + 2;
      const displayLabel = compactStageLabels[row.label] || row.label;
      return `<div class="pipeline-row-badge" style="top: ${((y / height) * 100).toFixed(3)}%;"><span>${escapeHtml(displayLabel)}</span></div>`;
    })
    .join("");

  const breakMarkers = (analytics.logic_break_examples || []).map((event) => {
    const x = left + (event.global_position || 0) * usableWidth;
    const y = top + 8;
    return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="4.5" fill="#ff6d6a"><title>${escapeHtml(
      `${event.stage_label}: ${String(event.token).trim()} | drop ${formatFloat(event.drop_from_prev, 2)}`
    )}</title></circle>`;
  });

  const guideLines = [0.25, 0.5, 0.75].map((tick) => {
    const y = lineBottom - tick * lineHeight;
    return `<line x1="${left}" y1="${y.toFixed(2)}" x2="${width - right}" y2="${y.toFixed(2)}" stroke="rgba(18,79,78,0.08)" stroke-dasharray="5 8" />`;
  });

  pipelineFigure.innerHTML = `
    <div class="pipeline-figure-shell">
      <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Full pipeline confidence figure">
        <rect x="0" y="0" width="${width}" height="${height}" rx="24" fill="#f7fbf9" stroke="#dbe6e2"></rect>
        <text class="chart-label" x="${left}" y="54">Workflow confidence trend</text>
        <text class="chart-note" x="${left}" y="${lineBottom + 26}">Top: confidence trend as each stage progresses. Bottom: stage heatmaps. Amber markers flag notable drops that may deserve manual review.</text>
        ${stageBands.join("")}
        ${guideLines.join("")}
        <line x1="${left}" y1="${lineBottom}" x2="${width - right}" y2="${lineBottom}" stroke="rgba(22,47,39,0.18)" />
        <line x1="${left}" y1="${top}" x2="${left}" y2="${lineBottom}" stroke="rgba(22,47,39,0.18)" />
        <text class="chart-note" x="${left - 16}" y="${top + 4}" text-anchor="end">1.0</text>
        <text class="chart-note" x="${left - 16}" y="${lineBottom + 4}" text-anchor="end">0.0</text>
        ${boundaryLines.join("")}
        ${stageLabels.join("")}
        <path d="${linePath}" fill="none" stroke="#169f93" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></path>
        ${linePoints
          .map(
            (point) =>
              `<circle cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="3.2" fill="#ffbf69" stroke="#fff7e8" stroke-width="1.5"><title>${escapeHtml(
                `${point.stage}: ${(point.meanProb * 100).toFixed(2)}%`
              )}</title></circle>`
          )
          .join("")}
        ${breakMarkers.join("")}
        ${heatmapRects.join("")}
      </svg>
      <div class="pipeline-row-labels" aria-hidden="true">${rowLabelBadges}</div>
    </div>
  `;
}

function renderPerplexityFigure(analytics) {
  const activeStages = (analytics.stages || []).filter((stage) => stage.available);
  if (!activeStages.length) {
    perplexityFigure.innerHTML = '<p class="empty-state">No token uncertainty curve is available yet.</p>';
    if (perplexityState) perplexityState.textContent = "No data";
    return;
  }

  const width = 1680;
  const height = 560;
  const left = 92;
  const right = 42;
  const top = 74;
  const bottom = 86;
  const usableWidth = width - left - right;
  const usableHeight = height - top - bottom;
  const colors = ["#0d9488", "#f59e0b", "#ef4444", "#6366f1"];
  const maxPerplexity = Math.max(
    2,
    ...activeStages.flatMap((stage) =>
      (stage.bucket_summary || [])
        .map((bucket) => bucket.mean_perplexity)
        .filter((value) => typeof value === "number")
    )
  );

  const seriesData = activeStages.map((stage, stageIndex) => {
    const buckets = (stage.bucket_summary || []).filter((bucket) => typeof bucket.mean_perplexity === "number");
    if (!buckets.length) return null;
    const points = buckets.map((bucket) => ({
      x: left + ((bucket.bin + 0.5) / buckets.length) * usableWidth,
      y: top + usableHeight - (Math.min(bucket.mean_perplexity, maxPerplexity) / maxPerplexity) * usableHeight,
    }));
    return { points, color: colors[stageIndex % colors.length], label: stage.label };
  }).filter(Boolean);

  const areaSvg = seriesData.map(({ points, color }) => {
    if (points.length < 2) return "";
    const areaPath = buildSvgPath(points)
      + ` L ${points[points.length - 1].x.toFixed(2)} ${(top + usableHeight).toFixed(2)}`
      + ` L ${points[0].x.toFixed(2)} ${(top + usableHeight).toFixed(2)} Z`;
    return `<path d="${areaPath}" fill="${color}" opacity="0.07"></path>`;
  }).join("");

  const seriesSvg = seriesData.map(({ points, color }) => {
    if (points.length < 2) return "";
    return `<path d="${buildSvgPath(points)}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>`;
  }).join("");

  const dotSvg = seriesData.map(({ points, color, label }) => {
    return points.map((p) =>
      `<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="4" fill="${color}" opacity="0.7"><title>${escapeHtml(label)}</title></circle>`
    ).join("");
  }).join("");

  const legendSvg = seriesData.map(({ color, label }, i) =>
    `<rect x="${left + i * 220}" y="${height - 36}" width="14" height="14" rx="4" fill="${color}" opacity="0.9"></rect>` +
    `<text class="chart-note" x="${left + i * 220 + 22}" y="${height - 24}">${escapeHtml(label)}</text>`
  ).join("");

  const gridLines = [0, 0.25, 0.5, 0.75, 1].map((tick) => {
    const y = top + usableHeight - tick * usableHeight;
    return `<line x1="${left}" y1="${y.toFixed(2)}" x2="${width - right}" y2="${y.toFixed(2)}" stroke="rgba(18,79,78,${tick === 0 ? "0.16" : "0.08"})" stroke-dasharray="${tick === 0 ? "0" : "5 10"}"></line>`;
  }).join("");

  perplexityFigure.innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Token uncertainty curves by stage">
      <rect x="0" y="0" width="${width}" height="${height}" rx="28" fill="#fbfdfc" stroke="#dce8e3"></rect>
      ${gridLines}
      <line x1="${left}" y1="${top + usableHeight}" x2="${width - right}" y2="${top + usableHeight}" stroke="var(--border)" />
      <line x1="${left}" y1="${top}" x2="${left}" y2="${top + usableHeight}" stroke="var(--border)" />
      <text class="chart-note" x="${left - 12}" y="${top + 6}" text-anchor="end">${formatFloat(maxPerplexity, 1)}</text>
      <text class="chart-note" x="${left - 12}" y="${top + usableHeight + 6}" text-anchor="end">1.0</text>
      <text class="chart-label" x="${left}" y="${top - 26}">Token uncertainty within each stage</text>
      <text class="chart-note" x="${left}" y="${top - 6}">Each colored line runs from the start to the end of one stage. Higher curves indicate lower certainty.</text>
      ${areaSvg}
      ${seriesSvg}
      ${dotSvg}
      ${legendSvg}
      <text class="chart-note" x="${left}" y="${height - 54}">Start of that stage</text>
      <text class="chart-note" x="${(left + (width - right)) / 2}" y="${height - 54}" text-anchor="middle">Relative progress within the stage</text>
      <text class="chart-note" x="${width - right}" y="${height - 54}" text-anchor="end">End of that stage</text>
    </svg>
  `;
  if (perplexityState) perplexityState.textContent = "Completed";
  const overallPerplexity = analytics.overall_summary?.overall_perplexity;
  perplexitySummary.textContent =
    typeof overallPerplexity === "number"
      ? `The overall uncertainty score for this workflow is about ${overallPerplexity.toFixed(3)}. Read the horizontal axis as start-to-end progress within each stage, not as a clinical timeline.`
      : "Read the horizontal axis as start-to-end progress within each stage, not as a clinical timeline.";
}

function renderLogicBreakPanel(analytics) {
  const summary = analytics.overall_summary || {};
  const examples = analytics.logic_break_examples || [];
  logicBreakCount.textContent =
    typeof summary.logic_break_count === "number" ? String(summary.logic_break_count) : "-";
  logicBreakRate.textContent =
    typeof summary.logic_break_rate === "number" ? `${(summary.logic_break_rate * 100).toFixed(2)}%` : "-";
  logicBreakState.textContent = examples.length ? "Drops detected" : "No clear drops";
  logicBreakList.innerHTML = "";

  if (!examples.length) {
    logicBreakList.innerHTML = '<p class="empty-state">No notable confidence drops were detected in this run.</p>';
    return;
  }

  for (const event of examples.slice(0, 6)) {
    const node = document.createElement("div");
    node.className = "event-card";
    node.innerHTML = `
      <strong>${escapeHtml(event.stage_label)} · token ${escapeHtml(String(event.token).trim() || "(blank)")}</strong>
      <span>Position ${event.position}, confidence drop ${formatFloat(event.drop_from_prev, 2)}</span>
    `;
    logicBreakList.appendChild(node);
  }
}

function renderMutationPanel(analytics) {
  const summary = analytics.overall_summary || {};
  const stages = analytics.mutation_by_stage || [];
  mutationCount.textContent =
    typeof summary.mutation_count === "number" ? String(summary.mutation_count) : "-";
  mutationRate.textContent =
    typeof summary.mutation_rate === "number" ? `${(summary.mutation_rate * 100).toFixed(2)}%` : "-";
  mutationState.textContent = stages.length ? "Completed" : "No data";

  if (!stages.length) {
    mutationFigure.innerHTML = '<p class="empty-state">No large confidence shifts are available yet.</p>';
    return;
  }

  const maxMutation = Math.max(...stages.map((stage) => stage.mutation_count || 0), 1);
  const wrapper = document.createElement("div");
  wrapper.className = "bar-stack";
  for (const stage of stages) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-label">${escapeHtml(stage.label)}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${((stage.mutation_count || 0) / maxMutation) * 100}%"></div>
      </div>
      <span class="bar-meta">${stage.mutation_count || 0} shifts, rate ${(stage.mutation_rate * 100).toFixed(2)}%, larger shifts ${stage.high_magnitude_count || 0}</span>
    `;
    wrapper.appendChild(row);
  }
  mutationFigure.innerHTML = "";
  mutationFigure.appendChild(wrapper);
  mutationSummary.textContent =
    "This panel counts large confidence shifts in each stage and helps highlight where the answer becomes less stable.";
}

function renderEmptyAnalytics() {
  pipelineFigure.innerHTML = '<p class="empty-state">There is not enough confidence data to draw this chart yet.</p>';
  perplexityFigure.innerHTML = '<p class="empty-state">No token uncertainty curve is available yet.</p>';
  logicBreakList.innerHTML = '<p class="empty-state">No confidence drop events are available yet.</p>';
  mutationFigure.innerHTML = '<p class="empty-state">No large confidence shifts are available yet.</p>';
  if (perplexityState) perplexityState.textContent = "No data";
  if (perplexitySummary) {
    perplexitySummary.textContent =
      `Read the horizontal axis as start-to-end progress within each stage, not as a clinical timeline. Detailed charting is available when ${tokenAnalyticsModel} is selected.`;
  }
  logicBreakState.textContent = "No data";
  mutationState.textContent = "No data";
  logicBreakCount.textContent = "-";
  logicBreakRate.textContent = "-";
  mutationCount.textContent = "-";
  mutationRate.textContent = "-";
}

function buildSvgPath(points) {
  if (!Array.isArray(points) || !points.length) {
    return "";
  }
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");
}

function colorForLogprob(value) {
  if (typeof value !== "number") {
    return "rgba(146, 186, 210, 0.12)";
  }
  const minLogprob = -4.5;
  const maxLogprob = 0;
  const normalized = clamp((value - minLogprob) / (maxLogprob - minLogprob), 0, 1);
  const red = Math.round(255 - normalized * 154);
  const green = Math.round(109 + normalized * 99);
  const blue = Math.round(106 + normalized * 92);
  return `rgb(${red}, ${green}, ${blue})`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatFloat(value, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "N/A";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderMarkdownBlock(element, markdown) {
  if (!element) {
    return;
  }
  const source = markdown || "";
  element.dataset.rawText = source;
  element.innerHTML = source ? renderMarkdown(source) : "";
}

function renderMarkdown(markdown) {
  const normalized = String(markdown).replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return "";
  }

  const lines = normalized.split("\n");
  const html = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const fence = trimmed.slice(0, 3);
      const language = trimmed.slice(3).trim();
      index += 1;
      const codeLines = [];
      while (index < lines.length && !lines[index].trim().startsWith(fence)) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      html.push(
        `<pre><code${language ? ` data-language="${escapeHtml(language)}"` : ""}>${escapeHtml(codeLines.join("\n"))}</code></pre>`
      );
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      html.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^>\s?/.test(trimmed)) {
      const quoteLines = [];
      while (index < lines.length && /^>\s?/.test(lines[index].trim())) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ""));
        index += 1;
      }
      html.push(`<blockquote>${quoteLines.map((item) => `<p>${renderInlineMarkdown(item)}</p>`).join("")}</blockquote>`);
      continue;
    }

    if (/^[-*+]\s+/.test(trimmed)) {
      const items = [];
      while (index < lines.length && /^[-*+]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*+]\s+/, ""));
        index += 1;
      }
      html.push(`<ul>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+\.\s+/, ""));
        index += 1;
      }
      html.push(`<ol>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ol>`);
      continue;
    }

    if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
      html.push("<hr />");
      index += 1;
      continue;
    }

    const paragraphLines = [];
    while (index < lines.length) {
      const candidate = lines[index];
      const candidateTrimmed = candidate.trim();
      if (!candidateTrimmed) {
        index += 1;
        break;
      }
      if (
        candidateTrimmed.startsWith("```") ||
        /^(#{1,6})\s+/.test(candidateTrimmed) ||
        /^>\s?/.test(candidateTrimmed) ||
        /^[-*+]\s+/.test(candidateTrimmed) ||
        /^\d+\.\s+/.test(candidateTrimmed) ||
        /^---+$/.test(candidateTrimmed) ||
        /^\*\*\*+$/.test(candidateTrimmed)
      ) {
        break;
      }
      paragraphLines.push(candidate);
      index += 1;
    }

    html.push(`<p>${renderInlineMarkdown(paragraphLines.join("\n"))}</p>`);
  }

  return html.join("");
}

function renderInlineMarkdown(text) {
  let html = escapeHtml(text).replace(/\n/g, "<br />");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(/_([^_]+)_/g, "<em>$1</em>");
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  return html;
}

function resetOutputs() {
  translatedQuestionBuffer = "";
  translatedAnswerBuffer = "";
  englishAnswerBuffer = "";
  currentRunModel = getSelectedModelValue();
  renderMarkdownBlock(translatedQuestion, "");
  renderMarkdownBlock(translatedAnswer, "");
  renderMarkdownBlock(englishAnswer, "");
  tokenStream.innerHTML = "";
  activityFeed.innerHTML = "";
  languageBadge.textContent = "Not checked";
  languageRationale.textContent = "After submission, this area will show whether the input is English and the model rationale for that decision.";
  translationState.textContent = "Idle";
  backTranslationState.textContent = "Idle";
  answerState.textContent = "Idle";
  statementConfidenceState.textContent = "Waiting for answer";
  statementConfidenceSummary.textContent =
    "This review groups the English answer into full clinical statements so you can quickly see which parts look stable and which parts deserve manual review.";
  statementConfidenceList.innerHTML =
    '<p class="empty-state">After the answer is ready, full clinical statements will be listed here with a simple review label.</p>';
  logprobState.textContent = "Waiting for answer";
  tokenLegend.innerHTML =
    `Displayed as word-like chunks for readability. Detailed token-level confidence is currently available only when <code>${escapeHtml(tokenAnalyticsModel)}</code> is selected.`;
  metricAvgProb.textContent = "-";
  metricPerplexity.textContent = "-";
  metricLowConfidence.textContent = "-";
  metricTotalTokens.textContent = "-";
  pipelineAnalyticsState.textContent = "Waiting for completion";
  pipelineBadge.textContent = "Ready";
  renderEmptyAnalytics();
  setAllStepStates("idle");
  if (advancedReview) {
    advancedReview.open = false;
  }
  syncSubmitButton();
}

function hideResultsShell() {
  if (resultsShell) {
    resultsShell.classList.add("is-prerun");
  }
  if (advancedReview) {
    advancedReview.open = false;
  }
}

function revealResultsShell() {
  if (resultsShell) {
    resultsShell.classList.remove("is-prerun");
  }
}

function renderIdleProcessingLog() {
  if (!activityFeed || activityFeed.childElementCount) {
    return;
  }
  pushActivity("Ready to start", "Add your API key, enter the clinical question, and click Run analysis.");
}

function setBusy(isBusy) {
  isSubmitting = isBusy;
  syncSubmitButton();
}

function syncSubmitButton() {
  if (!submitButton) {
    return;
  }
  const missingApiKey = requiresUserApiKey && !isUsableApiKey(getApiKeyValue());
  submitButton.disabled = isSubmitting || missingApiKey;
  if (isSubmitting) {
    submitButton.dataset.state = "running";
    submitButton.textContent = "Running...";
    return;
  }
  if (missingApiKey) {
    submitButton.dataset.state = "api";
    submitButton.textContent = "Add API key to continue";
    return;
  }
  submitButton.dataset.state = "idle";
  submitButton.textContent = "Run analysis";
}

function markStepActive(stage) {
  for (const [key, node] of Object.entries(stepChips)) {
    if (!node) {
      continue;
    }
    if (key === stage) {
      node.classList.remove("done", "error");
      node.classList.add("active");
    } else if (!node.classList.contains("done")) {
      node.classList.remove("active", "error");
    }
  }
}

function markStepDone(stage) {
  const node = stepChips[stage];
  if (!node) {
    return;
  }
  node.classList.remove("active", "error");
  node.classList.add("done");
}

function setAllStepStates(state) {
  for (const node of Object.values(stepChips)) {
    node.classList.remove("active", "done", "error");
    if (state === "error") {
      node.classList.add("error");
    }
  }
}

function pushActivity(title, message) {
  const node = activityItemTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".activity-title").textContent = title;
  node.querySelector(".activity-message").textContent = message;
  activityFeed.prepend(node);
}

async function readErrorMessage(response) {
  const text = await response.text();
  if (!text) {
    return "Request failed.";
  }
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch (error) {
    // fall through to raw text
  }
  return text;
}

function stageLabel(stage) {
  const mapping = {
    detect_language: "Language detection",
    translate_input: "Input translation",
    answer_in_english: "English answer",
    translate_answer_back: "Answer translation",
  };
  return mapping[stage] || stage;
}

async function fileToPayload(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      resolve({
        name: file.name,
        media_type: file.type || "image/png",
        data_url: reader.result,
      });
    reader.onerror = () => reject(new Error(`Unable to read image: ${file.name}`));
    reader.readAsDataURL(file);
  });
}

function renderImagePreview() {
  imagePreview.innerHTML = "";
  for (const image of selectedImages) {
    const tile = document.createElement("div");
    tile.className = "image-tile";

    const img = document.createElement("img");
    img.src = image.data_url;
    img.alt = image.name;

    const caption = document.createElement("span");
    caption.textContent = image.name;

    tile.append(img, caption);
    imagePreview.appendChild(tile);
  }
}

function updateImagePickerStatus(files = []) {
  if (!files.length) {
    imagePickerStatus.textContent = "No files selected";
    return;
  }
  if (files.length === 1) {
    imagePickerStatus.textContent = files[0].name;
    return;
  }
  imagePickerStatus.textContent = `${files.length} files selected`;
}
