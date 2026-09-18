"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const percent = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";
const fixed = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "-";
const state = { overview: null, source: "example", example: null, exampleNumber: 0, file: null,
  objectURL: null, tab: "search", result: null, comparison: null, selected: new Set(), epoch: 0,
  busy: false, collectionCategory: "all", collectionPage: 1, collectionPages: 1, collectionEpoch: 0 };

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); }
  catch { throw new Error("The local server returned an unexpected response. Please restart run.bat."); }
  if (!response.ok) throw new Error(data.error || "Something went wrong. Please try again.");
  return data;
}

function showError(error) {
  $("#app-error").textContent = error.message || String(error);
  $("#app-error").hidden = false;
}
function clearError() { $("#app-error").hidden = true; }
let toastTimer;
function toast(message) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 4000);
}
function queryReady() { return Boolean(state.source === "example" ? state.example : state.file); }
function setBusy(busy) {
  state.busy = busy;
  $("#search-button").disabled = busy || !queryReady();
  $("#search-button span:first-child").textContent = busy ? "Finding connections…" : state.tab === "compare" ? "Compare both methods" : "Find similar images";
  $("#search-button").classList.toggle("busy", busy);
  $("#reset-feedback").disabled = busy;
  $("#refine-button").disabled = busy || !state.selected.size;
}

function invalidate() {
  state.epoch += 1;
  state.result = null;
  state.comparison = null;
  state.selected.clear();
  $("#search-results").replaceChildren();
  $("#search-empty").hidden = false;
  $("#search-status").textContent = "";
  $("#compare-status").textContent = "";
  $("#compare-results").innerHTML = '<div class="empty-state compact"><h4>Let the methods speak.</h4><p>Choose an image and compare both searches.</p></div>';
  $("#feedback-bar").hidden = true;
  $("#export-results").hidden = true;
  $("#score-note").hidden = true;
  $("#results-title").textContent = "A little resemblance awaits.";
  clearError();
  setBusy(false);
}

function setSource(source) {
  state.source = source;
  $$("[data-source]").forEach((button) => {
    const active = button.dataset.source === source;
    button.classList.toggle("selected", active);
    button.setAttribute("aria-pressed", String(active));
  });
  $("#example-controls").hidden = source !== "example";
  $("#upload-controls").hidden = source !== "upload" || Boolean(state.file);
  $("#next-example").hidden = source !== "example";
  $("#change-upload").hidden = source !== "upload" || !state.file;
  invalidate();
  renderQuery();
}

function renderQuery() {
  const query = state.source === "example" ? state.example : state.file;
  const preview = $("#query-preview");
  preview.replaceChildren();
  if (!query) {
    preview.hidden = true;
    $("#query-name").textContent = "Choose an image to begin";
    setBusy(false);
    return;
  }
  preview.hidden = false;
  const image = document.createElement("img");
  image.src = state.source === "example" ? state.example.image_url : state.objectURL;
  image.alt = state.source === "example" ? `Query image: ${state.example.label}` : `Uploaded query: ${state.file.name}`;
  preview.append(image);
  $("#query-name").textContent = state.source === "example" ? `${state.example.label} · example ${state.exampleNumber + 1}` : state.file.name;
  setBusy(false);
}

function chooseExample(reset = true) {
  if (!state.overview) return;
  const examples = state.overview.examples.filter((item) => item.category === $("#example-category").value);
  state.exampleNumber = reset ? 0 : (state.exampleNumber + 1) % examples.length;
  state.example = examples[state.exampleNumber];
  invalidate();
  renderQuery();
}

async function chooseFile(file) {
  if (!file) return;
  clearError();
  if (file.size > 10 * 1024 * 1024) { showError(new Error("Please choose an image smaller than 10 MB.")); return; }
  if (!/\.(jpe?g|png|webp)$/i.test(file.name)) { showError(new Error("Choose a JPG, PNG, or WebP image.")); return; }
  const url = URL.createObjectURL(file);
  try {
    const probe = new Image();
    probe.src = url;
    await probe.decode();
    if (probe.naturalWidth * probe.naturalHeight > 20000000 || Math.max(probe.naturalWidth, probe.naturalHeight) > 20000 || Math.max(probe.naturalWidth, probe.naturalHeight) / Math.min(probe.naturalWidth, probe.naturalHeight) > 50) {
      throw new Error("Choose a conventional photo with no more than 20 million pixels.");
    }
    if (state.objectURL) URL.revokeObjectURL(state.objectURL);
    state.file = file;
    state.objectURL = url;
    setSource("upload");
  } catch (error) {
    URL.revokeObjectURL(url);
    showError(new Error(error.message.includes("million") ? error.message : "This file could not be read as an image. Try another photo."));
  }
}

function renderCard(item, selectable = false, showScore = true) {
  const selected = state.selected.has(item.index_id);
  const label = escapeHTML(item.label);
  return `<article class="image-card${selectable && selected ? " selected" : ""}">
    ${item.rank ? `<span class="rank${item.rank === 1 ? " rank-best" : ""}">#${item.rank}${item.rank === 1 ? " · Best match" : ""}</span>` : ""}
    <button class="card-image-button" data-preview="${escapeHTML(item.image_url)}" data-label="${label}" data-score="${showScore && Number.isFinite(item.score) ? item.score : ""}" aria-label="Preview ${label}${item.rank ? ` match ${item.rank}` : ""}"><img loading="lazy" src="${escapeHTML(item.thumbnail_url)}" alt="${label}"></button>
    ${selectable ? `<button class="select-match" data-select="${item.index_id}" aria-label="Select match ${item.rank} as relevant" aria-pressed="${selected}">${selected ? "✓" : "+"}</button>` : ""}
    <div class="card-caption"><div><strong>${label}</strong><small>${showScore ? "SIMILARITY" : escapeHTML(item.filename)}</small></div>${showScore ? `<span class="score" title="Cosine similarity: ${fixed(item.score)}">${fixed(item.score * 100, 1)}%</span>` : ""}</div>
    ${showScore ? `<meter class="similarity-meter" min="0" max="100" value="${Math.max(0, Math.min(100, item.score * 100))}" aria-label="Positive similarity for match ${item.rank}">${fixed(item.score * 100, 1)}%</meter>` : ""}
  </article>`;
}

function updateFeedback() {
  const count = state.selected.size;
  $("#selection-count").textContent = `${count} image${count === 1 ? "" : "s"} selected`;
  $("#refine-button").disabled = !count || state.busy;
  $("#reset-feedback").hidden = !state.result?.refined;
}

function renderSearch(result) {
  state.result = result;
  state.selected.clear();
  $("#search-empty").hidden = true;
  $("#results-title").textContent = `${result.results.length} ways to see the resemblance.`;
  $("#search-status").innerHTML = `${escapeHTML(result.method_label)}<span class="dot-separator">·</span>${fixed(result.total_ms, 1)} ms${result.refined ? '<span class="dot-separator">·</span>Refined with your feedback' : ""}`;
  $("#search-results").innerHTML = result.results.map((item) => renderCard(item, true)).join("");
  $("#export-results").hidden = false;
  $("#feedback-bar").hidden = false;
  $("#score-note").hidden = false;
  updateFeedback();
}

function renderComparison(searches) {
  state.comparison = searches;
  $("#compare-status").textContent = "Same query. Same gallery. Scores belong to different feature spaces and should not be compared directly.";
  $("#compare-results").innerHTML = searches.map((result) => `<section><div class="method-heading"><h4>${result.method === "clip" ? "Visual meaning" : "Color & tone"}</h4><p>${result.method === "clip" ? "Pretrained CLIP" : "HSV histogram"} · ${fixed(result.total_ms, 1)} ms</p>${result.precision !== null ? `<span class="precision">${percent(result.precision)} Precision@${result.results.length} for this query</span>` : ""}</div><div class="results-grid">${result.results.map((item) => renderCard(item)).join("")}</div></section>`).join("");
}

async function search() {
  if (!queryReady() || state.busy) return;
  clearError();
  const epoch = ++state.epoch;
  const compare = state.tab === "compare";
  const status = compare ? $("#compare-status") : $("#search-status");
  setBusy(true);
  status.innerHTML = '<span class="spinner" aria-hidden="true"></span>Finding the closest visual connections…';
  if (!compare) {
    $("#search-empty").hidden = true;
    $("#search-results").innerHTML = Array.from({ length: 6 }, () => '<div class="skeleton" aria-hidden="true"></div>').join("");
    $("#feedback-bar").hidden = true;
    $("#export-results").hidden = true;
  }
  const form = new FormData();
  form.append("method", compare ? "both" : $("#search-method").value);
  form.append("k", $("#result-count").value);
  if (state.source === "example") form.append("query_id", state.example.id);
  else form.append("image", state.file);
  try {
    const data = await api("/api/search", { method: "POST", body: form });
    if (epoch !== state.epoch) return;
    compare ? renderComparison(data.searches) : renderSearch(data.searches[0]);
  } catch (error) {
    if (epoch !== state.epoch) return;
    status.textContent = "Search couldn't finish. Please try again.";
    if (!compare) {
      $("#search-results").replaceChildren();
      $("#search-empty").hidden = false;
    }
    showError(error);
  } finally { if (epoch === state.epoch) setBusy(false); }
}

async function feedback(reset = false) {
  if (!state.result || state.busy) return;
  clearError();
  const epoch = ++state.epoch;
  setBusy(true);
  updateFeedback();
  $("#reset-feedback").disabled = true;
  try {
    const result = await api("/api/feedback", { method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ token: state.result.token, selected: [...state.selected], reset }) });
    if (epoch === state.epoch) { renderSearch(result); toast(reset ? "Back to your original image." : "A new perspective, guided by your selections."); }
  } catch (error) { if (epoch === state.epoch) showError(error); }
  finally {
    if (epoch === state.epoch) { setBusy(false); updateFeedback(); $("#reset-feedback").disabled = false; }
  }
}

function setTab(tab) {
  state.tab = tab;
  clearError();
  $$("[data-tab]").forEach((button) => {
    const active = button.dataset.tab === tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  for (const name of ["search", "compare", "evaluation", "collection"]) $("#panel-" + name).hidden = tab !== name;
  $("#search-workspace").hidden = !["search", "compare"].includes(tab);
  $("#method-control").hidden = tab === "compare";
  setBusy(state.busy);
  if (tab === "evaluation") loadEvaluation();
  if (tab === "collection" && state.overview) loadCollection();
}

function chartBar(value, maximum, baseline = false, suffix = "%") {
  return `<div class="bar-row"><div class="bar-track${baseline ? " baseline" : ""}"><progress max="${maximum}" value="${value}" aria-label="${fixed(value, 1)}${suffix}"></progress></div><span>${fixed(value, 1)}${suffix}</span></div>`;
}

function renderEvaluation(report) {
  const container = $("#evaluation-content");
  if (!report.available) { container.innerHTML = `<div class="empty-state compact"><h4>Let’s put it to the test.</h4><p>${escapeHTML(report.reason)}</p></div>`; return; }
  const clip = report.summaries.find((item) => item.method === "clip");
  const baseline = report.summaries.find((item) => item.method === "histogram");
  const metrics = [["CLIP · PRECISION@10", percent(clip?.precision_at_10), "Category-relevant matches"],
    ["COLOR · PRECISION@10", percent(baseline?.precision_at_10), "The traditional baseline"],
    ["HELD-OUT QUERIES", report.query_count, "Never included in the gallery"],
    ["CLIP · MEAN QUERY TIME", `${fixed(clip?.total_ms, 1)} ms`, "After model warm-up"]];
  const quality = [["Precision@5", "precision_at_5"], ["Precision@10", "precision_at_10"], ["mAP@10", "map_at_10"]];
  const maxLatency = Math.max(...report.summaries.map((row) => row.total_ms)) * 1.1;
  const categoryRows = state.overview.categories.map((category) => {
    const a = report.per_category.find((row) => row.method === "clip" && row.category === category.id);
    const b = report.per_category.find((row) => row.method === "histogram" && row.category === category.id);
    return `<tr><td>${escapeHTML(category.label)}</td><td>${percent(a?.precision_at_10)}</td><td>${percent(b?.precision_at_10)}</td></tr>`;
  }).join("");
  container.innerHTML = `<div class="metric-cards">${metrics.map(([title, value, note]) => `<article class="metric-card"><span>${title}</span><strong>${value}</strong><small>${note}</small></article>`).join("")}</div>
    <div class="chart-grid"><article class="chart-card"><h4>The quality of a match.</h4><p>Average relevance across all held-out queries.</p><div class="chart-key"><span>CLIP</span><span>Color histogram</span></div>${quality.map(([name, key]) => `<div class="chart-group"><span>${name}</span>${chartBar((clip?.[key] || 0) * 100, 100)}${chartBar((baseline?.[key] || 0) * 100, 100, true)}</div>`).join("")}</article>
    <article class="chart-card"><h4>A closer look at speed.</h4><p>Decode + feature extraction + exact vector search.</p>${report.summaries.map((row) => `<div class="chart-group"><span>${escapeHTML(row.name)}</span>${chartBar(row.total_ms, maxLatency, row.method === "histogram", " ms")}</div>`).join("")}<div class="table-wrap"><table><thead><tr><th>Method</th><th>Encode</th><th>Search</th></tr></thead><tbody>${report.summaries.map((row) => `<tr><td>${row.method === "clip" ? "CLIP" : "Color"}</td><td>${fixed(row.encode_ms, 2)} ms</td><td>${fixed(row.search_ms, 2)} ms</td></tr>`).join("")}</tbody></table></div><p>Model loading and interface rendering are excluded.</p></article></div>
    <div class="table-wrap"><table><thead><tr><th>Method</th><th>P@5</th><th>P@10</th><th>Recall@10</th><th>mAP@10</th><th>MRR@10</th><th>NDCG@10</th></tr></thead><tbody>${report.summaries.map((row) => `<tr><td>${escapeHTML(row.name)}</td>${["precision_at_5", "precision_at_10", "recall_at_10", "map_at_10", "mrr_at_10", "ndcg_at_10"].map((key) => `<td>${fixed(row[key])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <details class="methodology"><summary>Explore performance by category <span>+</span></summary><div class="table-wrap"><table><thead><tr><th>Category</th><th>CLIP · P@10</th><th>Color · P@10</th></tr></thead><tbody>${categoryRows}</tbody></table></div></details>
    <details class="methodology"><summary>What these measurements mean <span>+</span></summary><p><b>Relevance</b> means belonging to the same category as the query. Each query has 40 relevant gallery images.</p><ul><li><b>Precision@K:</b> relevant images in the first K results, divided by K.</li><li><b>Recall@10:</b> relevant images in the first 10 results, divided by 40. The maximum here is 0.25.</li><li><b>mAP@10:</b> average AP@10, with AP divided by min(total relevant, 10).</li><li><b>MRR@10:</b> reciprocal rank of the first relevant image within the top 10, otherwise zero.</li><li><b>NDCG@10:</b> relevance discounted by rank and normalized against an ideal ranking.</li></ul><p>This is a small educational subset, not an official Caltech benchmark. Category labels approximate visual relevance. Exact file duplicates are excluded; near duplicates and CLIP pretraining overlap have not been exhaustively checked. Evaluation uses original queries without user feedback.</p></details>
    <div class="download-row"><a class="outline-button" href="/api/download/summary">Download summary ↓</a><a class="outline-button" href="/api/download/queries">All query measurements ↓</a><span class="fine-print">Measured ${escapeHTML(new Date(report.evaluated_at).toLocaleString())}</span></div>`;
}

async function loadEvaluation() {
  if (!state.overview) return;
  $("#evaluation-status").textContent = "Loading measurements…";
  try { renderEvaluation(await api("/api/evaluation")); $("#evaluation-status").textContent = ""; }
  catch (error) { $("#evaluation-status").textContent = "Could not load evaluation."; showError(error); }
}

async function runEvaluation() {
  clearError();
  $("#run-evaluation").disabled = true;
  $("#evaluation-status").innerHTML = '<span class="spinner" aria-hidden="true"></span>Running 100 queries through both methods. This may take a few seconds…';
  try { renderEvaluation(await api("/api/evaluation", { method: "POST" })); $("#evaluation-status").textContent = "Evaluation complete. These results were measured on this computer."; }
  catch (error) { $("#evaluation-status").textContent = "Evaluation could not finish."; showError(error); }
  finally { $("#run-evaluation").disabled = false; }
}

async function loadCollection() {
  const epoch = ++state.collectionEpoch;
  $("#collection-status").textContent = "Opening the collection…";
  try {
    const parameters = new URLSearchParams({ category: state.collectionCategory, split: $("#collection-split").value, page: state.collectionPage });
    const data = await api("/api/collection?" + parameters);
    if (epoch !== state.collectionEpoch) return;
    state.collectionPages = data.pages;
    $("#collection-images").innerHTML = data.images.map((item) => renderCard(item, false, false)).join("");
    $("#collection-status").textContent = `${data.total} images · ${$("#collection-split").value === "gallery" ? "Included in the search index" : "Reserved for evaluation"}`;
    $("#page-label").textContent = `${data.page} / ${data.pages}`;
    $("#previous-page").disabled = data.page === 1;
    $("#next-page").disabled = data.page === data.pages;
  } catch (error) { if (epoch === state.collectionEpoch) { showError(error); $("#collection-status").textContent = "Could not open this collection page."; } }
}

async function rebuild(method, button) {
  clearError();
  $$("[data-build]").forEach((item) => { item.disabled = true; });
  $("#build-status").textContent = `Rebuilding the ${method === "clip" ? "CLIP" : "color"} index…`;
  try {
    const result = await api(`/api/indexes/${method}/build`, { method: "POST" });
    invalidate();
    $("#build-status").textContent = `${result.message} Run evaluation to refresh the measurements.`;
  } catch (error) { $("#build-status").textContent = error.message; showError(error); }
  finally { $$("[data-build]").forEach((item) => { item.disabled = false; }); }
}

function exportResults() {
  if (!state.result) return;
  const quote = (value) => `"${String(value).replaceAll('"', '""')}"`;
  const rows = [["rank", "category", "cosine_similarity", "filename"], ...state.result.results.map((row) => [row.rank, row.category, row.score, row.filename])];
  const blob = new Blob([rows.map((row) => row.map(quote).join(",")).join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "lookalike-results.csv";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$$("[data-source]").forEach((button) => button.addEventListener("click", () => setSource(button.dataset.source)));
$$("[data-tab]").forEach((button) => button.addEventListener("click", () => setTab(button.dataset.tab)));
$(".workspace-nav").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const tabs = $$("[data-tab]");
  const index = tabs.indexOf(document.activeElement);
  const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  tabs[next].focus(); setTab(tabs[next].dataset.tab);
});
$("#example-category").addEventListener("change", () => chooseExample());
$("#next-example").addEventListener("click", () => chooseExample(false));
$("#search-method").addEventListener("change", invalidate);
$("#result-count").addEventListener("change", invalidate);
$("#search-button").addEventListener("click", search);
$("#refine-button").addEventListener("click", () => feedback());
$("#reset-feedback").addEventListener("click", () => feedback(true));
$("#export-results").addEventListener("click", exportResults);
$("#run-evaluation").addEventListener("click", runEvaluation);
$("#drop-zone").addEventListener("click", () => $("#file-input").click());
$("#change-upload").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (event) => { chooseFile(event.target.files[0]); event.target.value = ""; });
for (const name of ["dragenter", "dragover"]) $("#drop-zone").addEventListener(name, (event) => { event.preventDefault(); $("#drop-zone").classList.add("dragging"); });
for (const name of ["dragleave", "drop"]) $("#drop-zone").addEventListener(name, (event) => { event.preventDefault(); $("#drop-zone").classList.remove("dragging"); });
$("#drop-zone").addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));
$("#collection-split").addEventListener("change", () => { state.collectionPage = 1; loadCollection(); });
$("#previous-page").addEventListener("click", () => { if (state.collectionPage > 1) { state.collectionPage--; loadCollection(); } });
$("#next-page").addEventListener("click", () => { if (state.collectionPage < state.collectionPages) { state.collectionPage++; loadCollection(); } });
$$("[data-build]").forEach((button) => button.addEventListener("click", () => rebuild(button.dataset.build, button)));
$("#close-dialog").addEventListener("click", () => $("#image-dialog").close());
$("#image-dialog").addEventListener("click", (event) => { if (event.target === $("#image-dialog")) $("#image-dialog").close(); });
document.addEventListener("click", (event) => {
  const preview = event.target.closest("[data-preview]");
  if (preview) {
    $("#dialog-image").src = preview.dataset.preview;
    $("#dialog-image").alt = preview.dataset.label;
    $("#dialog-title").textContent = preview.dataset.label;
    $("#dialog-score").textContent = preview.dataset.score ? `${fixed(Number(preview.dataset.score) * 100, 1)}% similarity · cosine ${Number(preview.dataset.score).toFixed(3)}` : "Caltech-101 collection";
    $("#image-dialog").showModal();
  }
  const select = event.target.closest("[data-select]");
  if (select && !state.busy) {
    const id = Number(select.dataset.select);
    state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
    const active = state.selected.has(id);
    select.setAttribute("aria-pressed", String(active));
    select.textContent = active ? "✓" : "+";
    select.closest(".image-card").classList.toggle("selected", active);
    updateFeedback();
  }
  const category = event.target.closest("[data-category]");
  if (category) {
    state.collectionCategory = category.dataset.category;
    state.collectionPage = 1;
    $$("[data-category]").forEach((button) => { const active = button === category; button.classList.toggle("selected", active); button.setAttribute("aria-pressed", String(active)); });
    loadCollection();
  }
});

async function initialize() {
  try {
    state.overview = await api("/api/overview");
    $("#gallery-count").textContent = state.overview.gallery_count;
    $("#category-count").textContent = state.overview.categories.length;
    $("#example-category").innerHTML = state.overview.categories.map((item) => `<option value="${escapeHTML(item.id)}">${escapeHTML(item.label)}</option>`).join("");
    $("#example-category").value = state.overview.categories.some((item) => item.id === "cup") ? "cup" : state.overview.categories[0].id;
    $("#example-category").disabled = false;
    $("#hero-images").innerHTML = state.overview.showcase.map((item, i) => `<div class="ribbon-card"><div class="ribbon-image"><img src="${escapeHTML(item.thumbnail_url)}" alt="${escapeHTML(item.label)} from the collection"></div><span>${escapeHTML(item.label)} <small>0${i + 1}</small></span></div>`).join("");
    $("#category-filters").innerHTML = [{ id: "all", label: "All images" }, ...state.overview.categories].map((item) => `<button class="category-chip${item.id === "all" ? " selected" : ""}" data-category="${escapeHTML(item.id)}" aria-pressed="${item.id === "all"}">${escapeHTML(item.label)}</button>`).join("");
    chooseExample();
  } catch (error) {
    showError(error);
    $("#query-preview").innerHTML = '<span class="loading-label">Start the local Python server to connect.</span>';
    $("#query-name").textContent = "Collection unavailable";
  }
}
initialize();
