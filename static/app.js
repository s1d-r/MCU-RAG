// ---------- helpers ----------
const $ = (id) => document.getElementById(id);

async function* sseLines(resp) {
  // Yields parsed JSON objects from a text/event-stream fetch response.
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (line.startsWith("data:")) {
          const payload = line.slice(5).trim();
          if (payload) {
            try { yield JSON.parse(payload); } catch (_) {}
          }
        }
      }
    }
  }
}

function fmt(text) {
  // Minimal, safe markdown: escape, then bold / inline-code / line breaks.
  const esc = text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

// ---------- view switching ----------
function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $(name).classList.add("active");
}

// ---------- ingest ----------
const dropzone = $("dropzone");
const fileInput = $("fileInput");

$("browseBtn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
dropzone.addEventListener("click", (e) => {
  if (!$("progressWrap").hidden) return;            // don't reopen picker mid-ingest
  if (e.target.closest(".drop-opts")) return;       // let option controls work
  fileInput.click();
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) startIngest(fileInput.files[0]); });

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f && f.type === "application/pdf") startIngest(f);
});

async function startIngest(file) {
  $("dropInner").style.display = "none";
  $("progressWrap").hidden = false;
  $("continueRow").hidden = true;
  $("progressLabel").textContent = "Uploading…";
  $("progressBar").style.width = "3%";
  $("progressSub").textContent = file.name;

  const form = new FormData();
  form.append("file", file);
  form.append("mode", $("modeSel").value);
  form.append("max_pages", $("maxPages").value || "");

  try {
    const resp = await fetch("/api/ingest", { method: "POST", body: form });
    for await (const ev of sseLines(resp)) {
      if (ev.type === "start") {
        $("progressLabel").textContent = `Rendering & indexing (${ev.mode} mode)…`;
        $("progressSub").textContent = `0 / ${ev.total} pages`;
      } else if (ev.type === "page") {
        const pct = Math.round((ev.done / ev.total) * 100);
        $("progressBar").style.width = pct + "%";
        $("progressPct").textContent = pct + "%";
        $("progressSub").textContent = `page ${ev.done} / ${ev.total}` + (ev.source === "vlm-fallback" ? "  (image-only → VLM)" : "");
      } else if (ev.type === "done") {
        $("progressBar").style.width = "100%";
        $("progressPct").textContent = "100%";
        $("progressLabel").textContent = "Done";
        messages.innerHTML = "";   // new document -> fresh chat (old history was about another doc)
        enterChat(ev.name, `${ev.pages} pages indexed`);
      } else if (ev.type === "error") {
        $("progressLabel").textContent = "Error";
        $("progressSub").textContent = ev.message;
      }
    }
  } catch (err) {
    $("progressLabel").textContent = "Error";
    $("progressSub").textContent = String(err);
  } finally {
    // reset the dropzone for a possible next upload
    setTimeout(() => {
      $("dropInner").style.display = "";
      $("progressWrap").hidden = true;
      $("progressBar").style.width = "0%";
      $("progressPct").textContent = "0%";
      fileInput.value = "";
    }, 800);
  }
}

// ---------- chat ----------
const messages = $("messages");
const qInput = $("qInput");
let activeController = null;

function enterChat(name, meta) {
  $("docName").textContent = name;
  $("docMeta").textContent = meta || "";
  showView("chat");
  setTimeout(() => qInput.focus(), 300);
}

$("newPdfBtn").addEventListener("click", () => showView("landing"));
$("continueBtn").addEventListener("click", () => showView("chat"));

qInput.addEventListener("input", () => {
  qInput.style.height = "auto";
  qInput.style.height = Math.min(qInput.scrollHeight, 180) + "px";
});
qInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
$("sendBtn").addEventListener("click", send);
$("stopBtn").addEventListener("click", stop);

function addMsg(role) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  wrap.appendChild(bubble);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return { wrap, bubble };
}

function renderCites(wrap, hits) {
  const row = document.createElement("div");
  row.className = "cites";
  hits.forEach((h) => {
    const chip = document.createElement("span");
    chip.className = "cite";
    chip.innerHTML = `<span class="dot ${h.source.includes("vlm") ? "vlm" : ""}"></span>page ${h.page} · ${h.score}`;
    chip.addEventListener("click", () => openLightbox(h.page));
    row.appendChild(chip);
  });
  wrap.insertBefore(row, wrap.firstChild);
}

function setBusy(busy) {
  $("sendBtn").hidden = busy;
  $("stopBtn").hidden = !busy;
  $("sendBtn").disabled = busy;
}

async function send() {
  const q = qInput.value.trim();
  if (!q || activeController) return;
  addMsg("user").bubble.textContent = q;
  qInput.value = "";
  qInput.style.height = "auto";

  const { wrap, bubble } = addMsg("bot");
  bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  setBusy(true);

  activeController = new AbortController();
  let answer = "";
  let started = false;
  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
      signal: activeController.signal,
    });
    for await (const ev of sseLines(resp)) {
      if (ev.type === "pages") {
        if (ev.hits && ev.hits.length) renderCites(wrap, ev.hits);
      } else if (ev.type === "token") {
        if (!started) { bubble.innerHTML = ""; started = true; }
        answer += ev.text;
        bubble.innerHTML = fmt(answer);
        messages.scrollTop = messages.scrollHeight;
      } else if (ev.type === "cancelled") {
        bubble.innerHTML = fmt(answer) + ' <span style="color:#ff6b6b">⏹ stopped</span>';
      } else if (ev.type === "done") {
        bubble.innerHTML = fmt(answer || "_(no answer)_");
      } else if (ev.type === "error") {
        bubble.innerHTML = `<span style="color:#ff6b6b">${fmt(ev.message)}</span>`;
      }
    }
  } catch (err) {
    if (!started) bubble.innerHTML = `<span style="color:#ff6b6b">${started ? "" : "stopped"}</span>`;
    else bubble.innerHTML = fmt(answer) + ' <span style="color:#ff6b6b">⏹ stopped</span>';
  } finally {
    setBusy(false);
    activeController = null;
    messages.scrollTop = messages.scrollHeight;
  }
}

async function stop() {
  try { await fetch("/api/stop", { method: "POST" }); } catch (_) {}
  if (activeController) activeController.abort();
}

// ---------- lightbox ----------
function openLightbox(page) {
  $("lightboxImg").src = `/api/page/${page}`;
  $("lightbox").hidden = false;
}
$("lightboxClose").addEventListener("click", () => ($("lightbox").hidden = true));
$("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") $("lightbox").hidden = true; });

// ---------- floating bits animation ----------
(function spawnBits() {
  const host = $("bits");
  if (!host) return;
  const n = 26;
  for (let i = 0; i < n; i++) {
    const s = document.createElement("span");
    s.textContent = Math.random() > 0.5 ? "1" : "0";
    s.style.left = Math.random() * 100 + "vw";
    s.style.fontSize = 11 + Math.random() * 12 + "px";
    const dur = 9 + Math.random() * 14;
    s.style.animationDuration = dur + "s";
    s.style.animationDelay = -Math.random() * dur + "s";
    host.appendChild(s);
  }
})();

// ---------- boot ----------
(async function boot() {
  try {
    const r = await fetch("/api/status");
    const { active } = await r.json();
    if (active) {
      $("loadedName").textContent = active.name;
      $("continueRow").hidden = false;
      $("docName").textContent = active.name;
      $("docMeta").textContent = `${active.pages} pages indexed`;
    }
  } catch (_) {}
})();
