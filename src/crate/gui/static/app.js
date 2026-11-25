const app = document.querySelector(".app");
const details = document.getElementById("details");
let state = null;
let screen = "organize";
let toolsOpen = false;
let pickedFile = "";
let pickedId = "";
let draggingId = "";
let revealFrom = false;
let poll = 0;

const copy = {
  organize: "Drop a library folder onto From or To, or select one and press Enter. Enter fills an empty To first, then From. Choose still opens a folder.",
  structure: "Type files one-shots and loops by sound. Genre, Flat, and Original keep a pack the way it arrived. Custom is one you build.",
  cleanup: "Analysis files are Ableton .asd sidecars. Duplicates are the same audio saved twice. Least used stays visible until you decide.",
};

const modes = [
  ["type", "Type"],
  ["genre", "Genre"],
  ["flat", "Flat"],
  ["original", "Original"],
  ["random", "Random"],
  ["custom", "Custom"],
];

async function api(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Request failed.");
  return body;
}

async function act(action, payload) {
  const next = await api("/api/action", { action, ...payload });
  render(next);
  watchJob(next);
}

async function refresh() {
  const next = await fetch("/api/state").then((response) => response.json());
  render(next);
  return next;
}

function watchJob(next) {
  clearInterval(poll);
  if (next.phase === "running") {
    poll = setInterval(async () => {
      const latest = await refresh();
      if (latest.phase !== "running") clearInterval(poll);
    }, 700);
  }
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function render(next) {
  state = next;
  document.getElementById("status").textContent = next.status || "";
  const dest = document.getElementById("dest-path");
  dest.textContent = next.ends.detail;
  dest.title = next.ends.to ? next.ends.to.path : "";
  document.querySelector(".destination").textContent = next.ends.title;
  document.getElementById("structure-jump").textContent = labelFor(next.structure.mode);
  paintEnds(next);
  paintLibrary(next);
  paintSaved(next);
  paintFiles(next);
  paintModes(next);
  paintToggles(next);
  paintPlan(next);
  paintTags(next);
  paintSettings(next);
  paintDetails(next);
  paintGo(next);
  showScreen(screen);
}

function labelFor(mode) {
  return (modes.find((item) => item[0] === mode) || ["", "Select"])[1];
}

function paintEnds(next) {
  const from = next.ends.from;
  const to = next.ends.to;
  document.getElementById("from-name").textContent = from ? from.name : "Choose a folder";
  const fromPath = document.getElementById("from-path");
  fromPath.textContent = from ? from.path : "Drop a folder, or choose one.";
  fromPath.title = from ? from.path : "";
  document.getElementById("to-name").textContent = to ? to.name : "Choose a folder";
  const toPath = document.getElementById("to-path");
  toPath.textContent = to ? to.path : "Drop a folder, or choose one.";
  toPath.title = to ? to.path : "";
  const line = document.getElementById("flow-line");
  line.textContent = next.notice ? `${next.ends.detail} ${next.notice}` : next.ends.detail;
  line.classList.toggle("blocked", Boolean(next.ends.blocked));
  const list = document.getElementById("flows");
  list.replaceChildren();
  for (const flow of next.flows) {
    const item = el("li");
    item.className = flow.active ? (flow.blocked ? "active blocked" : "active") : "";
    item.append(el("b", "", flow.title), document.createTextNode(` ${flow.detail}`));
    list.append(item);
  }
}

function paintLibrary(next) {
  const box = document.getElementById("library");
  const focused = document.activeElement && document.activeElement.dataset.id;
  if (pickedId && !next.library.some((folder) => folder.id === pickedId)) pickedId = "";
  box.replaceChildren();
  for (const folder of next.library) {
    const button = el("button", "folder");
    button.type = "button";
    button.dataset.id = folder.id;
    button.draggable = true;
    const marks = [folder.from ? "From" : "", folder.to ? "To" : ""].filter(Boolean).join(" · ");
    button.setAttribute("aria-pressed", folder.id === pickedId ? "true" : "false");
    button.setAttribute("aria-label", marks ? `${folder.name}, ${marks}` : folder.name);
    button.append(el("span", "", folder.name), el("span", "role", marks));
    button.addEventListener("click", () => {
      pickedId = folder.id;
      paintLibrary(state);
    });
    button.addEventListener("dragstart", (event) => {
      draggingId = folder.id;
      pickedId = folder.id;
      event.dataTransfer.setData("text/plain", folder.id);
      event.dataTransfer.effectAllowed = "copy";
    });
    button.addEventListener("dragend", () => {
      draggingId = "";
    });
    box.append(button);
    if (folder.id === focused) button.focus();
    if (revealFrom && folder.from) {
      button.scrollIntoView({ block: "nearest" });
      revealFrom = false;
    }
  }
}

function assignPicked() {
  if (!pickedId || !state) return;
  const action = state.ends.to ? "set-from" : "set-to";
  return act(action, { id: pickedId });
}

function paintSaved(next) {
  const box = document.getElementById("saved");
  box.replaceChildren();
  for (const item of next.structures) {
    const button = el("button", "saved-item", item.name);
    button.type = "button";
    button.setAttribute("aria-pressed", item.id === next.structure.id ? "true" : "false");
    button.addEventListener("click", () => act("set-structure", { id: item.id }));
    box.append(button);
  }
}

function paintFiles(next) {
  const box = document.getElementById("files");
  const search = document.getElementById("search");
  if (document.activeElement !== search) search.value = next.search || "";
  box.replaceChildren();
  if (!next.files.length) {
    box.append(el("p", "hint", next.cleanup.least && !next.search
      ? "Least used shows up after Ableton projects are indexed."
      : "No files yet."));
    return;
  }
  for (const file of next.files) {
    const button = el("button", "file");
    button.type = "button";
    button.append(el("span", "", file.name), el("small", "", file.note || ""));
    button.addEventListener("click", () => {
      pickedFile = file.path;
      details.textContent = file.path;
    });
    box.append(button);
  }
}

function paintModes(next) {
  const box = document.getElementById("modes");
  box.replaceChildren();
  for (const [mode, label] of modes) {
    const button = el("button", "mode");
    button.type = "button";
    button.setAttribute("aria-pressed", next.structure.mode === mode ? "true" : "false");
    const preview = el("div", "preview", mode === "custom" ? "+" : "");
    button.append(preview, el("div", "label", label));
    button.addEventListener("click", () => act("set-mode", { mode }));
    box.append(button);
  }
}

function paintToggles(next) {
  for (const button of document.querySelectorAll(".toggle")) {
    const on = Boolean(next.cleanup[button.dataset.flag]);
    button.setAttribute("aria-pressed", on ? "true" : "false");
    button.querySelector("em").textContent = on ? "On" : "Off";
  }
}

function paintPlan(next) {
  const list = document.getElementById("plan");
  const ops = (next.summary && next.summary.ops) || [];
  list.replaceChildren();
  if (next.phase !== "ready" && next.phase !== "running") return;
  for (const op of ops.slice(0, 8)) {
    const item = el("li");
    item.append(el("b", "", op.mode), el("span", "", `${op.src} → ${op.dst}`));
    list.append(item);
  }
}

function paintTags(next) {
  for (const row of document.querySelectorAll(".tag-block .row")) {
    const input = row.querySelector("input");
    if (document.activeElement === input) continue;
    input.value = next[row.dataset.which][row.dataset.field] || "";
  }
}

function paintSettings(next) {
  const root = document.getElementById("settings-fields");
  if (root.contains(document.activeElement)) return;
  root.replaceChildren();
  root.append(group("Library", next.settings.library));
  root.append(group("Synths", next.settings.synths));
  root.append(group("DAWs", next.settings.daws));
  const files = el("div", "group");
  files.append(el("h2", "", "Files"));
  const row = el("div", "setting style-cycle");
  const key = el("div", "key", "File tag style");
  const value = el("button", "chip grow");
  value.type = "button";
  value.textContent = next.tagStyle;
  value.addEventListener("click", () => {
    const order = ["keep", "lowercase", "tagged"];
    const style = order[(order.indexOf(next.tagStyle) + 1) % order.length];
    act("set-style", { style });
  });
  row.append(key, value);
  files.append(row);
  root.append(files);
}

function group(title, rows) {
  const box = el("div", "group");
  box.append(el("h2", "", title));
  for (const row of rows) {
    const line = el("div", "setting");
    const input = el("input", "path-input");
    input.value = row.path || "";
    input.dataset.key = row.key;
    input.setAttribute("aria-label", row.label);
    const set = el("button", "set-path", "Set path");
    set.type = "button";
    set.addEventListener("click", async () => {
      const picked = await api("/api/pick", { prompt: `Folder for ${row.label}` });
      if (picked.path) input.value = picked.path;
    });
    line.append(el("div", "key", row.label), input, set);
    box.append(line);
  }
  return box;
}

function paintDetails(next) {
  const error = next.phase === "error";
  details.className = error ? "error" : "";
  if (next.message) details.textContent = next.message;
  else if (next.phase === "ready") {
    const summary = next.summary || {};
    const notes = (summary.notes || []).join(" ");
    details.textContent = `${summary.count} files, ${summary.sidecars} sidecars, ${summary.skipped} skipped. ${notes}`;
  } else details.textContent = copy[screen] || copy.organize;
  const tools = document.getElementById("tools");
  tools.hidden = !toolsOpen;
  if (!toolsOpen) return;
  tools.replaceChildren();
  const reviewHead = el("p", "hint", `${next.tools.used} used in Live sets, ${next.tools.unused} unused.`);
  tools.append(reviewHead);
  tools.append(actionButton("Undo last batch", () => act("undo", {})));
  tools.append(actionButton(next.notice ? "Stop watching Splice" : "Watch Splice", () => act("watch", {})));
  tools.append(actionButton("Analyze the From folder", () => act("analyze", {})));
  tools.append(actionButton("Re-file this folder", () => act("go", { screen: "organize" })));
  tools.append(actionButton("Index Ableton projects", indexProjects));
  tools.append(actionButton("Score a labels CSV", scoreCsv));
  tools.append(actionButton("Sounds like the selected file", () => {
    if (!pickedFile) throw new Error("Choose a file in Clean Up first.");
    return act("similar", { path: pickedFile });
  }));
  for (const row of next.review) {
    const line = el("div", "review");
    const input = el("input", "path-input");
    input.placeholder = `${row.guess} · label`;
    input.setAttribute("aria-label", `Label for ${row.name}`);
    const save = el("button", "text-button", row.name);
    save.type = "button";
    save.addEventListener("click", () => act("correct", { path: row.path, label: input.value }));
    line.append(input, save);
    tools.append(line);
  }
}

function actionButton(text, onClick) {
  const button = el("button", "text-button", text);
  button.type = "button";
  button.addEventListener("click", () => Promise.resolve(onClick()).catch(showError));
  return button;
}

function paintGo(next) {
  const go = document.getElementById("go");
  const ready = next.phase === "ready";
  const blocked = Boolean(next.ends.blocked) && !ready;
  go.textContent = next.phase === "running" ? "…" : ready ? "Apply" : "Go";
  go.classList.toggle("apply", ready);
  go.disabled = next.phase === "running" || blocked;
  go.title = blocked ? next.ends.detail : "";
}

function showScreen(name) {
  screen = name;
  app.dataset.screen = name;
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.setAttribute("aria-selected", tab.dataset.screen === name ? "true" : "false");
  }
  if (state) paintDetails(state);
}

function showError(error) {
  details.className = "error";
  details.textContent = error.message || String(error);
}

function settingsPaths() {
  const paths = {};
  for (const input of document.querySelectorAll(".path-input")) {
    if (input.dataset.key) paths[input.dataset.key] = input.value;
  }
  return paths;
}

async function indexProjects() {
  const picked = await api("/api/pick", { prompt: "Ableton project folder" });
  if (picked.path) await act("projects", { path: picked.path });
}

async function scoreCsv() {
  const picked = await api("/api/pick", { prompt: "Folder that contains the labels CSV" });
  if (!picked.path) return;
  const name = window.prompt("CSV file name inside that folder", "labels.csv");
  if (!name) return;
  await act("eval", { path: `${picked.path.replace(/\/$/, "")}/${name}` });
}

document.querySelectorAll(".tabs button").forEach((tab) => {
  tab.addEventListener("click", () => showScreen(tab.dataset.screen));
});

document.getElementById("settings-toggle").addEventListener("click", () => {
  showScreen(screen === "settings" ? "organize" : "settings");
});

document.getElementById("structure-jump").addEventListener("click", () => showScreen("structure"));

document.getElementById("info").addEventListener("click", () => {
  toolsOpen = !toolsOpen;
  document.getElementById("info").setAttribute("aria-expanded", toolsOpen ? "true" : "false");
  if (state) paintDetails(state);
});

document.getElementById("add-folder").addEventListener("click", async () => {
  try {
    const picked = await api("/api/pick", { prompt: "Add a library folder" });
    if (picked.path) {
      revealFrom = true;
      await act("add-folder", { path: picked.path });
    }
  } catch (error) {
    showError(error);
  }
});

document.getElementById("choose-from").addEventListener("click", () => pickEnd("From folder", "set-from"));
document.getElementById("choose-to").addEventListener("click", () => pickEnd("To folder", "set-to"));
document.getElementById("clear-from").addEventListener("click", () => act("clear-from", {}).catch(showError));
document.getElementById("clear-to").addEventListener("click", () => act("clear-to", {}).catch(showError));
document.getElementById("swap-ends").addEventListener("click", () => act("swap-ends", {}).catch(showError));

async function pickEnd(prompt, action) {
  try {
    const picked = await api("/api/pick", { prompt });
    if (picked.path) await act(action, { path: picked.path });
  } catch (error) {
    showError(error);
  }
}

document.getElementById("add-file").addEventListener("click", async () => {
  try {
    const picked = await api("/api/pick", { prompt: "Folder to search" });
    if (picked.path) await act("pin", { path: picked.path });
  } catch (error) {
    showError(error);
  }
});

document.getElementById("new-structure").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = document.getElementById("structure-name");
  act("new-structure", { name: input.value || "Custom" }).then(() => {
    input.value = "";
  }).catch(showError);
});

document.getElementById("search").addEventListener("input", (event) => {
  act("search", { q: event.target.value }).catch(showError);
});

document.querySelectorAll(".toggle").forEach((button) => {
  button.addEventListener("click", () => {
    const flag = button.dataset.flag;
    act("set-cleanup", { [flag]: button.getAttribute("aria-pressed") !== "true" }).catch(showError);
  });
});

document.querySelectorAll(".tag-block .row").forEach((row) => {
  const input = row.querySelector("input");
  row.querySelector(".plus").addEventListener("click", () => input.focus());
  input.addEventListener("change", () => {
    const which = row.dataset.which;
    const payload = { which, genre: state[which].genre, mood: state[which].mood };
    payload[row.dataset.field] = input.value;
    act("set-tags", payload).catch(showError);
  });
});

document.getElementById("go").addEventListener("click", () => {
  const action = state && state.phase === "ready" ? "apply" : "go";
  act(action, { screen: screen === "cleanup" ? "cleanup" : "organize" }).catch(showError);
});

document.getElementById("save").addEventListener("click", () => {
  act("save-settings", { paths: settingsPaths() }).catch(showError);
});

document.getElementById("reset").addEventListener("click", () => {
  act("reset-settings", {}).catch(showError);
});

function bindDrop(id, action) {
  const node = document.getElementById(id);
  node.addEventListener("dragover", (event) => {
    if (!draggingId) return;
    event.preventDefault();
    node.classList.add("drop");
  });
  node.addEventListener("dragleave", (event) => {
    if (event.relatedTarget && node.contains(event.relatedTarget)) return;
    node.classList.remove("drop");
  });
  node.addEventListener("drop", (event) => {
    event.preventDefault();
    node.classList.remove("drop");
    const folderId = draggingId || event.dataTransfer.getData("text/plain");
    draggingId = "";
    if (folderId) act(action, { id: folderId }).catch(showError);
  });
}

bindDrop("from-drop", "set-from");
bindDrop("to-drop", "set-to");

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state && state.phase === "ready") {
    act("cancel", {}).catch(showError);
    return;
  }
  if (event.key !== "Enter" || event.repeat || screen !== "organize") return;
  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return;
  const folder = target instanceof Element ? target.closest(".folder") : null;
  if (!folder && target instanceof Element && target.closest("button, a")) return;
  const id = folder ? folder.dataset.id : pickedId;
  if (!id) return;
  event.preventDefault();
  pickedId = id;
  assignPicked().catch(showError);
}, true);

refresh().catch(showError);
