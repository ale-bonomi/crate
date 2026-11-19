const app = document.querySelector(".app");
const details = document.getElementById("details");
let state = null;
let screen = "organize";
let toolsOpen = false;
let pickedFile = "";
let poll = 0;

const copy = {
  organize: "Drop a pack in the zone. Callima files it into the destination using these tags and the structure you pick.",
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
  dest.textContent = next.destination.path;
  dest.title = next.destination.path;
  document.getElementById("structure-jump").textContent = labelFor(next.structure.mode);
  const pack = next.pack;
  document.getElementById("crate-title").textContent = pack ? pack.name : "The crate";
  document.getElementById("crate-copy").textContent = pack
    ? pack.path
    : next.notice || "Sleeves, packs, and folders land here.";
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

function paintLibrary(next) {
  const box = document.getElementById("library");
  box.replaceChildren();
  for (const folder of next.library) {
    const button = el("button", "folder", folder.name);
    button.type = "button";
    button.setAttribute("aria-pressed", folder.selected ? "true" : "false");
    button.addEventListener("click", () => act("select-folder", { id: folder.id }));
    box.append(button);
  }
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
  tools.append(actionButton("Analyze the pack", () => act("analyze", {})));
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
  go.textContent = next.phase === "running" ? "…" : ready ? "Apply" : "Go";
  go.classList.toggle("apply", ready);
  go.disabled = next.phase === "running";
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
    if (picked.path) await act("add-folder", { path: picked.path });
  } catch (error) {
    showError(error);
  }
});

document.getElementById("choose-pack").addEventListener("click", async () => {
  try {
    const picked = await api("/api/pick", { prompt: "Pack to file" });
    if (picked.path) await act("set-pack", { path: picked.path });
  } catch (error) {
    showError(error);
  }
});

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

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state && state.phase === "ready") {
    act("cancel", {}).catch(showError);
  }
});

refresh().catch(showError);
