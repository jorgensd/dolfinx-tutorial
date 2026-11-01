// Copyright (C) 2026 Jørgen S. Dokken
//
// SPDX-License-Identifier:    MIT
//
// Version switcher of the tutorial. .github/scripts/docs_site.py adds it to every page of every
// version when the web page is put together, so old builds get it without being rebuilt. It reads
// the versions from versions.json at the root of the web page, which is one folder up from here.
(() => {
  if (window.self !== window.top) return;
  const root = new URL("..", document.currentScript.src);
  if (!location.pathname.startsWith(root.pathname)) return;
  const [current, ...rest] = location.pathname.slice(root.pathname.length).split("/");
  // The page within its version, "" for the front page
  const page = rest.join("/");

  // The newest release is published as stable, and its own folder only has redirects
  const folder = (v) => v.folder ?? v.version;
  const home = (version) => new URL(version + "/", root).href;
  const samePage = (version) => new URL(version + "/" + page, root).href;
  const exists = (url) =>
    fetch(url, { method: "HEAD" }).then(
      (r) => r.ok,
      () => false
    );

  // The same page may not exist in the other version: then go to its front page
  async function go(version) {
    if (page !== "" && !(await exists(samePage(version)))) {
      location.href = home(version);
      return;
    }
    location.href = samePage(version) + location.search + location.hash;
  }

  const STYLE = `
    .tvs { position: relative; margin: 0.5rem 0; font-size: 0.9rem; }
    .tvs-float { position: fixed; right: 1rem; bottom: 1rem; z-index: 1000; margin: 0; }
    .tvs summary {
      display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
      padding: 0.35rem 0.6rem; cursor: pointer; list-style: none;
      border: 1px solid var(--pst-color-border, #ccc); border-radius: 0.25rem;
      background: var(--pst-color-surface, #f5f5f5); color: var(--pst-color-text-base, #222);
    }
    .tvs summary::-webkit-details-marker { display: none; }
    .tvs summary::after { content: "\\25BE"; }
    .tvs[open] summary::after { content: "\\25B4"; }
    .tvs summary:focus-visible { outline: 2px solid var(--pst-color-primary, #0a7d91); }
    .tvs ul {
      position: absolute; left: 0; right: 0; z-index: 1000; margin: 0.25rem 0 0; padding: 0.25rem 0;
      list-style: none; border: 1px solid var(--pst-color-border, #ccc); border-radius: 0.25rem;
      background: var(--pst-color-surface, #f5f5f5);
      box-shadow: 0 0.25rem 0.75rem var(--pst-color-shadow, rgba(0, 0, 0, 0.15));
    }
    .tvs-float ul { top: auto; bottom: 100%; margin: 0 0 0.25rem; min-width: 12rem; }
    .tvs li a {
      display: block; padding: 0.3rem 0.6rem; text-decoration: none;
      color: var(--pst-color-text-base, #222);
    }
    .tvs li a:hover, .tvs li a:focus-visible {
      background: var(--pst-color-primary-bg, #e0f2f5); color: var(--pst-color-primary, #0a7d91);
    }
    .tvs li a[aria-current] { font-weight: 600; color: var(--pst-color-primary, #0a7d91); }
    .tvs-banner {
      display: flex; align-items: flex-start; gap: 1rem; margin: 0 0 1rem; padding: 0.6rem 1rem;
      border-left: 4px solid var(--pst-color-warning, #f0b37e);
      background: var(--pst-color-surface, #f5f5f5); color: var(--pst-color-text-base, #222);
    }
    .tvs-banner p { margin: 0; flex: 1; }
    .tvs-banner button {
      border: none; background: none; cursor: pointer; font-size: 1.2rem; line-height: 1;
      color: var(--pst-color-text-muted, #666);
    }
  `;

  function dropdown(versions, entry, same) {
    const details = document.createElement("details");
    details.className = "tvs";
    const summary = document.createElement("summary");
    summary.setAttribute("aria-label", "Choose a version of the tutorial");
    summary.textContent = entry.name;
    const list = document.createElement("ul");
    for (const v of versions) {
      const a = document.createElement("a");
      a.textContent = v.name;
      // Points at the same page, so middle-click and copying the link work
      a.href = samePage(folder(v));
      if (same(v)) a.setAttribute("aria-current", "page");
      a.addEventListener("click", (e) => {
        if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
        e.preventDefault();
        if (same(v)) details.open = false;
        else go(folder(v));
      });
      const li = document.createElement("li");
      li.append(a);
      list.append(li);
    }
    details.append(summary, list);
    document.addEventListener("click", (e) => {
      if (!details.contains(e.target)) details.open = false;
    });
    details.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && details.open) {
        details.open = false;
        summary.focus();
      }
    });

    const brand = document.querySelector(".bd-sidebar-primary .navbar-brand");
    const item = brand?.closest(".sidebar-primary-item");
    if (item) {
      const wrapper = document.createElement("div");
      wrapper.className = "sidebar-primary-item";
      wrapper.append(details);
      item.after(wrapper);
    } else {
      details.classList.add("tvs-float");
      document.body.append(details);
    }
  }

  function banner(entry, stable) {
    const key = "tvs-dismissed-" + current;
    try {
      if (sessionStorage.getItem(key)) return;
    } catch {}
    const text =
      entry.version === "main"
        ? "You are reading the development version of the tutorial, built against the nightly " +
          "version of DOLFINx."
        : `You are reading the tutorial for ${entry.name}, an older release.`;
    const div = document.createElement("div");
    div.className = "tvs-banner";
    div.setAttribute("role", "note");
    const p = document.createElement("p");
    const a = document.createElement("a");
    a.href = samePage(stable.version);
    a.textContent = `the latest release (${stable.name.replace(/^stable \((.*)\)$/, "$1")})`;
    a.addEventListener("click", (e) => {
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      e.preventDefault();
      go(stable.version);
    });
    p.append(text + " See ", a, ".");
    const close = document.createElement("button");
    close.type = "button";
    close.setAttribute("aria-label", "Dismiss");
    close.textContent = "×";
    close.addEventListener("click", () => {
      div.remove();
      try {
        sessionStorage.setItem(key, "1");
      } catch {}
    });
    div.append(p, close);
    const main = document.querySelector(".bd-main");
    if (main) main.prepend(div);
    else document.body.prepend(div);
  }

  fetch(new URL("versions.json", root))
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((versions) => {
      const entry = versions.find((v) => folder(v) === current);
      if (!entry) return;
      // stable and the release it is a copy of are the same version
      const release = (v) => v.release ?? v.version;
      const same = (v) => release(v) === release(entry);
      const style = document.createElement("style");
      style.textContent = STYLE;
      document.head.append(style);
      dropdown(versions, entry, same);
      const stable = versions.find((v) => v.version === "stable");
      if (stable && !same(stable)) banner(entry, stable);
    })
    .catch(() => {});
})();
