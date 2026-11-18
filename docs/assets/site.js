// Shared header, navigation and footer for every page.
// Edit the PAGES list to add, rename or reorder pages: every page picks it up.
const PAGES = [
  { file: "index.html",       title: "Overview" },
  { file: "setup.html",       title: "Setup" },
  { file: "perception.html",  title: "Defect detection" },
  { file: "motion.html",      title: "Arm motion" },
  { file: "algorithms.html",  title: "Photo trigger & crop" },
  { file: "integration.html", title: "What happened" },
  { file: "reflection.html",  title: "On reflection" },
  { file: "evidence.html",    title: "Evidence" },
];

const REPO = "https://github.com/THampton-1/Junction2026_Intel_Robot_Arm";

function buildChrome() {
  // Work out which page we are on from the URL ("/" means the overview).
  let here = location.pathname.split("/").pop();
  if (here === "") here = "index.html";

  let links = "";
  for (const page of PAGES) {
    const cls = page.file === here ? ' class="here" aria-current="page"' : "";
    links += `<a href="${page.file}"${cls}>${page.title}</a>`;
  }

  const header = document.createElement("header");
  header.className = "top";
  header.innerHTML =
    `<div class="top-inner">` +
    `<a class="logo" href="index.html">adapt<b>.</b></a>` +
    `<span class="tagline">Junction 2026 · Intel Physical AI Challenge</span>` +
    `<nav class="nav" aria-label="Pages">${links}</nav>` +
    `</div>`;
  document.body.prepend(header);

  const footer = document.createElement("footer");
  footer.innerHTML =
    `<div class="inner">Built at Junction 2026 (14–15 Nov) by Christina Zhou, Lucas Kiewek, Siongan and Theo Hampton. ` +
    `Source and raw evidence: <a href="${REPO}">GitHub repository</a>.</div>`;
  document.body.append(footer);
}

// Turn any <a data-src="path/in/repo"> into a link to that file on GitHub.
function linkSources() {
  for (const a of document.querySelectorAll("a[data-src]")) {
    a.href = `${REPO}/blob/main/${a.dataset.src}`;
    if (!a.textContent.trim()) a.textContent = a.dataset.src;
  }
}

buildChrome();
linkSources();
