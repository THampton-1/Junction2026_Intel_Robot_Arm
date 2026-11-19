// Builds the shared retro frame (top bar, logo, left nav, footer) around each inner page's content.
// To add or rename a page, edit NAV here and the matching links on index.html.
const NAV = [
  { file: "happened.html",   title: "Overview" },
  { file: "detection.html",  title: "Anomaly Scores" },
];

function buildFrame() {
  const here = location.pathname.split("/").pop();
  const content = document.getElementById("content");

  let links = `<a class="nav" href="index.html">Live Inspection</a>`;
  for (const page of NAV) {
    let cls = "nav";
    if (page.gap) cls += " gap";
    if (page.file === here) cls += " here";
    links += `<a class="${cls}" href="${page.file}">${page.title}</a>`;
  }

  const pageDiv = document.createElement("div");
  pageDiv.id = "page";
  pageDiv.innerHTML =
    `<div id="topbar"><span>Adapt.com</span><span>Contents</span><span>Search</span><span>Contact Us</span><span>Support</span></div>` +
    `<a id="logo" href="index.html">adapt<b>.</b></a>` +
    `<div id="cols"><nav id="leftcol" aria-label="Pages">${links}</nav></div>` +
    `<div id="footer">*Built at Junction 2026 on Intel&reg; Core&trade; Ultra &copy; 2026 Adapt</div>`;

  // Move the page's own content into the right-hand column.
  document.body.prepend(pageDiv);
  pageDiv.querySelector("#cols").appendChild(content);
}

buildFrame();
