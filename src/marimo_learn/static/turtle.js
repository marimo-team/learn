// src/turtle.js
var NS = "http://www.w3.org/2000/svg";
var BG = "#1a1a2e";
function makeLine(x1, y1, x2, y2, color) {
  const el = document.createElementNS(NS, "line");
  el.setAttribute("x1", x1.toFixed(1));
  el.setAttribute("y1", y1.toFixed(1));
  el.setAttribute("x2", x2.toFixed(1));
  el.setAttribute("y2", y2.toFixed(1));
  el.setAttribute("stroke", color);
  el.setAttribute("stroke-width", "1.8");
  el.setAttribute("stroke-linecap", "round");
  return el;
}
function makeTurtle(x, y, angleDeg) {
  const r = Math.PI * angleDeg / 180;
  const R = 9;
  const pts = [0, 2 * Math.PI / 3, -(2 * Math.PI) / 3].map(
    (a) => `${(x + R * Math.cos(r + a)).toFixed(1)},${(y + R * Math.sin(r + a)).toFixed(1)}`
  ).join(" ");
  const poly = document.createElementNS(NS, "polygon");
  poly.setAttribute("points", pts);
  poly.setAttribute("fill", "#00ff88");
  poly.setAttribute("opacity", "0.9");
  return poly;
}
function render({ model, el }) {
  const W = model.get("width");
  const H = model.get("height");
  el.style.display = "inline-block";
  const controls = document.createElement("div");
  controls.style.cssText = "display:flex;gap:8px;align-items:center;margin-bottom:6px;font-family:sans-serif;font-size:14px";
  const btn = (label) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = "padding:4px 12px;cursor:pointer;border-radius:4px";
    return b;
  };
  const startBtn = btn("\u25B6 Start");
  const stopBtn = btn("\u25A0 Stop");
  stopBtn.disabled = true;
  const speedSlider = document.createElement("input");
  speedSlider.type = "range";
  speedSlider.min = 1;
  speedSlider.max = 60;
  speedSlider.style.width = "100px";
  const speedLabel = document.createElement("span");
  const syncSpeedLabel = () => {
    speedLabel.textContent = speedSlider.value + " fps";
  };
  speedSlider.value = Math.round(1 / model.get("delay"));
  syncSpeedLabel();
  controls.append(startBtn, stopBtn, speedSlider, speedLabel);
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.style.cssText = `background:${BG};border-radius:8px;display:block`;
  const segGroup = document.createElementNS(NS, "g");
  const turtleGroup = document.createElementNS(NS, "g");
  svg.append(segGroup, turtleGroup);
  el.append(controls, svg);
  model.on("change:_render", () => {
    const data = model.get("_render");
    if (!data || data.segments === void 0) return;
    segGroup.innerHTML = "";
    for (const [x1, y1, x2, y2, color] of data.segments) {
      segGroup.appendChild(makeLine(x1, y1, x2, y2, color));
    }
    turtleGroup.innerHTML = "";
    for (const [x, y, angle] of data.turtles) {
      turtleGroup.appendChild(makeTurtle(x, y, angle));
    }
    if (data.done) {
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  });
  startBtn.onclick = () => {
    segGroup.innerHTML = "";
    turtleGroup.innerHTML = "";
    model.set("_start_counter", model.get("_start_counter") + 1);
    model.save_changes();
    startBtn.disabled = true;
    stopBtn.disabled = false;
  };
  stopBtn.onclick = () => {
    model.set("_stop_requested", true);
    model.save_changes();
  };
  speedSlider.oninput = () => {
    syncSpeedLabel();
    model.set("delay", 1 / parseInt(speedSlider.value));
    model.save_changes();
  };
}
var turtle_default = { render };
export {
  turtle_default as default
};
//# sourceMappingURL=turtle.js.map
