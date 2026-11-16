# HTML starter instructions

Follow the `marimo-studio` skill for notebook ownership, projection selection,
view lifecycle, and validation. This file covers the browser-native project
supplied by this starter.

## Project intent

Keep this section current with the user's audience, analytical goal, concrete
project details, aesthetic direction, interaction priorities, and approved
library or framework preferences. Preserve decisions that should guide later
agents.

- Audience: students working through the marimo learn probability course.
- Point of view: evidence shrinks the world. Seeing E throws away every
  outcome where E did not happen, and what is left becomes the new whole. The
  page shows that move instead of describing it.
- Layout: the notebook's Markdown in reading order. Each worked example gets a
  before/after pair of small squares. In the interactive section, `area_model`
  and the notebook's live `area_walkthrough` sit beside a sticky workbench:
  the three sliders, the area model with an "All outcomes / Only where E
  happened" switch, a legend numbered like the walkthrough's rectangles, and
  the prior, evidence, and posterior. Narrow screens stack in reading order.
- Interaction: the switch is view state kept in the URL fragment (`#given-e`).
  Collapsing fades the regions outside E and stretches the two E regions to
  fill the square, so the H boundary slides from the prior (dashed line) to
  the posterior. Reduced motion turns the transitions off.
- Data: every number comes from the notebook. The model reads `area_inputs`,
  a hidden cell that collects the slider values, because a projected
  `p_h.value` is not republished when a slider placed with `<marimo-output>`
  moves. `p_e` and `bayes_result` give the evidence and posterior. The
  examples read the top-level values of their cells (`p_disease`,
  `medical_posterior`, and so on). Keep analytical prose in notebook cells.
- Design: marimo's DESIGN.md tokens, PT Sans text and Lora headings, with the
  notebook's yellow hypothesis and teal evidence colors.

## Use the supplied Studio integration

`index.html` places the notebook's named cells and outputs by name. Keep,
reorder, group, or replace those hosts as the page design develops. Cell names
remain stable Studio targets for the notebook cells.

The starter defines `observeMarimoValue` inside the `index.html` module script.
Keep it inline or move it to a JavaScript file referenced directly from the
entry document. Use it when page JavaScript consumes a notebook value or eager
dataframe. Keep the corresponding `mo-value` host in authored HTML so Studio
can inspect and authorize its selector. Group hosts that only feed page code in
a hidden container. Lens selects the parent of a hidden host, so a host placed
directly in `#app-shell` makes the whole page one selection target.

```html
<div hidden>
  <span id="rows-data" hidden mo-value="rows"></span>
</div>
```

```js
const source = document.querySelector("#rows-data");
if (source) {
  const stop = observeMarimoValue(source, {
    onValue: (rows) => renderRows(rows),
    onError: (error) => renderError(error.message),
  });
  window.addEventListener("pagehide", stop, { once: true });
}
```

Eager dataframes arrive as a shared Flechette `Table`. Use
[https://github.com/uwdata/flechette](https://github.com/uwdata/flechette) as
the table API reference. Treat the table as immutable. Keep data columnar with
`getChild()`, `select()`, and `toColumns()`. Call `toArray()` when browser code
needs row objects.

## Add dependencies

`index.html` loads PT Sans and Lora from jsDelivr (Fontsource) and has no
script dependencies. The fonts need browser network access and a content
security policy with `style-src` and `font-src` access to jsDelivr; the
fallback fonts keep the page readable without them. Import any new browser
module from a versioned URL at the top of the module script:

```js
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
```

## Work within the HTML project

- Keep the document structure and projection hosts in `index.html`.
- Keep styles and browser behavior inline for a compact page, or reference CSS
  through `<link rel="stylesheet">` and JavaScript through `<script src>`
  directly from `index.html`. Studio copies these exact `.css`, `.js`, and
  `.mjs` source files into the browser artifact.
- Keep each direct source as a leaf file. Bundle or inline local CSS `url()` and
  `@import` dependencies and local JavaScript imports or re-exports. Explicit
  HTTPS, data, and fragment references remain available.
- Leave `<base href>` out of the entry document. HTTP and HTTPS dependency URLs
  need `//` and a host.
- Choose a provider that builds the JavaScript module graph for import maps,
  computed imports, and source-phase imports.
- Keep direct JavaScript within the pinned parser's accepted grammar. Inspection
  fails closed when it cannot establish the dependency boundary.
- End `break` and `continue` with `;` when another statement follows. The pinned
  grammar rejects a following regex statement when automatic semicolon
  insertion separates it from the restricted statement.
- The pinned grammar rejects import attributes on re-export statements. Import
  the remote module with its attributes, then use
  `export { value as default }` to preserve a default re-export. Enumerate named
  exports or choose a graph-building provider for a wildcard re-export.
- Keep projection hosts inside `#app-shell`.
- Inline project-owned images and fonts with the document. External HTTP URLs
  and `data:` URLs remain available.
- Use browser APIs for focused interaction. Choose the React or Svelte starter
  when the page needs component compilation, a local import graph, or separate
  browser assets.

Studio publishes the entry document and its declared local sources after
validating the HTML and projection hosts. Treat the built artifact as the
acceptance boundary for the page.

## Link custom results to notebook inputs

Keep projection hosts explicit in authored source. Custom regions need every
kernel input, a readable label, and a rendering-source reference such as
`{"path":"index.html"}`. Keep these attributes on authored elements outside
native output subtrees. Follow the installed Studio skill's
`references/projections.md` for the shared contract:

```python
import marimo_studio

print(marimo_studio.agent.skill().file("references/projections.md").read_text())
```

## Maintain project ignore rules

You own this view project's `.gitignore`. When adding libraries, extensions, or
build tools, ignore their generated files, caches, local configuration, and
secrets. Keep authored source, dependency manifests, and lockfiles tracked.
Studio keeps its own build state and lock files out of Git. Check
`git status --short --ignored` after running new tooling and update the view's
ignore rules before committing.
