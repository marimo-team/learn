/**
 * Build script for marimo-learn.
 *
 * 1. Copies per-widget ESM files from @gvwilson/forma into
 *    src/marimo_learn/static/ for use by the Python anywidget package.
 * 2. Copies the standalone forma bundle into js/dist/ for the example pages.
 * 3. Builds turtle.js (marimo-learn-specific) with esbuild and places it
 *    alongside the other static files.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as esbuild from 'esbuild';

const dir      = path.dirname(fileURLToPath(import.meta.url));
const formaSrc = path.resolve(dir, 'node_modules/@gvwilson/forma/dist');
const widgetDst = path.resolve(dir, '../src/marimo_learn/static');
const bundleDst = path.resolve(dir, 'dist');

// --- 1. Copy forma per-widget files ---
fs.mkdirSync(widgetDst, { recursive: true });
for (const file of fs.readdirSync(path.join(formaSrc, 'widgets'))) {
  fs.copyFileSync(
    path.join(formaSrc, 'widgets', file),
    path.join(widgetDst, file),
  );
}
console.log('Copied forma widgets →', widgetDst);

// --- 2. Copy standalone forma bundle for example pages ---
fs.mkdirSync(bundleDst, { recursive: true });
fs.copyFileSync(path.join(formaSrc, 'forma.js'), path.join(bundleDst, 'forma.js'));
console.log('Copied forma bundle  →', path.join(bundleDst, 'forma.js'));

// --- 3. Build turtle.js with esbuild ---
await esbuild.build({
  entryPoints: ['src/turtle.js'],
  bundle: true,
  format: 'esm',
  outfile: path.join(widgetDst, 'turtle.js'),
  minify: false,
  sourcemap: true,
  logLevel: 'info',
});
console.log('Built turtle.js      →', path.join(widgetDst, 'turtle.js'));
