// Cut a 32x32 Lab scene from a Civ III save: terrain, rivers, route and
// improvement overlay bits, and the cities inside it. Parsing is delegated to
// the neighboring C3X Editor (savInspect.js builds a debug BIQ of the save).
// Usage (from the C3X checkout root):
//   node Renderer/lab/studies/roads/save_window.js <save.SAV> <out.csv> <out-cities.json> <centerX> <centerY>
// The raw tile (centerX, centerY) lands on the Lab tile (16, 16).
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const root = process.cwd();
const editor = path.resolve(root, '..', 'C3X_Editor', 'src');
const { loadMapImport } = require(path.join(editor, 'configCore'));
const { inspectSavFile } = require(path.join(editor, 'biq', 'savInspect.js'));

const [save, out, citiesOut, cxText, cyText] = process.argv.slice(2);
if (!save || !out || !citiesOut || cxText === undefined || cyText === undefined) {
  console.error('usage: node save_window.js <save.SAV> <out.csv> <out-cities.json> <centerX> <centerY>');
  process.exit(2);
}
const ox = Number(cxText) - 16, oy = Number(cyText) - 16;
const report = inspectSavFile(save, { debugBiqBuffer: true });
if (!report.ok) throw new Error(report.error);
const biq = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'c3x-save-window-')), 'save.biq');
fs.writeFileSync(biq, report.debugBiqBuffer);
const map = loadMapImport({ civ3Path: path.resolve(root, '..'), scenarioPath: biq, textEncoding: 'windows-1252' });
fs.rmSync(path.dirname(biq), { recursive: true, force: true });
const width = map.width;
const number = (record, key, fallback = 0) => {
  const value = Number.parseInt(String(record[key] == null ? '' : record[key]), 10);
  return Number.isFinite(value) ? value : fallback;
};
const tiles = new Map();
map.importedSections.find((section) => section.code === 'TILE').records.forEach((record, index) => {
  const half = width / 2, y = Math.floor(index / half), x = (index % half) * 2 + (y & 1);
  tiles.set(`${x},${y}`, record);
});
const rows = [];
for (let y = 0; y < 32; y += 1) {
  for (let x = y & 1; x < 32; x += 2) {
    const record = tiles.get(`${((x + ox) % width + width) % width},${y + oy}`);
    if (!record) continue;
    let packed = number(record, 'c3cbaserealterrain', -1);
    if (packed < 0) packed = number(record, 'baserealterrain', 2);
    packed &= 0xff;
    const base = packed <= 15 ? packed : packed & 15, real = packed <= 15 ? packed : (packed >>> 4) & 15;
    const overlays = number(record, 'c3coverlays', number(record, 'overlays')) >>> 0;
    const river = (number(record, 'riverconnectioninfo', number(record, 'river_connection_info')) >>> 0) & 0xaa;
    const bonus = number(record, 'c3cbonuses', number(record, 'bonuses')) >>> 0;
    rows.push(`${x},${y},${base},${real},${bonus},${overlays},${river}`);
  }
}
fs.writeFileSync(out, `C3X_BIQ_TERRAIN_V3,32,32,${rows.length}\n${rows.join('\n')}\n`);
const cities = [];
for (const city of report.cities.records) {
  const x = ((city.x - ox) % width + width) % width, y = city.y - oy;
  if (x >= 0 && x < 32 && y >= 0 && y < 32) cities.push({ name: city.name, x, y, pop: city.population });
}
fs.writeFileSync(citiesOut, JSON.stringify(cities));
console.log(`${rows.length} tiles; cities: ${cities.map((city) => `${city.name}@${city.x},${city.y}`).join(' ')}`);
