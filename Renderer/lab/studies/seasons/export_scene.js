'use strict';
// Lab-only extension of the existing CSV with authoritative map wrap flags.
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '../../../..');
const {tileValues} = require(path.join(root, 'Renderer/tools/export_biq_terrain_scene.js'));
const {loadMapImport} = require(path.join(root, '../C3X_Editor/src/configCore'));
const [source, output] = process.argv.slice(2);
if (!source || !output) throw new Error('export_scene.js <map.biq> <scene.csv>');
const result = loadMapImport({civ3Path: path.resolve(root, '..'),
  scenarioPath: path.resolve(source), textEncoding: 'windows-1252'});
const section = result.importedSections.find(s => s.code === 'TILE');
const world = result.importedSections.find(s => s.code === 'WMAP')?.records[0];
if (!world || section?.records.length !== result.tileCount)
  throw new Error('Complete TILE and WMAP records are required');
const flags = Number(world.flags);
if (!Number.isInteger(flags)) throw new Error('Invalid WMAP flags');
const lines = [`C3X_BIQ_TERRAIN_V3,${result.width},${result.height},${result.tileCount},${flags & 1 ? 1 : 0},${flags & 2 ? 1 : 0}`];
for (const record of section.records) {
  const t = tileValues(record);
  lines.push(`${t.sourceX},${t.sourceY},${t.base},${t.real},${t.bonus},${t.overlays},${t.riverMask}`);
}
fs.writeFileSync(output, lines.join('\n') + '\n');
console.log(`Exported ${result.tileCount} authoritative BIQ tiles, river topology and wrapping`);
