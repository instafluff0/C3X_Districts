'use strict';
// Reuse the existing editor parser; preserve every source tile and river bit.
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '../../..');
const {loadMapImport} = require(path.join(root, '../C3X_Editor/src/configCore'));
const {tileValues} = require(path.join(root, 'Renderer/tools/export_biq_terrain_scene'));
const source = process.argv[2];
const output = process.argv[3];
if (!source || !output) throw new Error('usage: node export_world.js <source.biq> <world.csv>');
const map = loadMapImport({civ3Path: path.dirname(root), scenarioPath: path.resolve(source), textEncoding: 'windows-1252'});
const section = map.importedSections.find(s => s.code === 'TILE');
if (!section || section.records.length !== map.tileCount || map.tileCount !== map.width * map.height / 2)
  throw new Error('Incomplete authoritative TILE section');
const lines = [`C3X_BIQ_TERRAIN_V3,${map.width},${map.height},${map.tileCount}`];
for (const record of section.records) {
  const t = tileValues(record);
  lines.push([t.sourceX,t.sourceY,t.base,t.real,t.bonus,t.overlays,t.riverMask].join(','));
}
fs.mkdirSync(path.dirname(path.resolve(output)), {recursive: true});
fs.writeFileSync(output, lines.join('\n') + '\n');
console.log(`Exported ${map.tileCount} authoritative tiles with river topology.`);
