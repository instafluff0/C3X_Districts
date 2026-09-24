#!/usr/bin/env node
'use strict';

// Keep the BIQ parser and terrain decoding owned by the production tool.
// The native preview needs the whole world, including the river edge byte.
const fs = require('fs');
const path = require('path');
const {tileValues} = require('../tools/export_biq_terrain_scene.js');

const biq = process.argv[2];
const output = process.argv[3];
if (!biq || !output) {
  console.error('usage: node export_biq.js <scenario.biq> <scene.csv>');
  process.exit(2);
}
const editorCore = path.resolve(__dirname, '..', '..', '..', 'C3X_Editor', 'src', 'configCore');
const {loadMapImport} = require(editorCore);
const result = loadMapImport({
  civ3Path: path.resolve(__dirname, '..', '..', '..'),
  scenarioPath: path.resolve(biq),
  textEncoding: 'windows-1252'
});
const section = result.importedSections.find(item => item.code === 'TILE');
if (!section || !Array.isArray(section.records) || section.records.length !== result.tileCount)
  throw new Error('BIQ TILE section is incomplete');
const lines = [`C3X_BIQ_TERRAIN_V3,${result.width},${result.height},${result.tileCount}`];
for (const record of section.records) {
  const tile = tileValues(record);
  lines.push(`${tile.sourceX},${tile.sourceY},${tile.base},${tile.real},${tile.bonus},${tile.overlays},${tile.riverMask}`);
}
fs.mkdirSync(path.dirname(path.resolve(output)), {recursive: true});
fs.writeFileSync(output, `${lines.join('\n')}\n`);
console.log(`Exported ${result.tileCount} BIQ tiles with river topology`);
