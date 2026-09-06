'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {
  riverTopologyMetrics,
  selectViewport,
  terrainCode,
  writeViewport
} = require('./export_biq_terrain_scene');

function makeMap(width, height, realAt, riverAt = () => 0) {
  const records = [];
  for (let y = 0; y < height; y += 1) {
    for (let column = 0; column < width / 2; column += 1) {
      const x = column * 2 + (y & 1);
      const real = realAt(column, y);
      records.push({
        xpos: String(x),
        ypos: String(y),
        c3cbaserealterrain: String((real << 4) | 2),
        c3cbonuses: '0',
        c3coverlays: '0',
        riverconnectioninfo: String(riverAt(column, y))
      });
    }
  }
  return records;
}

test('terrain preferences accept names and numeric codes', () => {
  assert.equal(terrainCode('marsh'), 9);
  assert.equal(terrainCode('10'), 10);
  assert.throws(() => terrainCode('bogus'), /Unknown terrain preference/);
});

test('dynamic selector returns exactly 96 source-backed tiles and favors marsh', () => {
  const records = makeMap(40, 12, (column, row) =>
    column >= 14 && column < 19 && row >= 3 && row < 8 ? 9 : 2
  );
  const selected = selectViewport(records, 40, 12, 12, 8, ['marsh']);
  assert.equal(selected.tiles.length, 96);
  assert.equal(selected.preferredCount, 25);
  assert.ok(selected.tiles.every((tile) =>
    tile.sourceX === ((selected.originColumn + tile.column) % 20) * 2 +
      ((selected.originRow + tile.row) & 1)
  ));
});

test('dynamic selector has a deterministic row/column tie break', () => {
  const records = makeMap(24, 8, () => 2);
  const selected = selectViewport(records, 24, 8, 12, 8, []);
  assert.equal(selected.originColumn, 0);
  assert.equal(selected.originRow, 0);
});

test('diamond selector follows both true Civ III edge-adjacency axes', () => {
  const records = makeMap(40, 20, (column, row) =>
    column >= 6 && column < 15 && row >= 4 && row < 17 ? 9 : 2
  );
  const selected = selectViewport(records, 40, 20, 12, 8, ['marsh'], 'diamond');
  assert.equal(selected.tiles.length, 96);
  assert.equal(selected.windowShape, 'diamond');
  assert.ok(selected.tiles.every((tile) => {
    const expectedX = (selected.originSourceX + tile.column + tile.row) % 40;
    const expectedY = selected.originSourceY + tile.column - tile.row;
    return tile.sourceX === expectedX && tile.sourceY === expectedY;
  }));
});

test('diamond selector honors an exact raw BIQ origin', () => {
  const records = makeMap(80, 80, () => 2);
  const selected = selectViewport(records, 80, 80, 12, 8, [], 'diamond', 53, 55);
  assert.equal(selected.originSourceX, 53);
  assert.equal(selected.originSourceY, 55);
  assert.equal(selected.tiles.length, 96);
  assert.equal(selected.tiles.at(-1).sourceX, 71);
  assert.equal(selected.tiles.at(-1).sourceY, 59);
});

test('dynamic selector rejects incomplete BIQ tile data', () => {
  const records = makeMap(24, 8, () => 2);
  records.pop();
  assert.throws(
    () => selectViewport(records, 24, 8, 12, 8, []),
    /no complete viewport/
  );
});

test('window export preserves an authoritative two-cell adjacency halo', () => {
  const records = makeMap(40, 30, (column, row) =>
    column >= 12 && row >= 8 ? 11 : 2
  );
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'c3x-biq-window-'));
  const outputPath = path.join(directory, 'window.csv');
  const selected = writeViewport(outputPath, {width: 40, height: 30}, {
    records,
    windowColumns: 12,
    windowRows: 8,
    preferredReal: [],
    windowShape: 'diamond',
    originX: 8,
    originY: 12,
    requirePreferred: false
  });
  const lines = fs.readFileSync(outputPath, 'utf8').trim().split('\n');
  assert.equal(selected.tiles.length, 96);
  assert.ok(selected.halo.length > 0);
  assert.match(lines[0], /^C3X_BIQ_TERRAIN_WINDOW_V1,12,8,96,/);
  assert.equal(lines.length, 1 + 96 + selected.halo.length);
  assert.ok(selected.halo.some((tile) => tile.column === 12));
});

test('L12-sized diamond exports exactly 192 visible tiles plus deterministic halo', () => {
  const records = makeMap(64, 48, (column, row) =>
    column === 18 && row === 24 ? 10 : ((column + row) % 5 === 0 ? 7 : 2)
  );
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'c3x-biq-l12-'));
  const outputPath = path.join(directory, 'window.csv');
  const options = {
    records,
    windowColumns: 16,
    windowRows: 12,
    preferredReal: ['volcano'],
    windowShape: 'diamond',
    originX: null,
    originY: null,
    requirePreferred: true
  };
  const first = writeViewport(outputPath, {width: 64, height: 48}, options);
  const firstBytes = fs.readFileSync(outputPath);
  const second = writeViewport(outputPath, {width: 64, height: 48}, options);
  const secondBytes = fs.readFileSync(outputPath);
  assert.equal(first.tiles.length, 192);
  assert.equal(first.preferredCount, 1);
  assert.match(firstBytes.toString('utf8').split('\n')[0],
    /^C3X_BIQ_TERRAIN_WINDOW_V1,16,12,192,/);
  assert.deepEqual(secondBytes, firstBytes);
  assert.deepEqual(second.halo, first.halo);
});

test('L13 river fixture selects river-rich terrain and exports V2 topology', () => {
  const records = makeMap(
    64,
    48,
    (column, row) => (column + row) % 7 === 0 ? 5 : 2,
    (column, row) => column >= 10 && column <= 20 && row >= 12 && row <= 34
      ? ((column + row) & 1 ? 2 | 8 : 32 | 128)
      : 0
  );
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'c3x-biq-l13-'));
  const outputPath = path.join(directory, 'window.csv');
  const options = {
    records,
    windowColumns: 16,
    windowRows: 12,
    preferredReal: [],
    windowShape: 'diamond',
    originX: null,
    originY: null,
    requirePreferred: false,
    preferRiver: true,
    requireRiver: true
  };
  const first = writeViewport(outputPath, {width: 64, height: 48}, options);
  const firstBytes = fs.readFileSync(outputPath);
  const second = writeViewport(outputPath, {width: 64, height: 48}, options);
  const lines = firstBytes.toString('utf8').trim().split('\n');
  assert.equal(first.tiles.length, 192);
  assert.ok(first.riverCount >= 12);
  assert.match(lines[0], /^C3X_BIQ_TERRAIN_WINDOW_V2,16,12,192,/);
  assert.ok(lines.slice(1).every((line) => line.split(',').length === 9));
  assert.deepEqual(fs.readFileSync(outputPath), firstBytes);
  assert.equal(second.riverCount, first.riverCount);
});

test('river topology canonicalizes reciprocal BIQ edge flags', () => {
  const tiles = [
    {column: 0, row: 0, base: 2, riverMask: 8},
    {column: 1, row: 0, base: 2, riverMask: 128}
  ];
  const metrics = riverTopologyMetrics(tiles, 2, 1);
  assert.equal(metrics.riverEdgeCount, 1);
  assert.equal(metrics.riverComponentCount, 1);
  assert.equal(metrics.longestRiverEdgeCount, 1);
  assert.equal(metrics.sourceCount, 2);
});

test('selector can require every preferred terrain family', () => {
  const records = makeMap(40, 30, (column, row) => {
    if (column === 8 && row === 12) return 10;
    if (column === 9 && row === 12) return 9;
    return 2;
  });
  const selected = selectViewport(
    records, 40, 30, 12, 8, ['volcano', 'marsh'], 'diamond',
    16, 12, false, false, true, false
  );
  assert.deepEqual(selected.preferredTypesPresent, [9, 10]);
  assert.throws(
    () => selectViewport(
      records, 40, 30, 12, 8, ['volcano', 'jungle'], 'diamond',
      16, 12, false, false, true, false
    ),
    /no complete viewport/
  );
});

test('selector can require a horizontal-wrap inspection viewport', () => {
  const records = makeMap(40, 30, () => 2);
  const selected = selectViewport(
    records, 40, 30, 16, 12, [], 'diamond',
    null, null, false, true, false, true
  );
  assert.equal(selected.tiles.length, 192);
  assert.equal(selected.wrapsHorizontal, true);
  assert.ok(selected.tiles.some((tile) => tile.sourceX < selected.originSourceX));
});
