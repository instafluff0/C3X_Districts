#!/usr/bin/env node
'use strict';

// Convert a BIQ map to a tiny, source-independent CSV consumed by the native
// preview runner. Parsing is delegated to the neighboring C3X Editor's tested
// BIQ implementation; no copied binary-layout guesses live in the renderer.

const fs = require('fs');
const path = require('path');

function fieldInt(record, key, fallback = 0) {
  const direct = record && record[key];
  const parsed = Number.parseInt(String(direct == null ? '' : direct), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function decodeTerrain(record) {
  let packed = fieldInt(record, 'c3cbaserealterrain', -1);
  if (packed < 0) packed = fieldInt(record, 'baserealterrain', 2);
  packed &= 0xff;
  if (packed <= 0x0f) return { base: packed, real: packed };
  return { base: packed & 0x0f, real: (packed >>> 4) & 0x0f };
}

const TERRAIN_CODES = Object.freeze({
  desert: 0,
  plains: 1,
  grassland: 2,
  tundra: 3,
  floodplain: 4,
  hills: 5,
  mountain: 6,
  forest: 7,
  jungle: 8,
  marsh: 9,
  volcano: 10,
  coast: 11,
  sea: 12,
  ocean: 13
});

function tileValues(record) {
  const terrain = decodeTerrain(record);
  return {
    sourceX: fieldInt(record, 'xpos'),
    sourceY: fieldInt(record, 'ypos'),
    base: terrain.base,
    real: terrain.real,
    bonus: fieldInt(record, 'c3cbonuses', fieldInt(record, 'bonuses')) >>> 0,
    overlays: fieldInt(record, 'c3coverlays', fieldInt(record, 'overlays')) >>> 0,
    // TILE stores river connections in four non-adjacent bits. Other bits in
    // the byte are unrelated/unknown and must not influence topology scoring.
    riverMask: (fieldInt(record, 'riverconnectioninfo',
      fieldInt(record, 'river_connection_info')) >>> 0) & 0xaa
  };
}

function terrainCode(value) {
  const key = String(value == null ? '' : value).trim().toLowerCase();
  if (Object.prototype.hasOwnProperty.call(TERRAIN_CODES, key)) return TERRAIN_CODES[key];
  const parsed = Number.parseInt(key, 10);
  if (Number.isInteger(parsed) && parsed >= 0 && parsed <= 15) return parsed;
  throw new Error(`Unknown terrain preference: ${value}`);
}

function riverTopologyMetrics(tiles, windowColumns, windowRows) {
  const edges = new Map();
  const waterNodes = new Set();
  for (const tile of tiles) {
    const centerX = tile.column + tile.row;
    const centerY = tile.column - tile.row;
    if (tile.base >= TERRAIN_CODES.coast) {
      for (const [dx, dy] of [[0, -1], [1, 0], [0, 1], [-1, 0]]) {
        waterNodes.add(`${centerX + dx},${centerY + dy}`);
      }
    }
    const candidates = [
      [2, 0, tile.column, tile.row],
      [8, 1, tile.column, tile.row],
      [32, 0, tile.column, tile.row - 1],
      [128, 1, tile.column - 1, tile.row]
    ];
    for (const [bit, family, column, row] of candidates) {
      if ((tile.riverMask & bit) === 0) continue;
      const key = `${family},${column},${row}`;
      const latticeX = column + row;
      const latticeY = column - row;
      edges.set(key, family === 0
        ? [`${latticeX},${latticeY - 1}`, `${latticeX + 1},${latticeY}`]
        : [`${latticeX + 1},${latticeY}`, `${latticeX},${latticeY + 1}`]);
    }
  }
  const adjacency = new Map();
  for (const [edge, endpoints] of edges) {
    for (const endpoint of endpoints) {
      if (!adjacency.has(endpoint)) adjacency.set(endpoint, []);
      adjacency.get(endpoint).push(edge);
    }
  }
  let sourceCount = 0;
  let mouthCount = 0;
  let junctionCount = 0;
  for (const [node, connected] of adjacency) {
    if (connected.length === 1) {
      if (waterNodes.has(node)) mouthCount += 1;
      else sourceCount += 1;
    }
    if (connected.length >= 3) junctionCount += 1;
  }
  const visited = new Set();
  const componentSizes = [];
  for (const first of edges.keys()) {
    if (visited.has(first)) continue;
    const pending = [first];
    let size = 0;
    visited.add(first);
    while (pending.length > 0) {
      const edge = pending.pop();
      size += 1;
      for (const endpoint of edges.get(edge)) {
        for (const adjacent of adjacency.get(endpoint) || []) {
          if (!visited.has(adjacent)) {
            visited.add(adjacent);
            pending.push(adjacent);
          }
        }
      }
    }
    componentSizes.push(size);
  }
  componentSizes.sort((a, b) => b - a);
  return {
    riverEdgeCount: edges.size,
    riverComponentCount: componentSizes.length,
    longestRiverEdgeCount: componentSizes[0] || 0,
    sourceCount,
    mouthCount,
    junctionCount,
    windowColumns,
    windowRows
  };
}

function selectViewport(records, mapWidth, mapHeight, windowColumns, windowRows,
                        preferredReal = [], windowShape = 'rectangle',
                        forcedOriginX = null, forcedOriginY = null,
                        preferRiver = false, preferWrap = false,
                        requireAllPreferred = false, requireWrap = false) {
  if (!Number.isInteger(mapWidth) || mapWidth < 2 || (mapWidth & 1) !== 0 ||
      !Number.isInteger(mapHeight) || mapHeight < 1) {
    throw new Error('BIQ map dimensions are invalid');
  }
  const mapColumns = mapWidth / 2;
  if (!Number.isInteger(windowColumns) || windowColumns < 1 || windowColumns > mapColumns ||
      !Number.isInteger(windowRows) || windowRows < 1 || windowRows > mapHeight ||
      !['rectangle', 'diamond'].includes(windowShape)) {
    throw new Error('Requested BIQ viewport does not fit the map');
  }
  const byCoordinate = new Map();
  for (const record of records) {
    const tile = tileValues(record);
    byCoordinate.set(`${tile.sourceX},${tile.sourceY}`, tile);
  }
  const preferred = new Set(preferredReal.map(terrainCode));
  const hasForcedOrigin = forcedOriginX !== null || forcedOriginY !== null;
  if (hasForcedOrigin &&
      (!Number.isInteger(forcedOriginX) || !Number.isInteger(forcedOriginY) ||
       forcedOriginX < 0 || forcedOriginX >= mapWidth ||
       forcedOriginY < 0 || forcedOriginY >= mapHeight ||
       (forcedOriginX & 1) !== (forcedOriginY & 1))) {
    throw new Error('Forced BIQ origin is invalid or does not name a real tile');
  }
  let best = null;
  const firstOriginRow = hasForcedOrigin ? forcedOriginY :
    (windowShape === 'diamond' ? windowRows - 1 : 0);
  const lastOriginRow = hasForcedOrigin ? forcedOriginY : (windowShape === 'diamond'
    ? mapHeight - windowColumns
    : mapHeight - windowRows);
  for (let originRow = firstOriginRow; originRow <= lastOriginRow; originRow += 1) {
    for (let originColumn = 0; originColumn < mapColumns; originColumn += 1) {
      const originX = originColumn * 2 + (originRow & 1);
      if (hasForcedOrigin && originX !== forcedOriginX) continue;
      const tiles = [];
      let valid = true;
      for (let row = 0; valid && row < windowRows; row += 1) {
        for (let column = 0; column < windowColumns; column += 1) {
          const sourceY = windowShape === 'diamond'
            ? originRow + column - row
            : originRow + row;
          const sourceX = windowShape === 'diamond'
            ? (originX + column + row + mapWidth) % mapWidth
            : ((originColumn + column) % mapColumns) * 2 + (sourceY & 1);
          const tile = byCoordinate.get(`${sourceX},${sourceY}`);
          if (!tile) {
            valid = false;
            break;
          }
          tiles.push({...tile, column, row});
        }
      }
      if (!valid) continue;

      let preferredCount = 0;
      const preferredTypesPresent = new Set();
      let riverCount = 0;
      let transitions = 0;
      let landWaterEdges = 0;
      const realTypes = new Set();
      const baseTypes = new Set();
      for (const tile of tiles) {
        realTypes.add(tile.real);
        baseTypes.add(tile.base);
        if (preferred.has(tile.real)) {
          preferredCount += 1;
          preferredTypesPresent.add(tile.real);
        }
        if (tile.riverMask !== 0) riverCount += 1;
        const neighbors = [[tile.column - 1, tile.row], [tile.column, tile.row - 1]];
        for (const [column, row] of neighbors) {
          if (column < 0 || row < 0) continue;
          const neighbor = tiles[row * windowColumns + column];
          if (neighbor.real !== tile.real) transitions += 1;
          if ((neighbor.base >= TERRAIN_CODES.coast) !== (tile.base >= TERRAIN_CODES.coast)) {
            landWaterEdges += 1;
          }
        }
      }
      if (requireAllPreferred && preferredTypesPresent.size !== preferred.size) continue;
      // Six preferred cells are enough to judge a gate. Beyond that point,
      // reward surrounding BIQ variety and transitions more heavily than a
      // monoculture, so each promotion retains useful prior-layer coverage.
      const preferredCoverage = requireAllPreferred
        ? preferredTypesPresent.size * 700 +
          Math.min(Math.max(0, preferredCount - preferredTypesPresent.size), 4) * 30
        : Math.min(preferredCount, 6) * 700 +
          Math.max(0, preferredCount - 6) * 30;
      const riverTopology = riverTopologyMetrics(tiles, windowColumns, windowRows);
      const wrapsHorizontal = windowShape === 'diamond'
        ? originX + windowColumns - 1 + windowRows - 1 >= mapWidth
        : originColumn + windowColumns > mapColumns;
      if (requireWrap && !wrapsHorizontal) continue;
      const score = preferredCoverage + transitions * 10 + landWaterEdges * 30 +
        realTypes.size * 180 + baseTypes.size * 260 +
        (preferRiver
          ? Math.min(riverCount, 12) * 300 + Math.max(0, riverCount - 12) * 12 +
            Math.min(riverTopology.riverEdgeCount, 30) * 80 +
            Math.min(riverTopology.longestRiverEdgeCount, 18) * 260 -
            Math.max(0, riverTopology.riverComponentCount - 3) * 80 +
            Math.min(riverTopology.sourceCount, 3) * 180 +
            Math.min(riverTopology.mouthCount, 3) * 260 +
            Math.min(riverTopology.junctionCount, 3) * 340
          : 0) + (preferWrap && wrapsHorizontal ? 5000 : 0);
      const candidate = {
        originColumn,
        originRow,
        originSourceX: originColumn * 2 + (originRow & 1),
        originSourceY: originRow,
        windowShape,
        preferredCount,
        preferredTypesPresent: [...preferredTypesPresent].sort((a, b) => a - b),
        riverCount,
        ...riverTopology,
        wrapsHorizontal,
        score,
        tiles
      };
      if (!best || candidate.score > best.score ||
          (candidate.score === best.score && candidate.originRow < best.originRow) ||
          (candidate.score === best.score && candidate.originRow === best.originRow &&
           candidate.originColumn < best.originColumn)) {
        best = candidate;
      }
    }
  }
  if (!best) throw new Error('BIQ map contains no complete viewport of the requested size');
  return best;
}

function writeViewport(outputPath, result, options) {
  const selected = selectViewport(
    options.records, result.width, result.height,
    options.windowColumns, options.windowRows, options.preferredReal, options.windowShape,
    options.originX, options.originY, options.preferRiver === true,
    options.preferWrap === true, options.requireAllPreferred === true,
    options.requireWrap === true
  );
  if (options.requirePreferred && selected.preferredCount === 0) {
    throw new Error(`No requested terrain occurs in any ${options.windowColumns}x${options.windowRows} viewport`);
  }
  if (options.requireRiver && selected.riverCount === 0) {
    throw new Error(`No river occurs in any ${options.windowColumns}x${options.windowRows} viewport`);
  }
  if (options.requireWrap && !selected.wrapsHorizontal) {
    throw new Error(`No horizontal wrap boundary occurs in the selected ` +
      `${options.windowColumns}x${options.windowRows} viewport`);
  }
  const count = options.windowColumns * options.windowRows;
  const byCoordinate = new Map();
  for (const record of options.records) {
    const tile = tileValues(record);
    byCoordinate.set(`${tile.sourceX},${tile.sourceY}`, tile);
  }
  const halo = [];
  // Two cells are required because the filtered shoreline samples the four
  // neighbors of each interpolated center. Preserve those authoritative BIQ
  // values instead of treating the edge of the inspection crop as map edge.
  for (let row = -2; row < options.windowRows + 2; row += 1) {
    for (let column = -2; column < options.windowColumns + 2; column += 1) {
      if (column >= 0 && column < options.windowColumns &&
          row >= 0 && row < options.windowRows) continue;
      const sourceY = selected.windowShape === 'diamond'
        ? selected.originSourceY + column - row
        : selected.originSourceY + row;
      if (sourceY < 0 || sourceY >= result.height) continue;
      const sourceX = selected.windowShape === 'diamond'
        ? (selected.originSourceX + column + row + result.width * 2) % result.width
        : (((selected.originColumn + column) % (result.width / 2) + result.width / 2) %
            (result.width / 2)) * 2 + (sourceY & 1);
      const tile = byCoordinate.get(`${sourceX},${sourceY}`);
      if (tile) halo.push({...tile, column, row});
    }
  }
  const includeRiver = options.includeRiver === true || options.preferRiver === true ||
    options.requireRiver === true;
  const lines = [
    `${includeRiver ? 'C3X_BIQ_TERRAIN_WINDOW_V2' : 'C3X_BIQ_TERRAIN_WINDOW_V1'},` +
      `${options.windowColumns},${options.windowRows},${count},` +
      `${selected.originColumn},${selected.originRow},${result.width},${result.height},${halo.length}`
  ];
  for (const tile of [...selected.tiles, ...halo]) {
    lines.push(
      `${tile.column},${tile.row},${tile.sourceX},${tile.sourceY},${tile.base},${tile.real},` +
      `${tile.bonus},${tile.overlays}${includeRiver ? `,${tile.riverMask}` : ''}`
    );
  }
  fs.mkdirSync(path.dirname(path.resolve(outputPath)), {recursive: true});
  fs.writeFileSync(outputPath, `${lines.join('\n')}\n`, 'utf8');
  return {...selected, halo};
}

function exportScene(biqPath, outputPath, options = null) {
  const editorCore = path.resolve(__dirname, '..', '..', '..', 'C3X_Editor', 'src', 'configCore');
  const { loadMapImport } = require(editorCore);
  const result = loadMapImport({
    civ3Path: path.resolve(__dirname, '..', '..', '..'),
    scenarioPath: path.resolve(biqPath),
    textEncoding: 'windows-1252'
  });
  const section = result.importedSections.find((item) => item.code === 'TILE');
  if (!section || !Array.isArray(section.records) || section.records.length !== result.tileCount) {
    throw new Error('Parsed BIQ did not expose its complete TILE section');
  }
  if (options && options.windowColumns && options.windowRows) {
    const selected = writeViewport(outputPath, result, {
      ...options,
      records: section.records
    });
    return {...result, selectedViewport: selected};
  }
  const lines = [`C3X_BIQ_TERRAIN_V0,${result.width},${result.height},${result.tileCount}`];
  for (const record of section.records) {
    const tile = tileValues(record);
    lines.push(`${tile.sourceX},${tile.sourceY},${tile.base},${tile.real},${tile.bonus},${tile.overlays}`);
  }
  fs.mkdirSync(path.dirname(path.resolve(outputPath)), { recursive: true });
  fs.writeFileSync(outputPath, `${lines.join('\n')}\n`, 'utf8');
  return result;
}

if (require.main === module) {
  const biqPath = process.argv[2];
  const outputPath = process.argv[3];
  if (!biqPath || !outputPath) {
    console.error('usage: node export_biq_terrain_scene.js <scenario.biq> <output.csv> ' +
      '[--window-columns 12 --window-rows 8] [--window-shape diamond] ' +
      '[--origin-x 53 --origin-y 55] [--prefer-real marsh] [--require-preferred] ' +
      '[--require-all-preferred] ' +
      '[--prefer-river] [--require-river] [--prefer-wrap] [--require-wrap]');
    process.exit(2);
  }
  try {
    const args = process.argv.slice(4);
    function optionValue(name, fallback) {
      const index = args.indexOf(name);
      return index >= 0 && index + 1 < args.length ? args[index + 1] : fallback;
    }
    const windowColumns = Number.parseInt(optionValue('--window-columns', '0'), 10);
    const windowRows = Number.parseInt(optionValue('--window-rows', '0'), 10);
    if ((windowColumns > 0) !== (windowRows > 0)) {
      throw new Error('--window-columns and --window-rows must be supplied together');
    }
    const preferredReal = args.reduce((values, argument, index) => {
      if (argument === '--prefer-real' && index + 1 < args.length) values.push(args[index + 1]);
      return values;
    }, []);
    const windowShape = optionValue('--window-shape', 'rectangle');
    const originXText = optionValue('--origin-x', '');
    const originYText = optionValue('--origin-y', '');
    if ((originXText !== '') !== (originYText !== '')) {
      throw new Error('--origin-x and --origin-y must be supplied together');
    }
    const originX = originXText === '' ? null : Number.parseInt(originXText, 10);
    const originY = originYText === '' ? null : Number.parseInt(originYText, 10);
    const result = exportScene(biqPath, outputPath, windowColumns > 0 ? {
      windowColumns,
      windowRows,
      windowShape,
      originX,
      originY,
      preferredReal,
      requirePreferred: args.includes('--require-preferred'),
      requireAllPreferred: args.includes('--require-all-preferred'),
      preferRiver: args.includes('--prefer-river'),
      requireRiver: args.includes('--require-river'),
      preferWrap: args.includes('--prefer-wrap'),
      requireWrap: args.includes('--require-wrap')
    } : null);
    if (result.selectedViewport) {
      const view = result.selectedViewport;
      console.log(`Selected ${windowColumns * windowRows} BIQ tiles as a ${view.windowShape} ` +
        `from raw BIQ origin ${view.originSourceX},${view.originSourceY} ` +
        `(score=${view.score}, preferred=${view.preferredCount}, rivers=${view.riverCount}, ` +
        `edges=${view.riverEdgeCount}, longest=${view.longestRiverEdgeCount}, ` +
        `sources=${view.sourceCount}, mouths=${view.mouthCount}, junctions=${view.junctionCount}, ` +
        `wrap=${view.wrapsHorizontal}) ` +
        `from ${result.width}x${result.height} map into ${outputPath}`);
    } else {
      console.log(`Exported ${result.tileCount} BIQ tiles (${result.width}x${result.height}) to ${outputPath}`);
    }
  } catch (error) {
    console.error(`error: ${error.message}`);
    process.exit(1);
  }
}

module.exports = {
  TERRAIN_CODES,
  decodeTerrain,
  exportScene,
  riverTopologyMetrics,
  selectViewport,
  terrainCode,
  tileValues,
  writeViewport
};
