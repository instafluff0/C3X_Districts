'use strict';

// Read-only local evidence adapter. It deliberately consumes the editor's
// already-tested BIQ parser instead of adding a second binary parser here.

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

function fail(message) {
  process.stderr.write(`error: ${message}\n`);
  process.exit(2);
}

function argsOf(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i];
    if (!key || !key.startsWith('--') || i + 1 >= argv.length) fail('expected --name value arguments');
    out[key.slice(2)] = argv[i + 1];
  }
  return out;
}

function hash(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function readPediaBindings(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const lines = fs.readFileSync(filePath, 'latin1').replace(/\r/g, '').split('\n');
  const out = [];
  for (let i = 0; i < lines.length; i += 1) {
    const key = lines[i].trim();
    if (!key.toUpperCase().startsWith('#ANIMNAME_PRTO_')) continue;
    let value = '';
    for (let j = i + 1; j < lines.length; j += 1) {
      const candidate = lines[j].trim();
      if (candidate.startsWith('#')) break;
      if (candidate && !candidate.startsWith(';')) { value = candidate; break; }
    }
    if (value) out.push({ key: key.slice(1), art_folder: value });
  }
  return out;
}

function section(parsed, code) {
  const found = parsed.sections.find((item) => item.code === code);
  return found ? found.records : [];
}

function sortedObject(value) {
  if (Array.isArray(value)) return value.map(sortedObject);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sortedObject(value[key])]));
  }
  return value;
}

const args = argsOf(process.argv);
for (const required of ['biq', 'editor-root', 'install-root', 'output']) {
  if (!args[required]) fail(`missing --${required}`);
}

const biqPath = path.resolve(args.biq);
const editorRoot = path.resolve(args['editor-root']);
const installRoot = path.resolve(args['install-root']);
const outputPath = path.resolve(args.output);
let compressed = fs.readFileSync(biqPath);
let inflated = compressed;
if (!compressed.subarray(0, 3).toString('ascii').startsWith('BIC')) {
  const result = require(path.join(editorRoot, 'src', 'biq', 'decompress.js')).decompress(compressed);
  if (!result.ok) fail(result.error || 'BIQ decompression failed');
  inflated = result.data;
}
const parsed = require(path.join(editorRoot, 'src', 'biq', 'biqSections.js')).parseAllSections(inflated);
if (!parsed.ok) fail(parsed.error || 'BIQ parse failed');

const pediaLayers = [
  { id: 'base', file: path.join(installRoot, 'Text', 'PediaIcons.txt') },
  { id: 'ptw', file: path.join(installRoot, 'Civ3PTW', 'Text', 'PediaIcons.txt') },
  { id: 'conquests', file: path.join(installRoot, 'Conquests', 'Text', 'PediaIcons.txt') },
];
const bindingByKey = new Map();
for (const layer of pediaLayers) {
  for (const binding of readPediaBindings(layer.file)) {
    bindingByKey.set(binding.key.toUpperCase(), { ...binding, source_layer: layer.id });
  }
}
const bindings = [...bindingByKey.values()].sort((a, b) => a.key.localeCompare(b.key, 'en', { sensitivity: 'base' }));
const primaries = section(parsed, 'PRTO').filter((record) => Number(record.otherStrategy) === -1);

const units = primaries.map((record) => {
  const civKey = String(record.civilopediaEntry || '');
  const animPrefix = `ANIMNAME_${civKey}`.toUpperCase();
  const variants = bindings.filter((binding) => {
    const key = binding.key.toUpperCase();
    return key === animPrefix || key.startsWith(`${animPrefix}_`);
  });
  const direct = variants.find((binding) => binding.key.toUpperCase() === animPrefix) || null;
  return {
    biq_index: record.index,
    civilopedia_entry: civKey,
    name: String(record.name || ''),
    icon_index: Number(record.iconIndex),
    unit_class: Number(record.unitClass),
    ranged_attack_animations: Boolean(Number(record.unitAbilities) & (1 << 25)),
    direct_art_folder: direct ? direct.art_folder : null,
    art_variants: variants,
  };
});

const resources = section(parsed, 'GOOD').map((record) => ({
  biq_index: record.index,
  civilopedia_entry: String(record.civilopediaEntry || ''),
  name: String(record.name || ''),
  resource_class: Number(record.type),
  icon_index: Number(record.icon),
}));
const terrains = section(parsed, 'TERR').map((record) => ({
  biq_index: record.index,
  civilopedia_entry: String(record.civilopediaEntry || ''),
  name: String(record.name || ''),
  landmark_enabled: Boolean(record.landmarkEnabled),
  landmark_name: String(record.landmarkName || ''),
}));
const civilizations = section(parsed, 'RACE').map((record) => ({
  biq_index: record.index,
  civilopedia_entry: String(record.civilopediaEntry || ''),
  name: String(record.civilizationName || record.name || ''),
  culture_group: Number(record.cultureGroup),
  king_unit: Number(record.kingUnit),
}));
const buildings = section(parsed, 'BLDG').map((record) => {
  const otherChar = Number(record.otherChar) >>> 0;
  const wonderClass = (otherChar & (1 << 2)) !== 0
    ? 'great'
    : ((otherChar & (1 << 3)) !== 0 ? 'small' : 'improvement');
  const isWonder = wonderClass !== 'improvement';
  return {
    biq_index: record.index,
    civilopedia_entry: String(record.civilopediaEntry || ''),
    name: String(record.name || ''),
    wonder_class: wonderClass,
    map_render_classification: isWonder ? 'not_map_rendered_without_c3x_wonder_instance' : 'not_map_rendered',
    evidence: isWonder
      ? 'BIQ otherChar classifies the record; map placement requires an authoritative C3X Wonder District instance'
      : 'BIQ improvements affect city state; Civ III exposes no per-building map sprite selector',
  };
});

const snapshot = {
  schema: 'c3x.vanilla_conquests_biq_semantics.v0',
  source: {
    biq: 'Conquests/conquests.biq',
    compressed_sha256: hash(compressed),
    inflated_sha256: hash(inflated),
    format: 'Conquests',
    major_version: parsed.io.majorVersion,
    minor_version: parsed.io.minorVersion,
    parser: 'C3X_Editor/src/biq/biqSections.js',
    pedia_layers: pediaLayers.filter((layer) => fs.existsSync(layer.file)).map((layer) => layer.id),
  },
  counts: {
    prto_records_including_strategy_rows: section(parsed, 'PRTO').length,
    primary_unit_types: units.length,
    resources: resources.length,
    terrain_types: terrains.length,
    civilizations: civilizations.length,
    improvements_and_wonders: buildings.length,
    animation_bindings: bindings.length,
  },
  terrain_types: terrains,
  resources,
  unit_types: units,
  animation_bindings: bindings,
  civilizations,
  city_selectors: {
    culture_groups: [...new Set(civilizations.map((item) => item.culture_group).filter((value) => value >= 0))].sort((a, b) => a - b),
    eras: [0, 1, 2, 3],
    sizes: ['town', 'city', 'metropolis'],
    additional_states: ['walled', 'destroyed', 'airport', 'harbor', 'barracks', 'spy_agency', 'capital', 'embassy'],
  },
  improvements_and_wonders: buildings,
  terrain_transformations: section(parsed, 'TFRM').map((record) => ({
    biq_index: record.index,
    civilopedia_entry: String(record.civilopediaEntry || ''),
    name: String(record.name || ''),
    order: String(record.order || ''),
  })),
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(sortedObject(snapshot), null, 2)}\n`, 'utf8');
process.stdout.write(`${JSON.stringify(snapshot.counts)}\n`);
