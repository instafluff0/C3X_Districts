#!/usr/bin/env node
'use strict';
// Offline adapter only. Runtime replay consumes generic immutable tile records.
const fs=require('fs'), path=require('path'), crypto=require('crypto');
const root=path.resolve(__dirname,'../../../..');
const editor=process.env.C3X_LAB_EDITOR_CORE || path.resolve(root,'../C3X_Editor/src/configCore.js');
const {loadMapImport}=require(editor);
const {tileValues}=require(path.join(root,'Renderer/tools/export_biq_terrain_scene.js'));
const source=path.resolve(process.argv[2]), output=path.resolve(process.argv[3]);
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const before=fs.readFileSync(source);
const result=loadMapImport({scenarioPath:source,textEncoding:'windows-1252'});
const sections=Object.fromEntries(result.importedSections.map(s=>[s.code,s.records]));
const map=sections.WMAP[0], flags=Number(map.flags);
if (!Number.isInteger(flags)) throw Error('Missing authoritative WMAP flags');
if (sections.TILE.length!==result.width*result.height/2) throw Error('Incomplete source map');
const dependencies=Object.keys(require.cache).filter(p=>fs.statSync(p).isFile()).map(p=>[path.relative(path.dirname(editor),p),hash(fs.readFileSync(p))]).sort((a,b)=>a[0].localeCompare(b[0]));
const parserHash=hash(Buffer.from(JSON.stringify(dependencies)));
const payload={schema:'c3x.lab_v2.biq_dataset.v1',source:{name:'test.biq',sha256:hash(before),bytes:before.length},parser:{adapter:1,closure_sha256:parserHash,text_encoding:'windows-1252'},width:result.width,height:result.height,wrap_x:!!(flags&1),wrap_y:!!(flags&2),wmap_flags:flags,tiles:sections.TILE.map(r=>({...tileValues(r),source_fields:Object.fromEntries(Object.entries(r).filter(([k,v])=>k!=='fields'&&typeof v!=='object'))})),section_counts:Object.fromEntries(Object.entries(sections).map(([k,v])=>[k,v.length]))};
if (hash(fs.readFileSync(source))!==payload.source.sha256) throw Error('Source changed during import');
fs.mkdirSync(path.dirname(output),{recursive:true});
fs.writeFileSync(output,JSON.stringify(payload));
console.log(JSON.stringify({source:payload.source,parser:payload.parser,width:payload.width,height:payload.height,wrap_x:payload.wrap_x,wrap_y:payload.wrap_y,section_counts:payload.section_counts}));
