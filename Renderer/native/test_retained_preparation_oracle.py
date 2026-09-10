"""Portable contracts for deterministic retained-preparation evidence."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from Renderer.native.analyze_navigation_run import compare, digest, inspect
from Renderer.native.record_navigation_evidence import preparation_mode


class RetainedPreparationOracleTests(unittest.TestCase):
    def test_preparation_mode_is_isolated_to_busy_workloads(self):
        self.assertEqual(preparation_mode("replay",None),"baseline")
        self.assertEqual(preparation_mode("replay","oracle"),"oracle")
        self.assertEqual(preparation_mode("session","baseline"),"baseline")
        with self.assertRaisesRegex(ValueError,"only for replay or session"):
            preparation_mode("zoom","oracle")

    @staticmethod
    def schedule():
        bounds=(0,10000000,20000000,28000000,34000000,40000000,44000000,50000000,60000000)
        events=(20000000,22000000,24000000,26000000,28000000,40000000,50000000)
        event_phase=(2,2,2,2,3,5,7)
        result=[]
        for phase in range(8):
            times=[value for value,event in zip(events,event_phase) if event==phase]
            index=0
            while len(times)<25:
                value=bounds[phase]+(2*index+1)*(bounds[phase+1]-bounds[phase])//50
                value=min(value,bounds[phase+1]-1);index+=1
                if value not in times:times.append(value)
            for logical in sorted(times):
                width=({20000000:160,22000000:192,24000000:160,26000000:128}.get(logical,128))
                result.append((phase,logical,events.index(logical) if logical in events else -1,width))
        return result

    def fixture(self,root,mode,*,structural_complete=True,timed_pose_misses=0):
        root.mkdir()
        for name in ("C3XRenderer.dll","biq_preview.exe"):(root/name).write_bytes(name.encode())
        args={"scenario":"replay","width":4,"height":4,"tile_width":128,"waves":"1","idle_units":24,
              "dense_scene":True,"unit_actions":"mixed","replay_samples_per_phase":25,
              "preparation_mode":mode,"region_diagnostics":False,"reflection_ablation":False}
        environment={"C3X_RENDERER_PREVIEW_PREPARATION_MODE":mode,"C3X_RENDERER_PREVIEW_SEASON":"0"}
        receipt={"invocation":"oracle-test","binaries":{n:digest(root/n) for n in ("C3XRenderer.dll","biq_preview.exe")},
                 "inputs":{"scene":"frozen"},"args":args,"environment":environment,"quality_mode":"current"}
        (root/"inputs.json").write_text(json.dumps(receipt))
        header=bytearray(54);header[:2]=b"BM";struct.pack_into("<I",header,10,54)
        struct.pack_into("<ii",header,18,4,-4);struct.pack_into("<H",header,28,32)
        for phase in range(8):(root/f"zoom.bmp.replay-phase-{phase}-128.bmp").write_bytes(header+bytes([phase])*64)
        schedule=self.schedule();lines=[f"REPLAY_PREPARE_BEGIN mode={mode} requests={200 if mode=='oracle' else 0} samples_per_phase=25"]
        retained=1 if mode=="oracle" else 0
        lines.append(f"REPLAY_PREPARE_END status=pass mode={mode} requests={200 if mode=='oracle' else 0} requested=200 capacity_limited=0 memory_safe=1 unique_views=19 unique_poses=2 unit_requests={4800 if mode=='oracle' else 0} ms=10 geometry_admissions={200 if mode=='oracle' else 0} geometry_evictions=0 capacity_geometry_evictions=0 capacity_pose_evictions=1 upload_bytes=100 cleared_viewport={64*retained} cleared_regions=0 cleared_blocks=0 cleared_backdrops=0 cleared_publication=0 retained_geometry={64*retained} retained_natural=0 retained_ground=0 retained_waves={64*retained} retained_pose={64*retained} retained_payload={64*retained} retained_shadow={64*retained} retained_other=0 geometry_entries={retained} pose_entries={retained} wave_entries={retained}")
        lines.append(f"REPLAY_BEGIN mode={mode} clock=logical requests=200 samples_per_phase=25 units_per_zone=24 dense=1 native_presented=0 final_map_cache=cleared")
        for step,(phase,logical,event,width) in enumerate(schedule):
            built=0 if structural_complete and mode=="oracle" else 1
            lines.append(f"REPLAY_FRAME step={step} phase={phase} label=p{phase} event={event} logical_us={logical} x={75+phase} y=39 tile_width={width} units=24 request_hash={1000+step} result=1 ms=20 capture_ms=1 map_ms=10 copy_ms=1 units_ms=8 geometry_ms=2 draw_ms=4 readback_ms=4 built={built} reused=10 evicted=0 upload_bytes={built*64} raster_cached_pixels=0 fallback=0 recoveries=0 fnv64={2000+step}")
        built=sum(0 if structural_complete and mode=="oracle" else 1 for _ in schedule)
        lines.append(f"REPLAY_TIMED_END status=pass mode={mode} frames=200 phase_mask=255 event_mask=127 zoom_mask=7 built={built} evicted=0 upload_bytes={built*64} unit_requests=4800 recoveries=0 structural_complete={int(mode=='oracle' and structural_complete)} snapshots=8")
        for phase in range(8):lines.append(f"REPLAY_PARITY label=phase-{phase} phase={phase} event=-1 tile_width=128 status=pass units=24 bytes=64")
        lines.extend((f"REPLAY_END status=pass mode={mode} frames=200 snapshots=8 verified=8 coverage_complete=1","BIQ 4x4 viewport: 0 fallback"))
        (root/"benchmark.log").write_text("\n".join(lines))
        trace=["[C3X renderer] stage=reset device and all caches"]
        if mode=="oracle":
            trace.extend(f"[C3X renderer] sequence={i} stage=unit-body cache_hit={int(i>=2)} cache_bytes=64" for i in range(4800))
            trace.append("[C3X renderer] stage=oracle-trim retained_geometry=64")
        for frame in range(200):
            trace.append(f"[C3X renderer] sequence={frame} stage=usage-view clock={frame}")
            trace.append(f"[C3X renderer] sequence={frame} stage=animation-phases pose_prepare_ms=1 backdrop_submit_ms=1 animated_submit_ms=1 readback_submit_ms=1 readback_wait_ms=1 cpu_copy_ms=1")
            trace.append(f"[C3X renderer] sequence={frame} stage=animation-frame ms=4 wave_upload_bytes=0 wave_cells_built=0 wave_cells_reused=1 backdrop_hits=0 backdrop_misses=1")
            trace.append(f"[C3X renderer] sequence={frame} stage=navigation-phases contributors_ms=1 lights_ms=1 shadows_ms=1 tile_validation_ms=1 tile_append_ms=1 topology_ms=1")
            for unit in range(24):
                miss=timed_pose_misses>0 and frame*24+unit<timed_pose_misses
                trace.append(f"[C3X renderer] sequence={frame} stage=unit-body cache_hit={int(not miss)} cache_bytes=64")
        trace.append("[C3X renderer] stage=reset device and all caches")
        (root/"renderer.log").write_text("\n".join(trace))
        (root/"completion.txt").write_text("0\n")
        (root/"evidence.json").write_text(json.dumps({"invocation":"oracle-test","returncode":0,
            "inputs_unchanged":True,"binaries_unchanged":True,
            "images":{p.name:digest(p) for p in root.glob("*.bmp")}}))
        return root

    def test_partial_and_capacity_limited_preparation_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"oracle","oracle",structural_complete=False,timed_pose_misses=2)
            report=inspect(root)[1]
            self.assertEqual(report["preparation"]["coverage"],"partial")
            self.assertEqual(report["preparation"]["incomplete_owners"],["structural_geometry","unit_poses"])
            self.assertEqual(report["unit_pose_cache"]["inferred_minimum_preparation_evictions"],1)

    def test_baseline_and_oracle_require_identical_semantic_streams_and_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline=self.fixture(Path(temporary)/"baseline","baseline",structural_complete=False)
            oracle=self.fixture(Path(temporary)/"oracle","oracle")
            result=compare(baseline,oracle)
            self.assertTrue(result["all_images_exact"])
            self.assertTrue(result["oracle_decision"]["streams_identical"])
            log=oracle/"benchmark.log";log.write_text(log.read_text().replace("request_hash=1000","request_hash=999",1))
            with self.assertRaisesRegex(ValueError,"environment or camera requests"):
                compare(baseline,oracle)


if __name__=="__main__":unittest.main()
