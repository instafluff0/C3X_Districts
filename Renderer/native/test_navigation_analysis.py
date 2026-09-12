"""Evidence must reject stale, partial, changed-quality and changed-camera runs."""
import json
import gzip
from pathlib import Path
import struct
import tempfile
import unittest

from Renderer.native.analyze_navigation_run import compare, digest, distribution, inspect, endpoint_accounting, session_accounting


class NavigationAnalysisTests(unittest.TestCase):
    def test_endpoint_accounting_orders_and_excludes_preparation(self):
        lines=["TIMING_SETUP schema=1 frequency=1000 process_enter=0 source_done=10 dll_done=20 definitions_done=30 initial_begin=40 initial_done=100 dropped=0",
               "TIMING_REQUEST id=0 capture_begin=30 capture_end=35 caller_enter=40 caller_return=99 correct_done=100 geometry_ticks=20 draw_ticks=10 readback_ticks=5 result=1 tiles=12",
               "TIMING_REQUEST id=1 capture_begin=110 capture_end=115 caller_enter=116 caller_return=140 correct_done=141 geometry_ticks=5 draw_ticks=6 readback_ticks=10 result=1 tiles=12"]
        trace=["qpc=117 sequence=2 stage=render-begin",
               "qpc=139 sequence=2 stage=submission-phases map_wait_ms=8 cpu_copy_ms=1",
               "qpc=200 sequence=3 stage=gpu-timing sample_sequence=2 valid=1 gpu_draw_ms=12 gpu_copy_ms=1"]
        report=endpoint_accounting(lines,trace)
        self.assertEqual(1,report["playback_samples"])
        r=report["requests"][1]
        self.assertEqual(5,r["capture_ms"])
        self.assertEqual(31,r["request_to_checked_result_ms"])
        self.assertEqual(3,r["unexplained_caller_ms"])
        self.assertEqual(12,r["nested_diagnostics"]["gpu_execution_ms"])
        self.assertFalse(r["accounting_complete"])
        self.assertEqual("initial_preparation",report["requests"][0]["role"])
        with self.assertRaisesRegex(ValueError,"order"):
            endpoint_accounting([l.replace("capture_end=115","capture_end=150") for l in lines])
        with self.assertRaisesRegex(ValueError,"duplicate"):
            endpoint_accounting(lines+[lines[-1]])
        call="qpc=140 stage=call-endpoints entered=116 locked=117 submitted=118 worker_begin=119 worker_rendered=138 worker_published=139 returned=140 queued=1"
        complete=endpoint_accounting(lines,trace+[call])["requests"][1]
        self.assertEqual(0,complete["unexplained_caller_ms"])
        self.assertEqual(1,complete["caller_cpu_spans_ms"]["lock_wait"])
        self.assertEqual(1,complete["caller_cpu_spans_ms"]["worker_publication_and_preparation"])
        self.assertTrue(complete["accounting_complete"])
        with self.assertRaisesRegex(ValueError,"order"):
            endpoint_accounting(lines,trace+[call.replace("worker_begin=119","worker_begin=141")])
        missing=endpoint_accounting(lines)["requests"][1]
        self.assertIsNone(missing["nested_diagnostics"]["gpu_execution_ms"])
        self.assertEqual("unmeasured",endpoint_accounting([])["status"])

    def test_persistent_cases_require_reset_and_input_checks(self):
        body=["TIMING_SETUP schema=1 frequency=1000 process_enter=0 source_done=10 dll_done=20 definitions_done=30 initial_begin=40 initial_done=100 dropped=0",
              "TIMING_REQUEST id=0 capture_begin=30 capture_end=35 caller_enter=40 caller_return=99 correct_done=100 geometry_ticks=20 draw_ticks=10 readback_ticks=5 result=1 tiles=12",
              "TIMING_REQUEST id=1 capture_begin=110 capture_end=115 caller_enter=116 caller_return=140 correct_done=141 geometry_ticks=5 draw_ticks=6 readback_ticks=10 result=1 tiles=12",
              "SCROLL_ABLATION delta_columns=4"]
        reset="CASE_RESET mode=1 result=1 ms=2 budgets=unchanged geometry_evictions=0 pose_evictions=0 retained_geometry=0 retained_natural=0 retained_ground=0 geometry_entries=0"
        check="CASE_INPUT_CHECK unchanged=1 paths=2 ms=1"
        lines=[check,"CASE_BEGIN id=scroll-0 config=config request_digest=digest reset=assets_loaded warmup=initial",reset,
               "CASE_PLAYBACK_BEGIN",*body,"CASE_END id=scroll-0 result=0",check,
               "CASE_SESSION_END result=0 completed=1 requested=1 elapsed_ms=200 time_limit_exceeded=0"]
        report=session_accounting(lines)
        self.assertEqual([4],report["cases"][0]["request_offsets"])
        self.assertEqual(1,report["cases"][0]["endpoints"]["playback_samples"])
        for before,after in (("unchanged=1","unchanged=0"),("retained_geometry=0","retained_geometry=1"),
                             ("time_limit_exceeded=0","time_limit_exceeded=1"),("mode=1","mode=2")):
            with self.assertRaises(ValueError):session_accounting([l.replace(before,after) for l in lines])
        with self.assertRaisesRegex(ValueError,"not checked"):
            session_accounting([l for l in lines if l!=check])

    def test_quick_or_changed_source_receipts_cannot_pass_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            original=json.loads((root/"evidence.json").read_text())
            for change in ({"provisional":True},{"sources_unchanged":False}):
                (root/"evidence.json").write_text(json.dumps(dict(original,**change)))
                with self.assertRaisesRegex(ValueError,"Unverified"):inspect(root)

    def test_busy_session_reports_skipped_phases_and_excludes_cold_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            receipt=json.loads((root/"inputs.json").read_text())
            receipt["args"].update(scenario="session",waves="1",idle_units=24,dense_scene=True,unit_actions="mixed")
            (root/"inputs.json").write_text(json.dumps(receipt))
            schedule=[(0,900,0,"idle"),(10000000,10000900,1,"scroll"),(59000000,60001000,7,"return_idle")]
            lines=["SESSION_BEGIN duration_us=60000000 input_slot_us=33333 clock=wall unit_warmup=0 native_presented=0 dense=1 units_per_zone=24 initial_render_ms=100"]
            trace=["[C3X renderer] stage=usage-view clock=1000000"]
            previous=-1;skipped=0
            for i,(dispatch,done,phase,label) in enumerate(schedule):
                slot=dispatch//33333;gap=slot-previous-1;skipped+=gap;previous=slot
                lines.append(f"SESSION_FRAME frame={i} phase={phase} label={label} dispatch_us={dispatch} done_us={done} requested_us={slot*33333} skipped_slots={gap} superseded=0 tile_width=128 x=75 y=39 units=24 moving=6 attacking=6 fortifying=6 idling=6 result=1 ms={(done-dispatch)/1000} capture_ms=0 map_ms=0 copy_ms=0 units_ms=0 built=0 reused=1 upload_bytes=0 recoveries=0")
                trace.append(f"[C3X renderer] stage=usage-view clock={1000000+dispatch} cities=3")
            lines.extend([f"SESSION_TIMED_END status=pass frames=3 wall_ms=60001.1 skipped_slots={skipped} phase_mask=131 zoom_mask=1 coverage_complete=0 snapshot_bytes=64 evidence_ms=1",
                          "SESSION_PARITY phase=0 tile_width=128 status=pass",
                          "SESSION_END status=pass frames=3 snapshots=1 verified=1 coverage_complete=0",
                          "BIQ 100x100 viewport: 0 fallback"])
            trace.extend(["[C3X renderer] stage=usage-view clock=1000000 cities=99",
                          "[C3X renderer] stage=animation-phases pose_prepare_ms=99999"])
            (root/"renderer.log").write_text("\n".join(trace))
            (root/"benchmark.log").write_text("\n".join(lines))
            data=(root/"zoom.bmp.resident0.bmp").read_bytes()
            for name in ("zoom.bmp","zoom.bmp.session-0-128.bmp"):(root/name).write_bytes(data)
            completion=json.loads((root/"evidence.json").read_text())
            completion["images"]={p.name:digest(p) for p in root.glob("*.bmp")}
            (root/"evidence.json").write_text(json.dumps(completion))
            archived=root/"zoom.bmp.session-0-128.bmp"
            archived.with_suffix(".bmp.gz").write_bytes(gzip.compress(archived.read_bytes()))
            archived.unlink()
            _,report=inspect(root)
            self.assertFalse(report["session"]["schedule_coverage_complete"])
            self.assertEqual([2,3,4,5,6],report["session"]["missing_phases"])
            self.assertEqual(3,report["session_trace_coverage"]["aligned_views"])
            self.assertEqual(3,report["session"]["workloads"]["idle"]["maximum_logged_objects"]["cities"])
            self.assertNotIn("animation_phases",report)
            (root/"benchmark.log").write_text("\n".join(lines).replace("coverage_complete=0","coverage_complete=1"))
            with self.assertRaisesRegex(ValueError,"coverage"):inspect(root)

    def test_busy_session_preserves_delayed_discrete_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            receipt=json.loads((root/"inputs.json").read_text())
            receipt["args"].update(scenario="session",waves="1",idle_units=24,dense_scene=True,unit_actions="mixed")
            (root/"inputs.json").write_text(json.dumps(receipt))
            events=[20000000,22000000,24000000,26000000,28000000,40000000,50000000]
            widths=[160,192,160,128,128,128,128];phases=[2,2,2,2,3,5,7]
            lines=["SESSION_BEGIN duration_us=60000000 input_slot_us=33333 clock=wall input_model=queued_discrete_v1 unit_warmup=0 native_presented=0 dense=1 units_per_zone=24 initial_render_ms=100"]
            previous=-1;skipped=0
            for i,event in enumerate(events):
                dispatch=65000000+i*100000;slot=dispatch//33333;gap=slot-previous-1;skipped+=gap;previous=slot
                lines.append(f"SESSION_FRAME frame={i} phase={phases[i]} label=queued dispatch_us={dispatch} done_us={dispatch+1000} requested_us={event} input_event={i} dispatch_delay_us={dispatch-event} skipped_slots={gap} superseded=0 tile_width={widths[i]} x=75 y=39 units=24 moving=6 attacking=6 fortifying=6 idling=6 result=1 ms=1 capture_ms=0 map_ms=0 copy_ms=0 units_ms=0 built=0 reused=1 upload_bytes=0 recoveries=0")
            lines.extend([f"SESSION_TIMED_END status=pass frames=7 wall_ms=65602 skipped_slots={skipped} phase_mask=172 zoom_mask=7 discrete_events=7 coverage_complete=0 snapshot_bytes=64 evidence_ms=1",
                          "SESSION_PARITY phase=0 tile_width=128 status=pass",
                          "SESSION_END status=pass frames=7 snapshots=1 verified=1 coverage_complete=0",
                          "BIQ 100x100 viewport: 0 fallback"])
            (root/"benchmark.log").write_text("\n".join(lines))
            data=(root/"zoom.bmp.resident0.bmp").read_bytes()
            for name in ("zoom.bmp","zoom.bmp.session-0-128.bmp"):(root/name).write_bytes(data)
            completion=json.loads((root/"evidence.json").read_text());completion["images"]={p.name:digest(p) for p in root.glob("*.bmp")}
            (root/"evidence.json").write_text(json.dumps(completion))
            (root/"renderer.log").write_text("")
            _,report=inspect(root)
            self.assertEqual(7,len(report["session"]["discrete_events"]))
            self.assertEqual(45000,report["session"]["discrete_events"][0]["dispatch_delay_ms"])
            self.assertEqual(5602,report["session"]["post_input_settle_ms"])
            (root/"benchmark.log").write_text("\n".join(lines).replace("input_event=1","input_event=2"))
            with self.assertRaisesRegex(ValueError,"event order"):inspect(root)

    def fixture(self, root):
        root.mkdir()
        for name in ("C3XRenderer.dll", "biq_preview.exe"):
            (root / name).write_bytes(name.encode())
        args = dict(scenario="navigation", resident=True, resident_steps=14, width=4, height=4,
                    tile_width=128, waves="0", dependency_control=False)
        receipt = dict(invocation="current", binaries={n: digest(root/n) for n in ("C3XRenderer.dll", "biq_preview.exe")},
                       inputs={"scene": "frozen"}, args=args, environment={"C3X_RENDERER_PREVIEW_SEASON": "0"})
        (root / "inputs.json").write_text(json.dumps(receipt))
        lines = []
        for cycle in range(2):
            for step in range(6):
                lines.append(f"NAV cycle={cycle} step={step} result=1")
                if cycle:
                    lines.append(f"NAV parity step={step} status=pass")
        header = bytearray(54)
        header[:2] = b"BM"
        struct.pack_into("<I", header, 10, 54)
        struct.pack_into("<ii", header, 18, 4, -4)
        struct.pack_into("<H", header, 28, 32)
        for i in range(14):
            lines.append(f"RESIDENT_NAV step={i} x=35 y={41+i*2} pixel_y={(i+1)*64} result=1 built=0 reused=10 upload_bytes=0 ms=10 capture_ms=1 geometry_ms=2 draw_ms=3 readback_ms=4")
            (root / f"zoom.bmp.resident{i}.bmp").write_bytes(header + bytes([i])*64)
        lines.extend(("RESIDENT_END status=pass", "BIQ 100x100 viewport: 0 fallback"))
        (root / "benchmark.log").write_text("\n".join(lines))
        (root / "renderer.log").write_text("")
        (root / "completion.txt").write_text("0\n")
        (root / "evidence.json").write_text(json.dumps(dict(invocation="current", returncode=0,
            inputs_unchanged=True, binaries_unchanged=True, images={p.name: digest(p) for p in root.glob("*.bmp")})))
        return root

    def test_pair_allows_only_cache_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            a=self.fixture(Path(temporary)/"a");b=self.fixture(Path(temporary)/"b")
            data=json.loads((b/"inputs.json").read_text())
            data["args"]["dependency_control"]=True
            (b/"inputs.json").write_text(json.dumps(data))
            result=compare(a,b)
            self.assertTrue(result["all_images_exact"])
            self.assertFalse(result["candidate"]["timing"]["ms"]["hundred_sample_requirement_met"])
            data["environment"]["C3X_RENDERER_PREVIEW_SEASON"]="1"
            (b/"inputs.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"environment"):compare(a,b)
            data["environment"]["C3X_RENDERER_PREVIEW_SEASON"]="0";data["args"]["waves"]="1"
            (b/"inputs.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"quality"):compare(a,b)

    def test_modified_pixels_binary_camera_and_incomplete_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            image=root/"zoom.bmp.resident0.bmp";original=image.read_bytes()
            image.write_bytes(original[:-1]+b"X")
            with self.assertRaisesRegex(ValueError,"Image changed"):inspect(root)
            image.write_bytes(original)
            log=root/"benchmark.log";original=log.read_text()
            log.write_text(original.replace("x=35", "x=37",1))
            with self.assertRaisesRegex(ValueError,"camera"):inspect(root)
            log.write_text(original.replace("RESIDENT_END status=pass", "RESIDENT_END status=FAIL"))
            with self.assertRaisesRegex(ValueError,"sweep"):inspect(root)
            log.write_text(original)
            (root/"C3XRenderer.dll").write_bytes(b"other")
            with self.assertRaisesRegex(ValueError,"Binary changed"):inspect(root)

    def test_distribution_does_not_promote_small_samples_or_invalid_times(self):
        self.assertEqual(distribution(list(range(1,101)))["p95_ms"],95)
        for values in ([], [float("nan")], [-1]):
            with self.assertRaises(ValueError):distribution(values)

    def test_dense_scene_settings_cannot_disappear_in_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            a=self.fixture(Path(temporary)/"a");b=self.fixture(Path(temporary)/"b")
            data=json.loads((b/"inputs.json").read_text())
            data["args"].update(idle_steps=100,idle_units=0,dense_scene=False)
            data["environment"].update(C3X_RENDERER_PREVIEW_IDLE_UNITS="0",C3X_RENDERER_PREVIEW_DENSE_SCENE="")
            (b/"inputs.json").write_text(json.dumps(data))
            self.assertTrue(compare(a,b)["all_images_exact"])
            data["args"]["dense_scene"]=True
            (b/"inputs.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"quality"):compare(a,b)

    def test_camera_call_times_require_verified_coalescing_and_are_not_presentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            receipt=json.loads((root/"inputs.json").read_text());receipt["args"]["camera_view"]=True
            (root/"inputs.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,"camera completion"):inspect(root)
            log=root/"benchmark.log";original=log.read_text()
            calls="\n".join(f"CAMERA ticket={i+1} accepted_ms=1 final_ms=9 poll_max_ms=0.5 repeat_max_ms=0.2 identical_coalesced=1 stale_rejected=1 result=1" for i in range(14))
            log.write_text(calls+"\n"+original)
            report=inspect(root)[1]
            self.assertEqual(report["standalone_queue"]["accepted_ms"]["p95_ms"],1)
            self.assertIsNone(report["native_presented_frames"])
            log.write_text(calls.replace("identical_coalesced=1","identical_coalesced=0",1)+"\n"+original)
            with self.assertRaisesRegex(ValueError,"camera completion"):inspect(root)

    def test_idle_requires_advancing_clocks_retained_geometry_and_changed_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            receipt=json.loads((root/"inputs.json").read_text())
            receipt["args"].update(scenario="idle",idle_steps=14)
            (root/"inputs.json").write_text(json.dumps(receipt))
            for i in range(14):
                (root/f"zoom.bmp.idle{i}.bmp").write_bytes((root/f"zoom.bmp.resident{i}.bmp").read_bytes())
            completion=json.loads((root/"evidence.json").read_text())
            completion["images"]={p.name:digest(p) for p in root.glob("*.bmp")}
            (root/"evidence.json").write_text(json.dumps(completion))
            lines=["IDLE_BEGIN steps=14 warmup=10 pose_hz=15 paced=0 x=75 y=39 tile_width=128 units=0"]
            lines += [f"IDLE_FRAME step={i} ticks={1000000+(i+11)*1000000//15} result=1 visible=3 built=0 reused=10 upload_bytes=0 changed=1 ms=10 recoveries=0" for i in range(14)]
            lines += ["IDLE_END status=pass changed_frames=13", "BIQ viewport: 0 fallback"]
            log=root/"benchmark.log";original="\n".join(lines);log.write_text(original)
            report=inspect(root)[1]
            self.assertEqual(report["changed_frames"],13)
            self.assertEqual(report["timing"]["ms"]["samples"],14)
            self.assertIsNone(report["native_presented_frames"])
            phase="pose_prepare_ms=1 backdrop_submit_ms=2 animated_submit_ms=3 readback_submit_ms=4 readback_wait_ms=5 cpu_copy_ms=6"
            animation="ms=21 wave_upload_bytes=0 wave_cells_built=0 wave_cells_reused=0 backdrop_hits=1 backdrop_misses=0"
            trace=[]
            for i in range(20):
                ticks=1000000+(i+1)*1000000//15
                trace.extend((f"sequence={i+2} stage=animation-phases {phase}",
                              f"sequence={i+2} stage=animation-frame clock={ticks//(1000000//15)} {animation}"))
            (root/"renderer.log").write_text("\n".join(trace))
            partial=inspect(root)[1]
            self.assertEqual(partial["idle_trace_coverage"]["animation_frames"],10)
            self.assertFalse(partial["idle_trace_coverage"]["animation_trace_complete"])
            self.assertEqual(partial["animation_phases"]["animated_submit_ms"]["samples"],10)
            self.assertEqual(partial["animation"]["totals"]["backdrop_hits"],10)
            self.assertEqual(partial["timing"]["ms"]["samples"],14)
            for invalid in (original.replace("ticks=1733333","ticks=1000000"),
                            original.replace("built=0","built=1",1),
                            original.replace("upload_bytes=0","upload_bytes=256",1)):
                log.write_text(invalid)
                with self.assertRaisesRegex(ValueError,"Idle clocks"):inspect(root)
            log.write_text(original.replace("changed_frames=13","changed_frames=0"))
            with self.assertRaisesRegex(ValueError,"pose changes"):inspect(root)

    def test_animation_requires_exact_scroll_and_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            receipt=json.loads((root/"inputs.json").read_text());receipt["args"]["scenario"]="animation"
            (root/"inputs.json").write_text(json.dumps(receipt))
            for i in range(6):
                (root/f"zoom.bmp.animation-{i}.bmp").write_bytes((root/f"zoom.bmp.resident{i}.bmp").read_bytes())
            (root/"zoom.bmp").write_bytes((root/"zoom.bmp.resident0.bmp").read_bytes())
            completion=json.loads((root/"evidence.json").read_text())
            completion["images"]={p.name:digest(p) for p in root.glob("*.bmp")}
            (root/"evidence.json").write_text(json.dumps(completion))
            lines=["ANIMATION zoom-return parity: pass"]
            lines += [f"ANIMATION temporal frame={i} visible=3 terrain_built=0 terrain_upload=0 ms=10" for i in range(6)]
            lines += ["ANIMATION temporal: pass changed_frames=5", "ANIMATION scroll parity: pass changed=0 error=0", "ANIMATION removal parity: pass changed=0 error=0", "BIQ viewport: 0 fallback"]
            log=root/"benchmark.log";log.write_text("\n".join(lines))
            report=inspect(root)[1]
            self.assertEqual(report["timing"]["ms"]["samples"],6)
            self.assertFalse(report["timing"]["ms"]["hundred_sample_requirement_met"])
            log.write_text(log.read_text().replace("error=0","error=1",1))
            with self.assertRaisesRegex(ValueError,"exact animation"):inspect(root)


if __name__ == "__main__":
    unittest.main()
