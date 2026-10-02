"""Refuse plausible but unlinked, partial, mixed or unequal route evidence."""
import copy
import unittest

from Renderer.tools.analyze_matched_native_route import analyze_capture,compare,parse_log


def fixture():
    plan={"client_width":2240,"client_height":1260,"map_width":130,"map_height":130,
          "steps":[{"step":1,"name":"first-visit-jump","x":100,"y":200,"width":160,"advance":"adopted"}]}
    result={"passed":True,"failure":None,"original_save_unchanged":True,"disposable_save_unchanged":True,"ini_restored":True,
            "ready_map_qpc":900,"ready_map_seconds":40,"idle_begin_qpc":950,"idle_end_qpc":1100,
            "events":[{"kind":"posted","step":1,"qpc":1050}]}
    inputs={"game_sha256":"1"*64,"save_sha256":"2"*64,"plan_sha256":"3"*64,
            "common_injected_diagnostic_delta":True,"baseline_full_system_pristine":False}
    cadence={"qpc_frequency":1000,"samples":[{"qpc":qpc,"frames":frames} for qpc,frames in ((950,1),(1050,1),(1250,2),(1350,3),(1650,4))]}
    native=parse_log("\n".join([
        "[C3X renderer] stage=scripted-route-accepted step=1 qpc=1100 frequency=1000 requested=100,200 target_width=160 capture_width=128 map_width=130 map_height=130",
        "[C3X renderer] stage=scripted-route-resolved step=1 qpc=1110 frequency=1000 requested=100,200 native=100,200 target_width=160",
        "[C3X renderer] stage=scripted-route-adopted step=1 qpc=1150 frequency=1000 requested=100,200 displayed=100,200 valid=1",
        "[C3X renderer] stage=route-native-source source_serial=7 qpc=1151 camera=100,200 frequency=1000 valid=1 genuine=1",
    ]))
    core=parse_log("\n".join([
        "[C3X renderer] stage=route-publication serial=7 qpc=1120 camera=100,200 phase_x=-100 phase_y=-200 width=2240 height=1260 tile_width=128 tile_height=64 anchor_x=-100 anchor_y=-200 anchor_basis=canonical identity=99",
        "[C3X renderer] stage=route-workload source_serial=7 source_generation=3 qpc=1180 complete=1 overflow=0 count=2 camera_x=-100 camera_y=-200 zoom=1.250000 facts_digest="+"4"*64+" main_units=2 reflected_units=1 shadow_units=1",
        "[C3X renderer] stage=route-workload-members source_serial=7 source_generation=3 qpc=1181 segment=0 total_segments=1 unit_ids=42,43 pass_masks=7,1",
        "[C3X renderer] stage=route-presented qpc=1200 source_serial=7 source_generation=3 zoom_q16=81920 result=1 mixed=0 frequency=1000 present_index=9",
        "[C3X renderer] stage=route-presented qpc=1300 source_serial=7 source_generation=3 zoom_q16=81920 result=1 mixed=0 frequency=1000 present_index=10",
        "[C3X renderer] stage=route-presented qpc=1600 source_serial=7 source_generation=3 zoom_q16=81920 result=1 mixed=0 frequency=1000 present_index=11",
        "[C3X renderer] stage=frame qpc=1205 render_ms=12 geometry_ms=4 draw_submit_ms=7",
    ]))
    return plan,result,inputs,cadence,native,core


def budget_fixture():
    args=fixture()
    for row in args[-1]:
        if row["stage"]=="route-presented":
            row["present_qpc"]=row["qpc"]
            row["qpc"]=str(int(row["qpc"])+{9:8,10:17,11:33}[int(row["present_index"])])
    args[-1].extend(parse_log("\n".join([
        "[C3X renderer] stage=frame-preparation-ready qpc=1191 source_serial=7 source_generation=3 begin=1170 candidate_end=1175 selection_end=1180 assets_end=1182 render_begin=1184 end=1190 frequency=1000 candidates_ms=999 selection_ms=999 assets_ms=999 target_ms=999 render_ms=999 total_ms=4995",
        "[C3X renderer] stage=route-frame-budget qpc=1210 source_serial=7 source_generation=3 present_index=9 begin=1160 sampled=1195 end=1200 frequency=1000 compose_ms=999 present_ms=999 total_ms=1998 operations=12 assemblies=3 copies=4 copied_pixels=1024 assembly_pixels=4096",
        "[C3X renderer] stage=route-frame-budget qpc=1319 source_serial=7 source_generation=3 present_index=10 begin=1260 sampled=1290 end=1300 frequency=1000 operations=13 assemblies=4 copies=5 copied_pixels=2048 assembly_pixels=8192",
        "[C3X renderer] stage=route-frame-budget qpc=1635 source_serial=7 source_generation=3 present_index=11 begin=1480 sampled=1590 end=1600 frequency=1000 operations=14 assemblies=5 copies=6 copied_pixels=4096 assembly_pixels=16384",
    ])))
    return args


def binding_fixture():
    args=fixture()
    source=args[-2][-1];source.pop("source_serial");source["local_image_ticket"]="20"
    args[-2].extend(parse_log("\n".join([
        "[C3X renderer] stage=route-native-request local_camera_ticket=20 camera=100,200 qpc=1115 frequency=1000 valid=1 genuine=1",
        "[C3X renderer] stage=route-ticket-binding local_camera_ticket=20 local_image_ticket=20 remote_camera_ticket=51 map_ticket=107 remote_session=77 qpc=1190 frequency=1000 valid=1 genuine=1",
    ])))
    args[-1][0].update(map_ticket="107",remote_camera_ticket="51",remote_session="77")
    return args


class MatchedNativeRouteAnalysis(unittest.TestCase):
    def test_exact_serial_join_keeps_three_latencies_separate(self):
        report=analyze_capture(*fixture());row=report["route"][0]
        self.assertTrue(row["route_eligible"])
        self.assertEqual(row["posted_to_native_acceptance_ms"],50)
        self.assertEqual(row["native_camera_adoption"]["native_acceptance_to_adoption_ms"],50)
        self.assertEqual(row["correct_destination_present"]["native_acceptance_to_present_ms"],100)
        self.assertEqual(row["correct_destination_present"]["posted_to_present_ms"],150)
        self.assertTrue(compare(report,copy.deepcopy(report))["all_step_comparisons_eligible"])

    def test_missing_source_serial_and_mixed_source_are_not_proximity_joined(self):
        for change in ({"source_serial":"0"},{"source_serial":"88"},{"mixed":"1"}):
            with self.subTest(change=change):
                args=fixture()
                for row in args[-1]:
                    if row["stage"]=="route-presented":row.update(change)
                report=analyze_capture(*args)
                self.assertEqual(report["route"][0]["correct_destination_present"]["status"],"unavailable")
                self.assertFalse(report["all_route_endpoints_eligible"])

    def test_old_camera_or_wrong_projection_is_not_correct_destination(self):
        for key,value,stage in (("camera","99,200","route-native-source"),("zoom_q16","65536","route-presented")):
            with self.subTest(key=key):
                args=fixture()
                for row in args[-2]+args[-1]:
                    if row["stage"]==stage:row[key]=value
                self.assertFalse(analyze_capture(*args)["all_route_endpoints_eligible"])

    def test_first_target_counter_with_almost_exact_zoom_is_skipped_for_later_exact_frame(self):
        args=fixture();args[-1][1]["zoom"]="1.249998"
        later_work=dict(args[-1][1]);later_work.update(source_generation="4",qpc="1250",zoom="1.250000")
        later_members=dict(args[-1][2]);later_members.update(source_generation="4",qpc="1251")
        args[-1].extend((later_work,later_members));args[-1][4]["source_generation"]="4"
        report=analyze_capture(*args);row=report["route"][0]
        self.assertTrue(row["route_eligible"])
        self.assertEqual(row["correct_destination_present"]["source_generation"],4)
        self.assertEqual(row["correct_destination_present"]["qpc"],1300)
        self.assertEqual(row["correct_destination_present"]["native_acceptance_to_present_ms"],200)
        self.assertEqual(row["actual_workload"]["unit_ids"],[42,43])
        args[-1][4]["source_generation"]="3"
        self.assertFalse(analyze_capture(*args)["all_route_endpoints_eligible"])

    def test_full_workload_guards_are_applied_before_choosing_earliest_endpoint(self):
        for mutation in ("missing-members","overflow","wrong-anchor","future-workload","wrong-mask"):
            with self.subTest(mutation=mutation):
                args=fixture();later_work=dict(args[-1][1]);later_work.update(source_generation="4",qpc="1250")
                later_members=dict(args[-1][2]);later_members.update(source_generation="4",qpc="1251")
                if mutation=="missing-members":args[-1][2]["source_generation"]="99"
                elif mutation=="overflow":args[-1][1]["overflow"]="1"
                elif mutation=="wrong-anchor":args[-1][1]["camera_x"]="-101"
                elif mutation=="future-workload":args[-1][1]["qpc"]="1700"
                else:args[-1][2]["pass_masks"]="1,1"
                args[-1].extend((later_work,later_members));args[-1][4]["source_generation"]="4"
                row=analyze_capture(*args)["route"][0]
                self.assertTrue(row["route_eligible"])
                self.assertEqual(row["correct_destination_present"]["qpc"],1300)
                # Contradictory camera proof still refuses every generation.
                args[-2].append({"stage":"route-native-source","source_serial":"7","camera":"101,200","qpc":"1260","frequency":"1000","valid":"1","genuine":"1"})
                self.assertFalse(analyze_capture(*args)["all_route_endpoints_eligible"])

    def test_partial_and_overflowing_unit_lists_refuse_workload_matching(self):
        for field,value in (("complete","0"),("overflow","1")):
            with self.subTest(field=field):
                args=fixture();args[-1][1][field]=value
                report=analyze_capture(*args)
                self.assertIsNone(report["route"][0]["actual_workload"])
                self.assertFalse(compare(report,copy.deepcopy(report))["all_step_comparisons_eligible"])

    def test_equal_counts_different_actual_ids_are_ineligible(self):
        baseline=analyze_capture(*fixture());args=fixture();args[-1][2]["unit_ids"]="42,44"
        candidate=analyze_capture(*args);comparison=compare(baseline,candidate)
        self.assertTrue(comparison["paired_steps"][0]["same_actual_route"])
        self.assertFalse(comparison["paired_steps"][0]["same_actual_workload"])
        self.assertFalse(comparison["all_step_comparisons_eligible"])

    def test_member_segments_and_source_generation_are_complete_exact_joins(self):
        for mutation in ("missing-segment","wrong-generation","wrong-total","wrong-count","wrong-mask-count"):
            with self.subTest(mutation=mutation):
                args=fixture()
                if mutation=="missing-segment":args[-1].pop(2)
                elif mutation=="wrong-generation":args[-1][2]["source_generation"]="4"
                elif mutation=="wrong-total":args[-1][2]["total_segments"]="2"
                elif mutation=="wrong-count":args[-1][1]["count"]="3"
                else:args[-1][2]["pass_masks"]="1,1"
                report=analyze_capture(*args)
                self.assertIsNone(report["route"][0]["actual_workload"])
                self.assertFalse(report["all_route_endpoints_eligible"])

    def test_wrapped_occurrences_may_share_one_native_unit_id(self):
        args=fixture();args[-1][2]["unit_ids"]="42,42"
        report=analyze_capture(*args)
        self.assertTrue(report["all_route_endpoints_eligible"])
        self.assertEqual(report["route"][0]["actual_workload"]["unit_ids"],[42,42])

    def test_explicit_request_ticket_can_certify_camera_before_native_poll(self):
        args=fixture();args[-2][-1]["qpc"]="1700"
        args[-2].append({"stage":"route-native-request","camera_ticket":"51","camera":"100,200","qpc":"1115","frequency":"1000","valid":"1","genuine":"1"})
        args[-1][0]["camera_ticket"]="51"
        report=analyze_capture(*args)
        self.assertTrue(report["all_route_endpoints_eligible"])
        self.assertEqual(report["route"][0]["camera_source_proof"]["stage"],"route-native-request")

    def test_actual_adoption_binding_joins_distinct_local_remote_map_and_source_ids(self):
        args=binding_fixture();report=analyze_capture(*args)
        self.assertTrue(report["all_route_endpoints_eligible"])
        proof=report["route"][0]["camera_source_proof"]
        self.assertEqual([int(proof[key]) for key in ("local_camera_ticket","remote_camera_ticket","map_ticket")],[20,51,107])
        self.assertEqual(report["route"][0]["correct_destination_present"]["source_serial"],7)
        self.assertEqual(int(proof["binding_qpc"]),1190)
        args[0].update(initial_camera=[100,200],initial_width=160);args[1].update(idle_begin_qpc=1200,idle_end_qpc=1601)
        self.assertTrue(analyze_capture(*args)["idle_workload_eligible"])

    def test_binding_must_precede_present_and_match_every_remote_field(self):
        args=binding_fixture();args[-2][-1]["qpc"]="1210"
        report=analyze_capture(*args)
        self.assertEqual(report["route"][0]["correct_destination_present"]["qpc"],1300)
        for key in ("map_ticket","remote_camera_ticket","remote_session","frequency","valid","genuine"):
            with self.subTest(key=key):
                args=binding_fixture();args[-2][-1][key]="999"
                self.assertFalse(analyze_capture(*args)["all_route_endpoints_eligible"])
        args=binding_fixture();args[-2].pop()
        # Even equality across the two old integer namespaces is no proof for
        # an explicitly labelled new publication without its adoption binding.
        args[-1][0]["camera_ticket"]="20"
        self.assertFalse(analyze_capture(*args)["all_route_endpoints_eligible"])
        for clock in ("-1","1700"):
            args=binding_fixture();args[-2][-1]["qpc"]=clock
            self.assertFalse(analyze_capture(*args)["all_route_endpoints_eligible"])

    def test_conflicting_binding_local_camera_and_legacy_serial_are_all_preserved(self):
        for mutation in ("binding","local-camera","legacy-source"):
            with self.subTest(mutation=mutation):
                args=binding_fixture()
                if mutation=="binding":
                    conflict=dict(args[-2][-1]);conflict["map_ticket"]="108";args[-2].append(conflict)
                elif mutation=="local-camera":args[-2][-2]["camera"]="101,200"
                else:args[-2].append({"stage":"route-native-source","source_serial":"7","camera":"101,200","qpc":"1190","frequency":"1000","valid":"1","genuine":"1"})
                report=analyze_capture(*args)
                self.assertFalse(report["all_route_endpoints_eligible"])
                if mutation=="binding":self.assertEqual(report["ticket_binding_refusals"],[["local_camera_ticket",20],["local_image_ticket",20]])

    def test_future_or_contradictory_camera_and_workload_evidence_is_refused(self):
        for changed in ("future-camera","future-workload","contradictory-camera"):
            with self.subTest(changed=changed):
                args=fixture()
                if changed=="future-camera":args[-2][-1]["qpc"]="1700"
                elif changed=="future-workload":args[-1][1]["qpc"]="1700"
                else:args[-2].append({"stage":"route-native-source","source_serial":"7","camera":"101,200","qpc":"1190","frequency":"1000","valid":"1","genuine":"1"})
                self.assertFalse(analyze_capture(*args)["all_route_endpoints_eligible"])

    def test_idle_has_independent_complete_workload_eligibility(self):
        args=fixture();args[0].update(initial_camera=[100,200],initial_width=160)
        args[1].update(idle_begin_qpc=1150,idle_end_qpc=1601)
        baseline=analyze_capture(*args)
        self.assertTrue(baseline["idle_workload_eligible"])
        self.assertTrue(compare(baseline,copy.deepcopy(baseline))["idle_performance_comparison_eligible"])
        args[-1][3]["mixed"]="1";candidate=analyze_capture(*args)
        self.assertFalse(candidate["idle_workload_eligible"])
        self.assertFalse(compare(baseline,candidate)["idle_performance_comparison_eligible"])

    def test_conflicting_publication_or_workload_on_one_serial_refuses_identity(self):
        for offset,key,value in ((0,"camera","99,200"),(1,"facts_digest","5"*64)):
            with self.subTest(key=key):
                args=fixture();conflict=dict(args[-1][offset]);conflict[key]=value;conflict["line"]=99
                args[-1].append(conflict);report=analyze_capture(*args)
                self.assertTrue(report["source_serial_refusals"]==[7] if offset==0 else report["source_generation_refusals"]==[[7,3]])
                self.assertFalse(report["all_route_endpoints_eligible"])

    def test_repeated_identical_contract_with_new_timestamp_is_not_ambiguous(self):
        args=fixture();repeat=dict(args[-1][0]);repeat.update(qpc="1140",line=99);args[-1].append(repeat)
        report=analyze_capture(*args)
        self.assertEqual(report["source_serial_refusals"],[])
        self.assertTrue(report["all_route_endpoints_eligible"])

    def test_failed_capture_or_modified_save_disqualifies_comparison(self):
        baseline=analyze_capture(*fixture())
        for field,value in (("passed",False),("original_save_unchanged",False)):
            with self.subTest(field=field):
                args=fixture();args[1][field]=value;candidate=analyze_capture(*args)
                self.assertFalse(compare(baseline,candidate)["all_step_comparisons_eligible"])

    def test_consecutive_present_indices_are_exact_but_missing_indices_are_aggregated(self):
        report=analyze_capture(*fixture())
        self.assertEqual(report["ready_successful_present_intervals_ms"]["n"],2)
        self.assertEqual(report["ready_successful_present_intervals_ms"]["max"],300)
        args=fixture();args[-1][5]["present_index"]="13";report=analyze_capture(*args)
        self.assertEqual(report["ready_successful_present_intervals_ms"]["n"],1)
        self.assertEqual(report["unqualified_present_spans"][0]["present_index_delta"],3)

    def test_eviction_candidate_requires_explicit_destination_evidence(self):
        args=fixture();args[0]["steps"][0]["name"]="eviction-candidate-return"
        report=analyze_capture(*args)
        self.assertEqual(report["route"][0]["evicted_jump_qualification"]["status"],"unavailable")

    def test_duplicate_input_step_and_backward_clock_are_refused(self):
        args=fixture();args[-2].append(dict(args[-2][0]));report=analyze_capture(*args)
        self.assertFalse(report["all_route_endpoints_eligible"])
        args=fixture();args[3]["samples"][1]["qpc"]=900
        with self.assertRaisesRegex(ValueError,"strictly ordered"):analyze_capture(*args)

    def test_explicit_present_return_precedes_diagnostic_write_and_drives_intervals(self):
        report=analyze_capture(*budget_fixture());endpoint=report["route"][0]["correct_destination_present"]
        self.assertEqual(endpoint["qpc"],1200)
        self.assertEqual(endpoint["diagnostic_log_qpc"],1208)
        self.assertEqual(endpoint["native_acceptance_to_present_ms"],100)
        self.assertEqual(endpoint["clock_endpoint"],"successful Present return")
        self.assertEqual(report["ready_successful_present_intervals_ms"]["max"],300)
        args=budget_fixture();args[0].update(initial_camera=[100,200],initial_width=160);args[1].update(idle_begin_qpc=1150,idle_end_qpc=1310)
        self.assertEqual(analyze_capture(*args)["idle_workload_coverage"]["successful_present_witnesses"],2)

    def test_nested_preparation_is_partitioned_once_using_disjoint_raw_spans(self):
        report=analyze_capture(*budget_fixture());frame=report["frame_budget"]["frames"][0]
        self.assertEqual(frame["compose_ms"],35)
        self.assertEqual(frame["same_generation_preparation_overlap_ms"],20)
        self.assertEqual(frame["residual_compose_ms"],15)
        self.assertEqual(frame["present_call_ms"],5)
        self.assertEqual(frame["total_frame_work_ms"],40)
        self.assertEqual(frame["preparation_subphases_ms"],dict(candidates=5,selection=5,assets=2,target=2,render=6))
        self.assertEqual(sum(frame["preparation_subphases_ms"].values()),frame["same_generation_preparation_overlap_ms"])
        self.assertEqual(frame["copies"],4)
        gap=report["frame_budget"]["consecutive_gaps"][0]
        self.assertEqual(gap["unclassified_cadence_queue_ownership_gap_ms"],60)
        self.assertEqual(gap["end_to_end_present_gap_ms"],100)
        self.assertEqual(report["frame_budget"]["frames"][1]["total_frame_work_ms"]+gap["unclassified_cadence_queue_ownership_gap_ms"],100)

    def test_intersection_clips_preparation_and_identical_duplicates_do_not_inflate_it(self):
        args=budget_fixture();prep=next(row for row in args[-1] if row["stage"]=="frame-preparation-ready")
        prep.update(begin="1150",candidate_end="1155",selection_end="1165",assets_end="1175",render_begin="1185")
        args[-1].append(dict(prep));frame=analyze_capture(*args)["frame_budget"]["frames"][0]
        self.assertEqual(frame["same_generation_preparation_overlap_ms"],30)
        self.assertEqual(frame["residual_compose_ms"],5)
        self.assertEqual(frame["preparation_span_records"],1)
        self.assertEqual(frame["preparation_subphases_ms"],dict(candidates=0,selection=5,assets=10,target=10,render=5))

    def test_budget_and_preparation_use_exact_index_serial_generation_not_clock_proximity(self):
        for mutation in ("budget-generation","preparation-generation","budget-end"):
            with self.subTest(mutation=mutation):
                args=budget_fixture()
                if mutation=="preparation-generation":
                    next(row for row in args[-1] if row["stage"]=="frame-preparation-ready")["source_generation"]="4"
                else:
                    budget=next(row for row in args[-1] if row["stage"]=="route-frame-budget")
                    budget["source_generation" if mutation=="budget-generation" else "end"]="4" if mutation=="budget-generation" else "1201"
                report=analyze_capture(*args);frames=report["frame_budget"]["frames"]
                if mutation=="preparation-generation":
                    self.assertEqual(frames[0]["same_generation_preparation_overlap_ms"],0)
                    self.assertIsNone(frames[0]["preparation_subphases_ms"])
                    self.assertIn("no matching",frames[0]["preparation_coverage"])
                else:self.assertNotIn(9,[row["present_index"] for row in frames])
                if mutation=="budget-end":self.assertTrue(report["frame_budget"]["refusals"])

    def test_missing_raw_subphase_boundaries_never_reconstructs_them_from_rounded_ms(self):
        args=budget_fixture();prep=next(row for row in args[-1] if row["stage"]=="frame-preparation-ready")
        del prep["candidate_end"]
        frame=analyze_capture(*args)["frame_budget"]["frames"][0]
        self.assertEqual(frame["same_generation_preparation_overlap_ms"],20)
        self.assertIsNone(frame["preparation_subphases_ms"])

    def test_negative_reversed_overlapping_or_future_raw_clock_intervals_refuse_analysis(self):
        for mutation in ("negative","reversed-budget","reversed-phase","overlap","next-before-present","present-after-log","present-clock-reversed"):
            with self.subTest(mutation=mutation):
                args=budget_fixture();prep=next(row for row in args[-1] if row["stage"]=="frame-preparation-ready")
                budgets=[row for row in args[-1] if row["stage"]=="route-frame-budget"]
                if mutation=="negative":budgets[0]["begin"]="-1"
                elif mutation=="reversed-budget":budgets[0]["sampled"]="1159"
                elif mutation=="reversed-phase":prep["selection_end"]="1174"
                elif mutation=="overlap":
                    conflict=dict(prep);conflict.update(begin="1171");args[-1].append(conflict)
                elif mutation=="next-before-present":budgets[1]["begin"]="1199"
                elif mutation=="present-after-log":args[-1][3]["present_qpc"]="1209"
                else:args[-1][4]["present_qpc"]="1199"
                with self.assertRaisesRegex(ValueError,"(clock|Overlapping)"):analyze_capture(*args)


if __name__=="__main__":unittest.main()
