// Private coverage-only entries use the same vertex/material/raster inputs.
// Ignore every surface family whose complete opacity is not established here.
#ifdef C3X_UNDERLAY_GROUND_COVERAGE
void PSSandboxOpaqueCoverage(P input) {
    clip(input.material.y-.5);
    clip(1.5-input.material.y-.00001);
    float alpha = saturate(input.coast_coverage+10);
    float2 edge_uv=input.world.xy*Detail.x*2.05+float2(.31,.17);
    float edge_height=GrassColor.Sample(Wrap,edge_uv).a;
    float edge_mean=GrassColor.SampleBias(Wrap,edge_uv,3).a;
    float edge_grain=saturate(.5+(edge_height-edge_mean)*3);
    alpha=lerp(coast_edge_coverage(alpha,edge_grain),alpha,input.coast_inland);
    // Retain uncertain shoreline fragments even if texture grain rounds their
    // alpha to one. Inland one is the conservative opaque receiver contract.
    clip(input.coast_inland-1);
    clip(alpha-1);
}
#else
void PSSandboxOpaqueCoverage(P input) {
    clip(input.material.y-.5);
#ifdef BEAUTY_TERRAIN_TRANSITIONS
    float coast_alpha = saturate(input.coast_coverage - 42);
#else
    float coast_alpha = 1;
#endif
    clip(coast_alpha-1);
}
#endif
