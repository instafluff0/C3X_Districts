# Third-Party Civ VI Conversion References

These repositories are checked out as local references for the Civ VI art conversion spike. Treat them as upstream code: inspect and wrap them from C3X-owned tools before modifying them.

## Pinned Repositories

- `CivNexus6`
  - URL: `https://github.com/deliverator23/CivNexus6.git`
  - Commit: `fc3ac86c2ef3b02459e101b5930df95e0b68f70d`
  - Purpose: Source for Civ VI `.fgx` tooling and CivNexus6 behavior.
- `Nexus-Buddy-2-Blender-Scripts`
  - URL: `https://github.com/Sukritact/Nexus-Buddy-2-Blender-Scripts.git`
  - Commit: `0a39dfbecdf23971f3df146765858a1f0e15505f`
  - Purpose: Blender 2.8+ / 3.x / 4.x import-export scripts for CivNexus6 `.cn6` workflows.
- `Civilization-Blender-Scripts`
  - URL: `https://github.com/deliverator23/Civilization-Blender-Scripts.git`
  - Commit: `1ce5d09384b34cc7be099228b017937f72646b88`
  - Purpose: Deliverator's original Blender import-export scripts for Civ V/Beyond Earth/Civ VI graphics workflows.

## Notes

- Blender is not assumed to be on `PATH`.
- The local Civ VI `BLPs` tree is cooked platform content. It is useful for discovery and references, but may not provide directly extractable model source.
- Prefer loose SDK/Pantry `.fgx` files or permissioned source-mod repositories for real model conversion.
