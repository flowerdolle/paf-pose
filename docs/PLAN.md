# Development plan

| Stage | Scope | Status |
| --- | --- | --- |
| 0 | Repository skeleton, registry draft, README outline | done |
| 1 | Host CLI: schema validation, registry loader, docker runner, `doctor` | done (2026-09-16; docker execution itself unverified: no docker on dev host) |
| 2 | Port fusion (GT-free) and metrics from SL_MST; reproduce one paper-table row | done (2026-09-16; two rows reproduced to 0.001 mm via tools/repro) |
| 3 | Backends, in order: pear, wilor, teaser, sam3dbody, mediapipe | done (2026-09-22): all five images build on the user's docker host; fixes needed were get-pip URL, torch constraints + pytorch3d tag (pear), dill (wilor), ffmpeg decoding (all) |
| 4 | Build the five images on a docker host, run the three presets end-to-end on real video, fix what breaks | done (2026-09-22): speed / balanced / accuracy presets run on a 127-frame KETI clip (pear 20 s, wilor 11 s, sam3dbody 196 s); previews rendered. teaser not yet exercised on the docker host |
| 5 | Optional: overlay visualization and a ground-truth evaluation command | 3D skeleton preview done (`pafpose visualize`); GT evaluation command deferred |
| 6 | Clean-machine validation from git clone, README finalization, registration source export | clone-to-run validated on the user's PC (2026-09-22); README finalization and source export remaining |

## Decisions

- New standalone repository; `SL_MST/papers/iccas2026` stays as the paper workspace.
- First-class backends: sam3dbody, pear, wilor, teaser, mediapipe.
- Input contract: single person, one `.mp4` or a folder of `.mp4`.
- No sample video is shipped.
- External repos are cloned at a pinned commit inside each Dockerfile, never vendored.
- Name: PAF-Pose, package/CLI `pafpose`.

## Open items

- WiLoR's own detector misses hands in about half the frames of the KETI clip (67/127 complete); the paper used SAM 3D Body hand boxes instead. A body-guided crop for WiLoR would raise coverage.
- The all-in-one image was removed on 2026-09-22 (never built; per-backend images are the supported path).
- Paper fusion-table face-frame issue (PEAR/TEASER face exports not in the body frame); reproducible with tools/repro --face-frame corrected.
- Copyright holder for registration (personal vs. institution).
