# Development plan

| Stage | Scope | Status |
| --- | --- | --- |
| 0 | Repository skeleton, registry draft, README outline | done |
| 1 | Host CLI: schema validation, registry loader, docker runner, `doctor` | done (2026-09-16; docker execution itself unverified: no docker on dev host) |
| 2 | Port fusion (GT-free) and metrics from SL_MST; reproduce one paper-table row | done (2026-09-16; two rows reproduced to 0.001 mm via tools/repro) |
| 3 | Backends, in order: pear, wilor, teaser, sam3dbody, mediapipe | adapters, to_common, Dockerfiles, weight scripts written (2026-09-16); docker build/run still unverified (no docker host yet); mediapipe verified end-to-end on CPU |
| 4 | Three presets end-to-end, overlay visualization, README sections 1/5/7 | todo |
| 5 | Optional evaluation path (`tools/nia`, `pafpose evaluate`) | todo |
| 6 | Clean-machine validation, manual finalization, registration source export | todo |

## Decisions

- New standalone repository; `SL_MST/papers/iccas2026` stays as the paper workspace.
- First-class backends: sam3dbody, pear, wilor, teaser, mediapipe.
- Input contract: single person, one `.mp4` or a folder of `.mp4`.
- No sample video is shipped.
- External repos are cloned at a pinned commit inside each Dockerfile, never vendored.
- Name: PAF-Pose, package/CLI `pafpose`.

## Open items

- Docker builds for pear (pytorch3d source build), sam3dbody, wilor, teaser are untested.
- Paper fusion-table face-frame issue (PEAR/TEASER face exports not in the body frame); reproducible with tools/repro --face-frame corrected.
- Copyright holder for registration (personal vs. institution).
