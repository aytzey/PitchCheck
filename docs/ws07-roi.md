# WS07: signed cortical literature priors

Implemented plan steps 1–10, K2/K3/K5/K8 and C.WS07. Step 11 has a working benchmark/artifact path; its GPU run is intentionally deferred. WS09 09-A/B/C1, WS12 scoring/contrasts and WS08/WS11 research jobs are separate work. No Rust scoring changes or release packaging are included.

The single builder consumes `scripts/roi_peaks.csv`, transcribed from the authoritative plan's PDF-image curation and prototype. It fetches the four CBIG annotations and MIT license at commit `634f676630929a71297852d01dd92a287103e861`, uses nibabel 5.4.2 / nilearn 0.14.1, and records their hashes, versions, coordinate transforms and every peak decision. Meshes are not redistributed. Runtime uses only numpy and stdlib, with no downloads or provenance-JSON reads.

Create a disposable environment, then build twice (use paths in your research scratch directory):

```sh
python3 -m venv "$SCRATCH/venv-roi"
"$SCRATCH/venv-roi/bin/python" -m pip install nibabel==5.4.2 nilearn==0.14.1 numpy
"$SCRATCH/venv-roi/bin/python" scripts/build_roi_assets.py --cache-dir "$SCRATCH/roi-build-cache"
"$SCRATCH/venv-roi/bin/python" scripts/build_roi_assets.py --cache-dir "$SCRATCH/roi-build-cache" \
  --output "$SCRATCH/roi-build-repeat" --ui-output "$SCRATCH/roi-centroids-repeat.json"
```

The two executed builds produced identical NPZ, provenance, license and UI centroid bytes. Committed NPZ SHA-256: `c5ca6bb873d2e0d140a83dd94ae30c2c1dfca2959e6b50a7ae6ed019424bfe0d` (23,239 bytes). The runtime SHA field is its first 16 hex characters. The nine non-overlapping parcel unions match the prototype: frontopolar 77, medial orbitofrontal 142, dorsomedial prefrontal 191, insula 93, right inferior frontal/anterior insula 100, posterior cingulate/precuneus 169, right temporoparietal junction 39, lateral temporal 264, left inferior frontal 76 vertices. All 13 requested location keys are present; they intentionally overlap.

The authoritative build records these geometry caveats:

- Right inferior frontal/anterior insula assignments differ between affine and no-affine paths: `RH_DefaultB_PFCv_1` and `RH_SalVentAttnB_Ins_1`; both are retained.
- Posterior cingulate/precuneus assignments differ on both hemispheres; all four parcels are retained.
- Left posterior superior temporal sulcus has a network mismatch (`LH_SomMotB_Aud_2`); the affine path selects the nearest allowed `LH_TempPar_3` vertex at 6.57771 mm. The no-affine alternative is 8.57778 mm away and is dropped.
- Cooper's unpublished `(0,-42,21)` centroid maps to the medial wall in both paths and is excluded. Subcortical/excluded and null/benchmark records never contribute selected parcels.
- The strict location rule drops the right dorsomedial prefrontal Chua 2011 peak at **6.09073 mm**, leaving the accepted left 22 vertices. The plan's approximate 39-vertex preview is not hardcoded. Left inferior frontal Chua 2011 has 47 vertices and ventromedial prefrontal Chua 2009 has 54; the builder JSON is authoritative.

`load_roi_assets()` is the single loader. It validates index types, bounds, hemisphere layout, medial wall, parcel count and non-overlap; missing/corrupt assets log a warning and disable overlays. Signed time means are referenced to labelled cortex, excluding the medial wall. ROI z is invariant to positive scaling and constant offsets. The reducer feeds Python and native analysis paths identically.

The six signed priors use the fixed K4 denominator. Compatible Llama gets weight 0.15; incompatible encoders get 0.075 by default. Robustness publishes the term, provenance, anchor-point shadow, total-prior counterfactual and K5 budget diagnostics. Legacy refine automatically uses the adjusted anchor and publishes `roi_prior_score`. Jev's whitelist and selection policy remain unchanged; neither Jev nor the legacy critic receives raw ROI fields. Only the evaluator's explicitly allowed WS07 block and deterministic synthesis contain ROI values. Desktop refine advice carries the ROI lever/citation without numeric observations.

Desktop normalizers preserve signed `rois`/`locations` and optional WS08/WS11 badges. The ROI row appears when summary/prior/points fields exist. Brain points use the builder's actual parcel-union centroids projected onto the existing top-down schematic; with absent fields, neutral whole-cortex codes remain. Both robustness and variant ranking share “About these signals”. Mobile UI is unchanged.

## Unit defaults and the deferred GPU run

Current fallback centers are 0 and `ROI_Z_SCALE=1.0`, tagged **`provisional_unit_defaults:ws07;gpu_not_run`** in every ROI prior. These are deliberately provisional, not calibrated values. The requested no-GPU exception leaves the §9.3 `unit_calibration:<run_id>` acceptance item pending; ROI remains default-on with the rollback env and ±6-point final budget.

Run the following inside the GPU image, separately for Llama and Hermes, with their cached weights and the usual memory/thread limits. No paid LLM calls are made by the benchmark:

```sh
TRIBE_ALLOW_MOCK=0 TRIBE_TEXT_MODEL=meta-llama/Llama-3.2-3B \
  python scripts/benchmark-runtime.py --output /audit/roi-llama --run-id ws07-llama-YYYYMMDD
TRIBE_ALLOW_MOCK=0 TRIBE_TEXT_MODEL=NousResearch/Hermes-3-Llama-3.2-3B \
  python scripts/benchmark-runtime.py --output /audit/roi-hermes --run-id ws07-hermes-YYYYMMDD
```

The benchmark logs per-case signed ROI/location statistics and prior raw values in `metrics.json`. It deduplicates repeated warm-cache cases, computes each center as median z and the scale as SD of centered raw, rejects degenerate distributions, and writes `roi_unit_calibration_<encoder.replace('/', '__')>.json`. Review and copy those numeric artifacts into `tribe_service/roi_assets/` before release; retain their `unit_calibration:<run_id>` tag. No benchmark artifact was generated from mock predictions. `--selftest` exercises only the unit calculation on CPU and loads no models.

Runtime selects the matching encoder's artifact and checks its encoder, atlas hash, finite centers, positive scale and provenance. Once WS08 supplies `engine.roi_reference(encoder)`, matching `unit_calibration` in that reference takes precedence. The same optional reader's `checks[k].weight_multiplier` and WS11's same-module `fidelity_factor(k)[0]` compose through `roi_multipliers`, for parcel and location keys alike. Missing research hooks return the unchanged multiplier 1.

`scripts/smoke-runtime.py` now asserts the image contains both ROI sections. Its authenticated, real-weight run was not executed here. No production host, paid API, GPU model or graphify job was contacted.
