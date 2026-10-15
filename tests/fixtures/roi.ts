import type { RoiPrior, RoiSummary } from "@/shared/types";

export const roiSummary: RoiSummary = {
  atlas: "schaefer2018_400_17n_fsaverage5", basis: "signed_parcel_union_minus_cortex_v1",
  roi_set_version: "roi_set_v1", location_set_version: "roi_locations_v1", asset_sha256: "fixture",
  subject_basis: "average_subject", caveat: "Predicted average-subject response; not this recipient.",
  cortex_mean_signed: -0.1, cortex_sd_signed: 0.2,
  rois: {
    vmpfc_frontopolar: { mean_signed: -0.2, minus_cortex: -0.1, z: -0.5, n_vertices: 30,
      external_fidelity: { status: "weak", factor: 0.25 }, in_silico_check: "null" },
  },
  locations: { mpfc_falk2010: { mean_signed: 0.1, minus_cortex: 0.2, z: 1, n_vertices: 20 } },
};

export const roiPrior: RoiPrior = {
  score: 42, raw: -0.2, weight: 0.15, applied: true, version: "roi_prior_v1",
  provenance: "literature_prior_unvalidated", terms: { vmpfc_frontopolar: -0.15 }, multipliers: { vmpfc_frontopolar: 1 },
  unit_calibration_provenance: "provisional_unit_defaults:ws07;gpu_not_run",
};
