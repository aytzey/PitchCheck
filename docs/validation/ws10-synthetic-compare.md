# WS10 predictive association

Decision: keep

predictive association only; production data are descriptive monitoring

```json
{
  "blind_sha256": null,
  "bootstrap": 2000,
  "calibration_basis": [
    "synthetic_not_validation"
  ],
  "command": "compare",
  "dataset_id": "ws10-synthetic-ci-v1",
  "decision": "keep",
  "exclusions": {},
  "implementation_sha256": {
    "scripts/validate_incremental.py": "83b139d9cea04a6a78902524247e5128029b1a468df7f5f35ee7fb52796d0b2b",
    "tribe_service/llm_layer.py": "b863ed3a60f27fc9fedf3711f7f0337e500d25cad662836f5cb050d2e8dc3b9f",
    "tribe_service/persuasion_features.py": "1164f71466d178ef984d3472c865d70cfff3cee30d2de954b8a0abb630f35932"
  },
  "input_sha256": "d94ced5b6491b771b5e7a0dfb5e72bb301f9dc3cb9a1262f7bb2f77b7722da9e",
  "outcomes_sha256": null,
  "permutations": 1000,
  "protocol_sha256": "0a18ec89b7a2d32abcec61d961132f3e27e0f95a7f3f74df53157d00a9675dbd",
  "recommended_calibration_basis": null,
  "refine_monitoring": {
    "n_selected": 0,
    "winner_change_rate": null
  },
  "refine_sha256": null,
  "scope": "predictive association only; production data are descriptive monitoring",
  "seed": 10,
  "strata": [
    {
      "active_C_w": 0.6586206896551724,
      "band_crossing_rate": 0.035,
      "decision": "keep",
      "mean_absolute_delta": 1.3437167830735228,
      "n": 200,
      "parity": "pass",
      "refine_winner_change_rate": null,
      "spearman_rho": 0.9451136278406961,
      "stratum": {
        "channel": "email",
        "language": "en",
        "llm_model": "synthetic-LLM",
        "methodology_version": "synthetic-v1",
        "outcome_family": "reply",
        "text_feature_model": "synthetic-encoder"
      },
      "without_C_w": 0.6827586206896552
    }
  ],
  "synthetic": true,
  "underpowered_message": null
}

```
