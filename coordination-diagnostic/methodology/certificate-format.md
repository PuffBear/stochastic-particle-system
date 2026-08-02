# Provenance Certificate Format

**Version:** 0.1

The provenance certificate is the primary deliverable of a coordination diagnostic engagement. It is a structured document that maps each reported conclusion to the set of numerical conditions it survived.

---

## JSON schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CoordinationDiagnosticCertificate",
  "version": "0.1",
  "type": "object",
  "required": ["meta", "conditions", "assumptions", "results", "diagnosis", "certificate_text"],
  "properties": {

    "meta": {
      "type": "object",
      "properties": {
        "system_name":        { "type": "string" },
        "protocol_name":      { "type": "string" },
        "diagnostic_version": { "type": "string" },
        "run_date":           { "type": "string", "format": "date" },
        "seed_set":           { "type": "array", "items": { "type": "integer" } },
        "metric_name":        { "type": "string" },
        "metric_higher_is_better": { "type": "boolean" },
        "engagement_type":    { "enum": ["diagnostic", "confirmatory"] }
      }
    },

    "conditions": {
      "type": "object",
      "properties": {
        "A": { "type": "string", "description": "Description of actual-messages condition" },
        "B": { "type": "string", "description": "Description of permuted-messages condition" },
        "C": { "type": "string", "description": "Description of no-messages condition" }
      }
    },

    "assumptions": {
      "type": "object",
      "properties": {
        "matched_initialization": { "enum": ["verified", "declared", "unverified"] },
        "matched_stochasticity":  { "enum": ["verified", "declared", "unverified"] },
        "no_leakage":             { "enum": ["verified", "declared", "unverified"] },
        "stationary_protocol":    { "enum": ["verified", "declared", "unverified"] },
        "notes":                  { "type": "string" }
      }
    },

    "results": {
      "type": "object",
      "properties": {
        "per_seed": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "seed":           { "type": "integer" },
              "Y_actual":       { "type": "number" },
              "Y_permuted":     { "type": "number" },
              "Y_none":         { "type": "number" },
              "delta_content":  { "type": "number" },
              "delta_structure":{ "type": "number" },
              "delta_net":      { "type": "number" }
            }
          }
        },
        "content": {
          "mean":       { "type": "number" },
          "sd":         { "type": "number" },
          "sign_count": { "type": "integer" },
          "n":          { "type": "integer" },
          "gate_threshold": { "type": "integer" },
          "gate_result": { "enum": ["PASS", "FAIL"] }
        },
        "structure": {
          "mean":       { "type": "number" },
          "sd":         { "type": "number" },
          "sign_count": { "type": "integer" },
          "n":          { "type": "integer" },
          "gate_threshold": { "type": "integer" },
          "gate_result": { "enum": ["PASS", "FAIL"] }
        },
        "net": {
          "mean": { "type": "number" },
          "sd":   { "type": "number" },
          "confirmatory_lower_bound": {
            "type": ["number", "null"],
            "description": "null if engagement_type is diagnostic"
          },
          "confirmatory_gate_result": {
            "type": ["string", "null"],
            "enum": ["PASS", "FAIL", null]
          }
        }
      }
    },

    "diagnosis": {
      "type": "object",
      "properties": {
        "pattern": {
          "enum": [
            "content_drives_gain",
            "structure_drives_gain",
            "content_harmful",
            "communication_inert",
            "mixed"
          ]
        },
        "plain_language": { "type": "string" }
      }
    },

    "certificate_text": {
      "type": "string",
      "description": "Human-readable summary suitable for an engineering manager or V&V report"
    }
  }
}
```

---

## Example certificate (SPS WO-07C reference case)

```json
{
  "meta": {
    "system_name": "Stochastic Particle System",
    "protocol_name": "shared_summary_v2",
    "diagnostic_version": "0.1",
    "run_date": "2026-08-01",
    "seed_set": [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008],
    "metric_name": "unique_particles_captured",
    "metric_higher_is_better": true,
    "engagement_type": "diagnostic"
  },
  "conditions": {
    "A": "shared_summary_v2: count-weighted team mean velocity + validity fraction",
    "B": "shared_summary_v2_shuffled: same format, messages permuted across agents at each step",
    "C": "capacity_matched_independent: agents receive own local estimate only (no shared channel)"
  },
  "assumptions": {
    "matched_initialization": "verified",
    "matched_stochasticity": "verified",
    "no_leakage": "verified",
    "stationary_protocol": "verified",
    "notes": "Matched Brownian noise tensor and initial positions confirmed per SPS environment contract."
  },
  "results": {
    "per_seed": [
      {"seed": 7001, "Y_actual": 11.0, "Y_permuted": 10.0, "Y_none": 8.0, "delta_content": 1.0, "delta_structure": 2.0, "delta_net": 3.0},
      {"seed": 7002, "Y_actual": 10.0, "Y_permuted": 9.0,  "Y_none": 7.0, "delta_content": 1.0, "delta_structure": 2.0, "delta_net": 3.0}
    ],
    "content": {
      "mean": 0.875, "sd": 1.96, "sign_count": 4, "n": 8,
      "gate_threshold": 5, "gate_result": "FAIL"
    },
    "structure": {
      "mean": 1.625, "sd": 1.77, "sign_count": 5, "n": 8,
      "gate_threshold": 5, "gate_result": "PASS"
    },
    "net": {
      "mean": 2.5, "sd": 1.41,
      "confirmatory_lower_bound": null,
      "confirmatory_gate_result": null
    }
  },
  "diagnosis": {
    "pattern": "structure_drives_gain",
    "plain_language": "The communication protocol provides reliable benefit, but most of that benefit comes from the structural scaffold of shared signalling rather than the specific content of the messages. The protocol captures +2.5 particles per episode over no communication; +1.6 of that is attributable to format alone. The content adds a further +0.9 on average but does not pass the sign-count gate (4/8 seeds positive). Recommendation: the content encoding can likely be improved without changing the channel architecture."
  },
  "certificate_text": "COORDINATION DIAGNOSTIC CERTIFICATE\n\nSystem: Stochastic Particle System\nProtocol: shared_summary_v2\nDate: 2026-08-01\nSeeds: 8 (diagnostic)\n\nNet coordination value: +2.5 particles/episode (8/8 seeds positive)\nGain from structure: +1.6 (5/8 positive) — GATE PASS\nGain from content: +0.9 (4/8 positive) — GATE FAIL\n\nDiagnosis: STRUCTURE_DRIVES_GAIN\n\nThe protocol adds significant value over no communication. The coordination scaffold (shared channel structure) accounts for most of the observed gain. Message content contributes positively on average but inconsistently across episodes. The content encoding is a candidate for improvement.\n\nAssumptions: All four verified.\nThis certificate covers the specific conditions and seed set listed above. It does not certify the system's behaviour under untested conditions."
}
```

---

## PDF rendering

The `certificate_text` field is rendered to PDF for human delivery. The JSON is the machine-readable record. Both are delivered together. The PDF is not the authoritative record — the JSON is.
