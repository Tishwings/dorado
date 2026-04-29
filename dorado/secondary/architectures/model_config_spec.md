# Secondary Model Config Specification

This document describes the TOML model config schema validated by
`dorado/secondary/architectures/model_config_validation.cpp`.

The config is versioned by the top-level `config_version` integer. Parameter
availability is controlled per key. Minimum versions are inclusive; maximum
versions are inclusive.

## Validation Model

The validator recursively walks the TOML table tree and builds dotted section
names such as `model`, `model.kwargs`, and `feature_encoder.kwargs`.

Typed sections use the section `type` value to select the parameter table. For
example:

```toml
[model]
type = "VariantPerceiver"

[model.kwargs]
read_max_depth = 100
```

selects the `model.kwargs::VariantPerceiver` parameter spec.

The validator checks:

| Check | Behavior |
| --- | --- |
| Unknown keys | Rejected in any section that has a validation spec. |
| Missing required keys | Rejected when the key is required for the current config version. |
| TOML value type | Rejected when the parsed TOML type does not match the spec. |
| Unknown `type` values | Rejected for typed sections. |
| Type version | Rejected when the chosen section type is not available for `config_version`. |

Sections without their own validation spec are not checked internally. They must
still be allowed as a table-valued key by their parent section. Currently this
applies to `model.kwargs.pooler_args`; the table is allowed where listed, but
its contents are not specified by the validator.

## Version Handling

| Spec behavior | In version range | Below minimum version | Above maximum version |
| --- | --- | --- | --- |
| Required | Key must exist. | Rejected unless an explicit below-min default behavior is configured. | Rejected unless an explicit above-max default behavior is configured. |
| Optional with default | If present, use it; otherwise use the default. | Same as normal unless additional version handling is configured. | Same as normal unless additional version handling is configured. |
| Required since version | Key must exist from `min_version` onward. | Use the specified historical default. | Same as required. |
| Parse or default before version | Key must exist from `min_version` onward. | If present, parse it; otherwise use the specified historical default. | Same as required. |
| Required until version | Key must exist through `max_version`. | Same as required. | Use the specified replacement default. |

The versioned getter strips quotes from TOML strings before returning them.
Arrays and tables are stored using their TOML-formatted string representation.

## Value Types

| Spec type | TOML examples | Notes |
| --- | --- | --- |
| `INTEGER` | `4`, `1000` | Floating point values are not accepted as integers. |
| `BOOLEAN` | `true`, `false` | Must be a TOML boolean, not a string. |
| `STRING` | `"VariantPerceiver"` | Returned without surrounding quotes by versioned getters. |
| `ARRAY` | `[1, 17]`, `[""]` | Element types are not validated by the schema. |
| `TABLE` | `[model.kwargs]` | Nested contents are checked only when a matching section spec exists. |

## Top-Level Keys

| Key | Type | Versions | Required | Default or note |
| --- | --- | --- | --- | --- |
| `config_version` | `INTEGER` | all | yes | Selects versioned validation behavior. |
| `basecaller_model` | `STRING` | before v4 | yes before v4 | In v4+, no longer required; versioned default is `""`. |
| `supported_basecallers` | `ARRAY` | v2+ | yes from v2 | Before v2, versioned default is `""`. |
| `chunk_size` | `INTEGER` | v4+ | yes from v4 | Before v4, default is `10000`. |
| `chunk_overlap` | `INTEGER` | v4+ | yes from v4 | Before v4, default is `1000`. |
| `model` | `TABLE` | all | yes | Typed section. |
| `feature_encoder` | `TABLE` | all | yes | Typed section. |
| `label_scheme` | `TABLE` | all | yes | Typed section. |

## Typed Sections

### `model`

`model` requires:

| Key | Type | Required | Note |
| --- | --- | --- | --- |
| `type` | `STRING` | yes | Selects the model architecture. |
| `kwargs` | `TABLE` | yes | Parameters are selected by `model.type`. |

Available model types:

| Type | Available since |
| --- | --- |
| `GRUModel` | v1 |
| `LatentSpaceLSTM` | v1 |
| `SlotAttentionConsensus` | v3 |
| `VariantPerceiver` | v4 |

### `feature_encoder`

`feature_encoder` requires:

| Key | Type | Required | Note |
| --- | --- | --- | --- |
| `type` | `STRING` | yes | Selects the feature encoder. |
| `kwargs` | `TABLE` | yes | Parameters are selected by `feature_encoder.type`. |

Available feature encoder types:

| Type | Available since |
| --- | --- |
| `CountsFeatureEncoder` | v1 |
| `ReadAlignmentFeatureEncoder` | v1 |

### `label_scheme`

`label_scheme` requires:

| Key | Type | Required | Note |
| --- | --- | --- | --- |
| `type` | `STRING` | yes | Selects the label scheme. |
| `kwargs` | `TABLE` | no | Defaults to an empty table. |

Available label scheme types:

| Type | Available since |
| --- | --- |
| `HaploidLabelScheme` | v1 |
| `DiploidLabelScheme` | v3 |

## Model Kwargs

### `model.kwargs` for `GRUModel`

Available since v1. All parameters are required.

| Key | Type |
| --- | --- |
| `num_features` | `INTEGER` |
| `num_classes` | `INTEGER` |
| `gru_size` | `INTEGER` |
| `n_layers` | `INTEGER` |
| `bidirectional` | `BOOLEAN` |

### `model.kwargs` for `LatentSpaceLSTM`

Available since v1.

| Key | Type | Required | Default or note |
| --- | --- | --- | --- |
| `num_classes` | `INTEGER` | yes |  |
| `lstm_size` | `INTEGER` | yes |  |
| `cnn_size` | `INTEGER` | yes |  |
| `pooler_type` | `STRING` | yes |  |
| `bases_alphabet_size` | `INTEGER` | yes |  |
| `bases_embedding_size` | `INTEGER` | yes |  |
| `kernel_sizes` | `ARRAY` | yes |  |
| `use_dwells` | `BOOLEAN` | yes |  |
| `bidirectional` | `BOOLEAN` | yes | Required for all `LatentSpaceLSTM` config versions. |
| `pooler_args` | `TABLE` | no | Defaults to an empty table. Contents are not validated. |

### `model.kwargs` for `SlotAttentionConsensus`

Available since v3.

| Key | Type | Required | Default or note |
| --- | --- | --- | --- |
| `num_slots` | `INTEGER` | yes |  |
| `classes_per_slot` | `INTEGER` | yes |  |
| `read_embedding_size` | `INTEGER` | yes |  |
| `cnn_size` | `INTEGER` | yes |  |
| `kernel_sizes` | `ARRAY` | yes |  |
| `pooler_type` | `STRING` | yes |  |
| `use_mapqc` | `BOOLEAN` | yes |  |
| `use_dwells` | `BOOLEAN` | yes |  |
| `use_haplotags` | `BOOLEAN` | yes |  |
| `use_snp_qv` | `BOOLEAN` | no | Defaults to `false`. |
| `bases_alphabet_size` | `INTEGER` | yes |  |
| `bases_embedding_size` | `INTEGER` | yes |  |
| `add_lstm` | `BOOLEAN` | yes |  |
| `use_reference` | `BOOLEAN` | yes |  |
| `pooler_args` | `TABLE` | no | Defaults to an empty table. Contents are not validated. |

### `model.kwargs` for `VariantPerceiver`

Available since v4. All parameters are required.

| Key | Type |
| --- | --- |
| `read_max_depth` | `INTEGER` |
| `ploidy` | `INTEGER` |
| `num_classes` | `INTEGER` |
| `cnn_size` | `INTEGER` |
| `kernel_sizes` | `ARRAY` |
| `dimension` | `INTEGER` |
| `num_blocks` | `INTEGER` |
| `num_heads` | `INTEGER` |
| `self_attn_layers_per_block` | `INTEGER` |
| `use_mapqc` | `BOOLEAN` |
| `use_dwells` | `BOOLEAN` |
| `use_haplotags` | `BOOLEAN` |
| `use_snp_qv` | `BOOLEAN` |
| `bases_alphabet_size` | `INTEGER` |
| `bases_embedding_size` | `INTEGER` |
| `use_decoder_lstm` | `BOOLEAN` |
| `use_per_read_embedding` | `BOOLEAN` |
| `embedding_type` | `STRING` |
| `update_read_embeddings` | `BOOLEAN` |
| `latent_init_method` | `STRING` |
| `shuffle_embeddings` | `BOOLEAN` |
| `mask_partial_rows` | `BOOLEAN` |
| `add_null_tokens` | `BOOLEAN` |

## Feature Encoder Kwargs

### `feature_encoder.kwargs` for `CountsFeatureEncoder`

Available since v1. All parameters are required.

| Key | Type |
| --- | --- |
| `normalise` | `STRING` |
| `tag_keep_missing` | `BOOLEAN` |
| `min_mapq` | `INTEGER` |
| `sym_indels` | `BOOLEAN` |
| `dtypes` | `ARRAY` |

### `feature_encoder.kwargs` for `ReadAlignmentFeatureEncoder`

Available since v1.

| Key | Type | Required | Default or note |
| --- | --- | --- | --- |
| `dtypes` | `ARRAY` | yes |  |
| `tag_keep_missing` | `BOOLEAN` | yes |  |
| `min_mapq` | `INTEGER` | yes |  |
| `max_reads` | `INTEGER` | yes |  |
| `row_per_read` | `BOOLEAN` | yes |  |
| `include_dwells` | `BOOLEAN` | yes |  |
| `include_haplotype` | `BOOLEAN` | yes |  |
| `include_snp_qv` | `BOOLEAN` | yes from v4 | Before v4, default is `false`. |
| `right_align_insertions` | `BOOLEAN` | yes from v4 | Before v4, parse if present; otherwise default to `false`. |
| `region_split` | `INTEGER` | no | Defaults to an empty value. |

`right_align_insertions` is intentionally more permissive before v4 because it
appeared inconsistently in v3 configs.

## Label Scheme Kwargs

The label scheme kwargs are recognized so configs containing them validate, but
they are not currently consumed by the label scheme factory.

These keys are accepted for both `HaploidLabelScheme` and `DiploidLabelScheme`.

| Key | Type | Required | Default or note |
| --- | --- | --- | --- |
| `ploidy` | `INTEGER` | no | Defaults to an empty value. |
| `ordered` | `BOOLEAN` | no | Defaults to an empty value. |
| `right_align_insertions` | `BOOLEAN` | no | Defaults to an empty value. |

## Minimal Shape

Every config must have this high-level shape:

```toml
config_version = 4
supported_basecallers = ["dna_r10.4.1_e8.2_400bps_hac@v5.2.0"]
chunk_size = 300
chunk_overlap = 100

[model]
type = "VariantPerceiver"

[model.kwargs]
# model-specific required keys

[feature_encoder]
type = "ReadAlignmentFeatureEncoder"

[feature_encoder.kwargs]
# encoder-specific required keys

[label_scheme]
type = "DiploidLabelScheme"

[label_scheme.kwargs]
# optional label-scheme keys
```
