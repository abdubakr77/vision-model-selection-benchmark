# results

- `per_image/`: one CSV per model and dataset. Columns: image_id, model, prediction, score, latency_ms, vram_mb, raw_output, parse_ok, tokens_in, tokens_out.
- `tables/`: aggregated comparison tables.
- `figures/`: charts used in the docs and blog.

Cached cloud API responses stay local (git-ignored) so runs are reproducible without paying twice.
