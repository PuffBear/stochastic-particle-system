# Source documents

- `CODEX_CLAUDE_PROJECT_BOOTSTRAP.md`: canonical implementation specification supplied by the project owner.
- `regime_a_benchmark_vision.txt`: searchable text extracted from the supplied three-page PDF.
- `regime_a_benchmark_vision.pdf.base64`: byte-preserving Base64 encoding of the original PDF.

Original PDF SHA-256:

`f68e9c6f4dad28c7774e2af7a9b0920f895fc70d54a5d125542c46684cd040ce`

Reconstruct the original PDF with:

```bash
base64 --decode docs/regime_a_benchmark_vision.pdf.base64 > regime_a_benchmark_vision.pdf
```
