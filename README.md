# frustracer — media host

This branch exists only so GitHub **Pages** can serve the README's video
assets, and it deliberately contains nothing else.

Why Pages, measured rather than assumed:

| route | content-type | nosniff | plays? |
|---|---|---|---|
| `raw.githubusercontent.com` | `application/octet-stream` | **yes** | no — `nosniff` forbids the browser decoding it |
| `releases/download/...` | octet-stream + `attachment`, signed URL expiring in 1 h | – | no, and not a stable URL |
| `*.github.io` (here) | real `video/mp4`, `Accept-Ranges: bytes` | no | **yes** |

GitHub's markdown renderer does *not* strip `<video>` — all five source domains
survive its `/markdown` API. The obstacle was always the serving headers.

Why a separate branch rather than `/docs` on master: Pages publishes everything
in its source tree, and `docs/` holds third-party academic PDFs and vendor
briefs. A branch that contains only videos cannot leak them.

Regenerate with `tools/media-encode.py av1` on master; see its `pages`
subcommand for the publish and header-verification steps.
