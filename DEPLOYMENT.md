# Hosting the inspection app

The Docker entrypoint runs the actual detector and gated Moondream model. It does
not substitute prerecorded results. Plan for at least 16 GB RAM for CPU fp32
inference, a writable model cache, and outbound HTTPS access for pinned model files.
The first permitted explanation is slower while Moondream downloads and loads.

## Permanent hosting

Use a Docker-capable hosting account with sufficient memory. The app listens on
`0.0.0.0:7860` (override with `PORT`) and its health endpoint is `/config`.

```powershell
docker build -t crack-inspection .
docker run --rm -p 7860:7860 -v crack-model-cache:/home/app/.cache/huggingface crack-inspection
```

The detector is downloaded from its pinned repository revision and its SHA-256 is
verified before the server starts. The calibration artifact is retained. Requests
are queued one at a time, with at most eight waiting requests and a 20 MB upload
limit. No private model token is required for these public model downloads.

Hugging Face's [current hardware documentation](https://huggingface.co/docs/hub/spaces-gpus)
states that CPU Basic has no hourly charge, but creating a new Docker or Gradio
compute Space requires a paid plan (checked October 6, 2026). This is not a
fully free hosting option for a new free account.

For Hugging Face Spaces, create a **Docker** Space and upload this repository's
application files with `Dockerfile`, `.dockerignore`, `deployment.py`,
`requirements.txt`, and `models/*.json`. Set its README metadata to:

```yaml
---
title: Crack Inspection
sdk: docker
app_port: 7860
---
```

Do not upload `.venv`, downloaded datasets, evaluation image files, or private
credentials. Confirm the provider's current plan requirements before enabling
compute. A connected hosting account is required to publish the permanent URL.

## Temporary public demo

From the repository directory:

```powershell
.\.venv\Scripts\python.exe app.py --share
```

Use the public URL printed by Gradio. The computer and server process must remain
running; this is a temporary tunnel, not permanent hosting. Stop the server with
Ctrl+C in its terminal. Without `--share`, only the local homepage is started at
http://127.0.0.1:7860.

## Deployment status

The hosting configuration is prepared. A permanent cloud deployment has not yet
been created or tested; no hosting account is connected in the current workspace.

The September 21, 2026 Gradio public link has expired. On October 6, the local
server was restarted and `/config` returned HTTP 200 at http://127.0.0.1:7860.
Gradio could not create a share link. A Cloudflare Quick Tunnel also failed:
its diagnostics reported QUIC and HTTP/2 connections blocked or unreachable,
and the public endpoint returned HTTP 530. No working public URL was established.

For normal local use, run the app in a terminal and keep that terminal open:

```powershell
.\.venv\Scripts\python.exe app.py
```

Stop it with Ctrl+C. Restart it after closing the terminal or rebooting the
computer. Historical logs and PID files in `.runtime` may refer to older sessions.
See [Test and share](TEST_AND_SHARE.md) for the downloadable Windows package.

The Docker image has not been built or smoke-tested here because the local Docker daemon is unavailable. The hosting entrypoint has unit coverage for pinned downloads and checksum rejection.
