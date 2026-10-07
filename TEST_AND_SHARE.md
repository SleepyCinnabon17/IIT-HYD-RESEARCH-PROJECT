# Test and share the inspection app

## Test on your computer

While the server is running, open http://127.0.0.1:7860.
Upload a JPG or PNG surface photograph and click the inspection button.
Check that the annotated image, observation, status, and Detailed evidence appear.
Try a clear crack, an uncertain surface/shadow, and an uncracked surface.
Review the detector results separately from the language gate: a withheld
description means human review is needed, not that a crack is absent.
The first permitted explanation downloads and loads Moondream and can be slow.
Do not treat the output as a structural safety assessment.

## Send a downloadable copy

Send `dist/Crack-Inspection-Demo.zip` using your preferred file-sharing service.
The recipient should:

1. Install Python 3.11 for Windows with the Python launcher (`py`).
2. Extract the entire ZIP into a writable folder; do not run inside the ZIP viewer.
3. Double-click `START_WINDOWS.bat` and keep its window open.
4. Wait for the local URL to appear, then open http://127.0.0.1:7860.

This is a runnable source package, not a standalone EXE or an offline HTML site.
Plan for at least 16 GB RAM, several GB of free disk space, and internet access
for dependency/model downloads. Models are downloaded on demand; no paid model
API key is required. Each recipient runs the computation on their own computer.
The ZIP includes source and configuration, not your virtual environment,
credentials, datasets, model weights, or uploaded images.

If setup fails, keep the error message. Confirm Python 3.11 is installed with
`py -3.11 --version`. Dependency installation on a clean recipient computer
has not been verified. Subsequent launches reuse the installed environment.

## Let someone on the same Wi-Fi open your running app

Stop the existing server first (Ctrl+C in its terminal), then run:

```powershell
.\.venv\Scripts\python.exe run_demo.py --host 0.0.0.0
ipconfig
```

Find your active Wi-Fi adapter's IPv4 address, for example `192.168.1.25`.
The other person opens `http://192.168.1.25:7860` using your actual address.
Their `localhost` points to their computer, so sending them your localhost link
does not work. Allow Python through Windows Firewall on your trusted private
network if prompted. Guest/campus Wi-Fi may prevent devices reaching each other.
Your computer and app must stay on. Anyone who can reach this port can use it.

For an extracted Windows package, `START_WINDOWS.bat --host 0.0.0.0` does the same.

## Let someone outside your network open it

On a network that permits Gradio's tunnel connection, stop the existing server
and run:

```powershell
.\.venv\Scripts\python.exe run_demo.py --share
```

Send the public HTTPS URL printed by Gradio. This is temporary public access;
your computer must remain awake and the process must remain running. The attempt
on October 6, 2026 failed from the current network, so no working public URL is
included here. A downloadable copy is an alternative when tunneling is blocked.

## Rebuild the ZIP after changing the app

```powershell
python package_demo.py
```

The packager uses an explicit file list to avoid including private workspace data.
