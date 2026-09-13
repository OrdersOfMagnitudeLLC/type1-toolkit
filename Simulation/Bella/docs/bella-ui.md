# Bella Streamlit UI

The lightweight, no-Docker, no-server GUI for Bella. Runs on any laptop with Python.

## Install

```bash
pip install streamlit
```

## Launch

```bash
bella ui
```

Bella opens a browser tab at `http://localhost:8501`.

## What it does

- **Domain selector**: choose from the built-in domain profiles.
- **Run discover**: one button to launch the full Bob → NSMace → SPARC → phonon pipeline. Output streams to the browser.
- **Results table**: shows the latest candidates with confidence scores.
- **Download PDF**: generate and download the Bella report PDF.
- **Chat**: single-turn questions handled by `bella chat`.

## Offline

The UI itself works offline. Only the actual discovery calls (Materials Project, MACE downloads, SPARC) need network access.

## Optional: Open WebUI integration (requires Docker)

If you have Docker and prefer a richer chat-style interface, you can also run Bella through an Open WebUI integration. This is completely optional: the Streamlit UI works without Docker or any Open WebUI setup.

## For low-income researchers

This lightweight Streamlit UI is the recommended GUI for machines that cannot run Docker. It requires only Python, pip, and `streamlit`.

## Stop

Press `Ctrl+C` in the terminal to stop the server.
