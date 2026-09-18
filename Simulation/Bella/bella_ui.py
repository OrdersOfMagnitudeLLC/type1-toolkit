# OOM Commercial License v1.0
# Copyright 2026 Orders of Magnitude LLC
# See ./LICENSE.md for terms.
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""Lightweight Streamlit UI for Bella."""
import json
import os
import subprocess
import time
from pathlib import Path

import streamlit as st
import yaml


def _load_domain_profiles():
    """Load all built-in and user domain profile YAMLs."""
    profiles = {}
    dirs = [
        Path(__file__).resolve().parent / 'profiles',
        Path.home() / '.bella' / 'profiles',
    ]
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob('*.yaml')):
            try:
                with open(f) as fh:
                    data = yaml.safe_load(fh) or {}
                if data.get('name') is None:
                    data['name'] = f.stem
                profiles[data['name']] = data
            except Exception:
                pass
    return profiles


def _latest_findings():
    """Return the most recent findings JSON path."""
    findings_dir = Path('findings')
    if not findings_dir.exists():
        return None
    files = sorted(findings_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return files[0]


def _latest_pdf(run_prefix=None):
    """Return the most recent findings PDF, optionally matching a run prefix."""
    findings_dir = Path('findings')
    if not findings_dir.exists():
        return None
    pdfs = sorted(findings_dir.glob('*.pdf'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs:
        return None
    if run_prefix:
        for p in pdfs:
            if run_prefix in p.name:
                return p
    return pdfs[0]


def _run_bella_streamed(args, output_box, elapsed_box, timeout=600):
    """Run a bella subcommand, streaming stdout line by line into output_box.

    Returns (stdout, returncode, elapsed_seconds).
    """
    cmd = ['bella'] + [str(a) for a in args]
    start = time.perf_counter()
    out_lines = []
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        ) as proc:
            for line in iter(proc.stdout.readline, ''):
                out_lines.append(line)
                output_box.text(''.join(out_lines))
                elapsed_box.text(f"Elapsed: {time.perf_counter() - start:.1f}s")
            proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out_lines.append('\n[timeout]')
    except Exception as e:
        out_lines.append(f'\n[error: {e}]')
    total = time.perf_counter() - start
    stdout = ''.join(out_lines)
    return stdout, proc.returncode if 'proc' in locals() else 1, total


def _command_panel(title, command, build_form, run_args_fn):
    """Generic streaming command panel."""
    st.header(title)
    with st.form(f'{command}_form'):
        values = build_form()
        submitted = st.form_submit_button(f'Run {command}')
    if submitted:
        args = run_args_fn(values)
        output_box = st.empty()
        elapsed_box = st.empty()
        stdout, rc, elapsed = _run_bella_streamed(args, output_box, elapsed_box, timeout=1200)
        elapsed_box.text(f"Elapsed: {elapsed:.1f}s")
        st.code(stdout)
        if rc == 0:
            st.success(f'{command.capitalize()} complete')
            pdf = _latest_pdf()
            if pdf and pdf.exists():
                with open(pdf, 'rb') as f:
                    st.download_button('Download Report (PDF)', f, pdf.name)
        else:
            st.error(f'{command.capitalize()} failed with exit code {rc}')


def main():
 st.title('Bella: Lightweight Universal Simulator')
    st.markdown("No Docker. No server. Just `pip install streamlit` and `bella ui`.")

    # Sidebar: domain profile selector
    st.sidebar.header('Domain profile')
    profiles = _load_domain_profiles()
    if not profiles:
        st.sidebar.error('No domain profiles found in profiles/ or ~/.bella/profiles/')
        return
    names = list(profiles.keys())
    selected = st.sidebar.selectbox(
        'Domain',
        names,
        help='Select a domain to pre-fill the discover form. Notes for the selected profile appear below.',
    )
    profile = profiles[selected]
    st.sidebar.markdown(f"**{profile.get('name', selected)}**")
    st.sidebar.markdown(profile.get('description', '_No description_'))
    if 'required_elements' in profile:
        st.sidebar.markdown(f"Required elements: {', '.join(profile['required_elements'])}")
    if profile.get('notes'):
        st.sidebar.caption(f"Notes: {profile['notes']}")

    # Discover
    st.header('1. Discover')
    st.markdown(f"Selected domain: **{selected}** : {profile.get('description', '')}")
    with st.form('discover_form'):
        sparc_quality = st.selectbox('SPARC quality', ['screen', 'confirm'], index=0)
        generative = st.checkbox('Generative mode', value=True)
        search_only = st.checkbox('Search only', value=False)
        limit = st.number_input('Candidate limit', min_value=1, max_value=5000, value=50)
        run = st.form_submit_button('Run discover')
    if run:
        args = ['discover', '--domain', selected, '--headless', '--sparc-quality', sparc_quality]
        if generative:
            args.append('--generative')
        if search_only:
            args.append('--search-only')
        args += ['--limit', str(limit)]

        output_box = st.empty()
        elapsed_box = st.empty()
        stdout, rc, elapsed = _run_bella_streamed(args, output_box, elapsed_box, timeout=1200)
        elapsed_box.text(f"Elapsed: {elapsed:.1f}s")

        if rc == 0:
            st.success('Discovery complete')
            st.code(stdout)
            pdf = _latest_pdf()
            if pdf and pdf.exists():
                with open(pdf, 'rb') as f:
                    st.download_button('Download Report (PDF)', f, pdf.name)
        else:
            st.error(f'Discovery failed with exit code {rc}')
            st.code(stdout)

    # Adsorb
    _command_panel(
        '2. Adsorb',
        'adsorb',
        lambda: {
            'formula': st.text_input('Surface formula (e.g. MoS2)', 'MoS2'),
            'molecule': st.text_input('Adsorbate molecule', 'N2'),
        },
        lambda v: ['adsorb', v['formula'], v['molecule']],
    )

    # NEB
    _command_panel(
        '3. NEB',
        'neb',
        lambda: {
            'initial': st.text_input('Initial formula', 'Li-metal'),
            'final': st.text_input('Final formula', 'Li-fcc'),
        },
        lambda v: ['neb', v['initial'], v['final']],
    )

    # Simulate
    _command_panel(
        '4. Simulate',
        'sim',
        lambda: {
            'equations': st.text_input('Equation string', 'reaction-diffusion'),
            'dim': st.selectbox('Dimension', [1, 2, 3], index=1),
        },
        lambda v: ['sim', '--equations', v['equations'], '--dim', str(v['dim'])],
    )

    # Latest results
    st.header('5. Latest results')
    findings = _latest_findings()
    if findings:
        with open(findings) as f:
            data = json.load(f)
        results = data.get('results', data.get('confirmed_results', []))
        if results:
            rows = []
            for r in results:
                rows.append({
                    'Formula': r.get('formula'),
                    'Source': r.get('source'),
                    'NSMace (eV)': r.get('nsmace_energy'),
                    'Phonon stable': r.get('phonon_stable'),
 'Confidence': f"{r.get('confidence', ': ')}",
                })
            st.dataframe(rows)
            with open(findings, 'rb') as f:
                st.download_button('Download findings JSON', f, findings.name)
    else:
        st.info('No findings yet. Run a discovery first.')

    # Chat
    st.header('6. Chat')
    msg = st.text_input('Ask Bella a question')
    if st.button('Send') and msg:
        output_box = st.empty()
        elapsed_box = st.empty()
        stdout, rc, _ = _run_bella_streamed(['chat', '--message', msg], output_box, elapsed_box, timeout=120)
        st.code(stdout)
        if rc != 0:
            st.error('Chat command failed')


if __name__ == '__main__':
    main()
