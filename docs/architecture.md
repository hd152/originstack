# OriginStack Architecture Diagram

This project is organized as a streaming astrophotography processing pipeline: raw frames are discovered, calibrated, aligned, stacked, and optionally post-processed without holding the full dataset in memory.

```mermaid
flowchart TB
    U[User / CLI / Desktop app]

    subgraph Entry[Entry and orchestration]
        CLI[originstack.py / cli.py]
        APP[desktop_app.py + desktop_control.py]
        AUTO[auto_settings.py + target_inference.py]
        CFG[Config + models.py]
    end

    subgraph Inputs[Inputs and frame discovery]
        FR[discover_frames.py]
        INFO[session_info.py]
        IO[io_fits.py\nio_raw.py\nio_tiff.py\nio_xisf.py\nio_ser.py]
    end

    subgraph Calibration[Calibration and quality]
        CAL[frame_processor.py]
        BIAS[master bias / dark / flat generation]
        Q[quality.py + star_detect.py]
        HOT[debayer.py + banding.py + vignette_calib.py]
        GPU[gpu_context.py + astro_native Rust kernels]
    end

    subgraph Phase2[Registration and alignment]
        REG[registration.py]
        AFF[affine_fit.py + phase_correlate.py + blind_match.py]
        TRANS[transparency.py + distortion.py]
    end

    subgraph Phase3[Stacking and merge]
        STACK[stacking.py]
        MERGE[merge.py]
        MOS[mosaic.py]
        CMP[channel_combine.py]
    end

    subgraph Phase4[Post-processing]
        POST[postprocess.py]
        BG[background.py]
        DENO[denoising.py + wavelet.py]
        PSF[psf_deconvolution.py]
        STAR[star_removal.py + star_repair.py]
        IMG[exposure_fusion.py + source_separation.py]
    end

    subgraph Analysis[Scientific analysis and outputs]
        PHOTO[photometry.py + photometric_calibration.py + photometry_core.py]
        SOLVE[plate_solve.py + local_solve.py]
        NET[net_query.py + gaia_cache.py + release check]
        REPORT[session_report.py + quality_sweep.py + annotation.py]
        OUT[Output FITS / previews / CSV / reports]
    end

    U --> CLI
    U --> APP
    CLI --> AUTO
    APP --> CLI
    APP --> NET
    AUTO --> CFG

    CLI --> FR
    FR --> IO
    FR --> INFO
    IO --> CAL
    INFO --> SOLVE

    CAL --> BIAS
    CAL --> HOT
    CAL --> Q
    CAL --> GPU

    Q --> REG
    REG --> AFF
    REG --> TRANS

    REG --> STACK
    STACK --> MERGE
    STACK --> MOS
    STACK --> CMP

    STACK --> POST
    POST --> BG
    POST --> DENO
    POST --> PSF
    POST --> STAR
    POST --> IMG

    SOLVE --> PHOTO
    PHOTO --> REPORT
    NET --> SOLVE
    STACK --> OUT
    POST --> OUT
    REPORT --> OUT

    classDef entry fill:#dbeafe,stroke:#1d4ed8,stroke-width:1px;
    classDef input fill:#dcfce7,stroke:#16a34a,stroke-width:1px;
    classDef core fill:#fef3c7,stroke:#d97706,stroke-width:1px;
    classDef output fill:#fce7f3,stroke:#db2777,stroke-width:1px;
    classDef ext fill:#ede9fe,stroke:#7c3aed,stroke-width:1px;

    class U,CLI,APP,AUTO,CFG entry;
    class FR,INFO,IO input;
    class CAL,BIAS,Q,HOT,GPU,REG,AFF,TRANS,STACK,MERGE,MOS,CMP,POST,BG,DENO,PSF,STAR,IMG core;
    class PHOTO,SOLVE,NET,REPORT,OUT output;
```

## Layered view

The project is built around a four-phase pipeline:

1. Input discovery and calibration
   - File format readers load FITS, RAW, TIFF, XISF, SER, and other camera frames.
   - Phase 1 normalizes channel-first RGB FITS cubes `(3, H, W)` to its internal channel-last `(H, W, 3)` layout.
   - `frame_discovery.py` classifies lights, darks, flats, and calibrations.
   - `frame_processor.py` runs the per-frame processing queue and quality scoring.

2. Registration
   - `registration.py` aligns frames onto a common coordinate system using star matching, phase correlation, and optional distortion models.
   - `affine_fit.py`, `phase_correlate.py`, and `blind_match.py` handle rigid transforms and unknown-rotation recovery.

3. Stacking and merging
   - `stacking.py` combines aligned frames into the linear master stack.
   - `merge.py` merges prior stacks into a larger hierarchical or multi-session result.
   - `mosaic.py` and `channel_combine.py` extend the stack into combined outputs and colour workflows.

4. Post-processing and analysis
   - `postprocess.py` orchestrates background extraction, denoising, star repair, deconvolution, and local contrast.
   - `photometry.py`, `plate_solve.py`, and `session_report.py` produce scientific and reporting outputs.
   - `net_query.py` and `gaia_cache.py` supply external catalog and WCS information when available.

## Key architectural principles

- Streaming memory model: frames are processed and released incrementally instead of loading the full dataset into memory.
- Modular pipeline: each phase is implemented in a dedicated module family with a clear responsibility boundary.
- Native acceleration: the hot paths can use `ext/astro_native` Rust kernels while falling back to NumPy/SciPy implementations.
- Desktop + CLI integration: the same pipeline is exposed via the terminal and the native desktop GUI.
- Desktop release checks run in the background and deliver any update notice through the Tkinter main-thread poll.
- Scientific extension: the stack is not just an image combine; it also supports plate solving, photometry, noise validation, and reporting.
