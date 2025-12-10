'''
    Run the lumi pipeline to process ERA5 data into daily statistics and regridded NPZ files.
    hourly --> daily statistics --> regridded --> daily NPZ files
    Resumes cleanly - a year is skipped, when all daily .npz files already exist
'''
import argparse
import logging
import multiprocessing
import os
import sys
import datetime
import pathlib
import yaml
import calendar
import shutil

from era5_download_pipeline.pipeline import cdo_utils


# Set up logging
logger = logging.getLogger(__name__)

# --- Helpers ---
def year_complete(vshort:str, 
                  year: int,
                  cfg: dict,
                  plev: int | None = None) -> bool: # type: ignore
    """
        Return True if all daily .npz files for <vshort, year> or <vshort, plev, year> already exist.
    """
    base = pathlib.Path(cfg['lumi']['npz_dir'].format(var=vshort, plev=plev) 
                        if plev is not None else cfg['lumi']['npz_dir'].format(var=vshort)
                        )
    target = base / str(year)
    if not target.is_dir():
        return False  # No directory means no files exist
    
    exp = 366 if calendar.isleap(year) else 365
    if len(list(target.glob("*.npz"))) == exp:
        return True  # All expected files are present
    
    logger.warning("\nYear %s for variable %s (pressure level: %s) is incomplete. Expected %d NPZ files, found %d.", 
                   year, vshort, plev if plev is not None else "N/A",
                   exp, len(list(target.glob("*.npz"))))
    shutil.rmtree(target)  # Clean up incomplete directory 

    return False  # Incomplete, will reprocess

def process_year(args):
    """
        Obe worker; hourly --> daily --> regrid --> daily npz, then clean up
    """
    vshort, year, cfg, plev = args
    tag = f"{vshort}_{plev}_{year}" if plev is not None else f"{vshort}_{year}"
    log = logging.getLogger(f"lumi.{tag}")

    # --- Paths (raw/daily/npz) ---
    raw_root = pathlib.Path(cfg['lumi']['raw_dir'].format(var=vshort, plev=plev) if plev is not None else cfg['lumi']['raw_dir'].format(var=vshort))
    daily_root = pathlib.Path(cfg['lumi']['daily_dir'].format(var=vshort, plev=plev) if plev is not None else cfg['lumi']['daily_dir'].format(var=vshort))
    npz_root = pathlib.Path(cfg['lumi']['npz_dir'].format(var=vshort, plev=plev) if plev is not None else cfg['lumi']['npz_dir'].format(var=vshort))

    raw_nc = raw_root / (f"{vshort}_{plev}_{year}.nc" if plev is not None else f"{vshort}_{year}.nc")
    daily_nc = daily_root / (f"{vshort}_{plev}_{year}_daily.nc" if plev is not None else f"{vshort}_{year}_daily.nc")
    rg_nc = daily_nc.with_name(daily_nc.stem + "_DG.nc")
    npz_dir = npz_root / str(year)

    exp_days = 366 if calendar.isleap(year) else 365

    # --- 1. Hourly --> Daily Statistics ---
    log.info("Processing year %s for variable %s (pressure level: %s)", year, vshort, plev if plev is not None else "N/A")
    if daily_nc.exists():
        log.info("Daily file %s already exists, skipping conversion hourly-->daily for year %s", daily_nc, year)
    elif raw_nc.exists():
        log.info("Raw file %s exists, proceeding with conversion hourly-->daily for year %s", raw_nc, year)
        cdo_utils.convert_to_daily_stat(
            raw_nc, 
            cfg['variables_by_short'][vshort]['daily_stat'],
            daily_nc
        )
        if cfg.get('delete_on_fly', False):
            log.info("Deleting raw file %s after processing", raw_nc)
            raw_nc.unlink(missing_ok=True)  # Remove the hourly raw file after processing
        log.info("Converted %s to daily statistics: %s. Removed raw file %s", raw_nc, daily_nc, raw_nc)
    else:
        log.warning("Raw file %s missing - skipping year %s for variable %s (pressure level %s)", raw_nc, year, vshort, plev if plev is not None else "N/A")
        return  # Skip if raw file does not exist
    

    # --- 2. Regrid to DANRA grid ---
    if rg_nc.exists():
        log.info("Regridded file %s already exists, skipping regridding for year %s", rg_nc, year)
    else:
        weights = pathlib.Path(cfg['weights_file'])
        if not weights.exists():
            raise FileNotFoundError(f"Regridding weights file {weights} does not exist. Please generate it first.")

        log.info("Regridding %s to DANRA grid", daily_nc)
        cdo_utils.regrid_to_danra(
            daily_nc, 
            rg_nc, 
            weights, 
            cfg['grid_file']
        )
        if cfg.get('delete_on_fly', False):
            log.info("Deleting daily file %s after regridding", daily_nc)
            daily_nc.unlink(missing_ok=True)  # Remove the daily file after regridding
        log.info("Regridded %s to %s", daily_nc, rg_nc)
        
    # --- 3. Split regridded data into NPZ files ---
    if npz_dir.is_dir() and len(list(npz_dir.glob("*.npz"))) == exp_days:
        log.info("All expected NPZ files for year %s already exist in %s, skipping split to NPZ", year, npz_dir)
        if cfg.get('delete_on_fly', False):
            log.info("Deleting regridded file %s after confirming NPZ files exist", rg_nc)
            rg_nc.unlink(missing_ok=True)  # Remove the regridded file after conversion
        return  # Skip if all NPZ files already exist
    
    log.info("Splitting regridded data %s into NPZ files in %s", rg_nc, npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)
    cdo_utils.convert_daily_to_npz(
        cfg,
        rg_nc, 
        npz_dir, 
        year, 
        vshort,
        plevel = plev if plev is not None else None
    )
    if cfg.get('delete_on_fly', False):
        log.info("Deleting regridded file %s after conversion to NPZ", rg_nc)
        rg_nc.unlink(missing_ok=True)  # Remove the regridded file after conversion
    log.info("Converted %s to NPZ files in %s", rg_nc, npz_dir)



# Main

def setup_logging(log_dir:str,
                  log_file:str,
                  level:str = 'INFO'
                  ):
    """    
        Set up logging configuration.
    """
    fmt = '%(asctime)s | %(levelname)s | %(name)s  | %(message)s'
    date = '%Y-%m-%d %H:%M:%S'

    log_path = pathlib.Path(log_dir) / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=date))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=date,
        handlers=[
            file_handler,
            logging.StreamHandler(sys.stdout)  # Also log to stdout
        ]
    )

def main():
    """
        Main function to run the lumi pipeline.
    """

    # Add a pre-run check for the grid file

    

    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="cfg/era5_pipeline.yaml")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    p.add_argument("--n-workers", type=int, # default=multiprocessing.cpu_count(),
                   help="Number of worker processes to use for parallel processing. Override worker pool size (default: $SLURM_CPUS_PER_TASK or all available cores).")
    args = p.parse_args()

    setup_logging(
        log_dir="/scratch/project_xxxxxxxxx/user/Code/SBGM_SD/era5_download_pipeline/era5_logs",
        log_file=f"era5_preprocess_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=args.log_level.upper()
    )
    cfg = yaml.safe_load(open(args.cfg, encoding='utf-8'))
    logger.info("\n\nStarting lumi pipeline with config: %s\n", args.cfg)

    # Make a pre-run check for the grid file
    grid_file = pathlib.Path(cfg['grid_file'])
    if not grid_file.exists():
        logger.error("\n\n !!! Grid file %s does not exist. Please check the configuration. !!!", grid_file)
        sys.exit(1)

    # Reverse lookup once
    cfg['variables_by_short'] = {v['short']: v for v in cfg['variables'].values()}
    
    # Normalise pressure level entry - turn into list (possibly [None])
    plevs = cfg.get('pressure_levels', [])

    # If plevs is 'None', '', 0, ..., treat as single-level data
    if not plevs:
        plevs = [None]
    elif not isinstance(plevs, (list, tuple)):
        plevs = [plevs]  # Wrap single value in a list

    # Build task list
    todo = []
    for v in cfg['variables'].values():
        vshort = v['short']
        for year in range(cfg['years'][0], cfg['years'][1] + 1):
            for plev in plevs:
                # Skip work that is already complete
                if year_complete(vshort, year, cfg, plev):
                    tag = f"{vshort}_{plev}_{year}" if plev else f"{vshort}_{year}"
                    logger.info("Skipping %s for %d - all .npz files already exist", tag, year)
                    continue
                todo.append((vshort, year, cfg, plev))

    if not todo:
        logger.info("All years and variables are already processed. Exiting.")
        return

    # ---------- generate weights once (serial) ----------
    weights_file = pathlib.Path(cfg["weights_file"])
    if not weights_file.exists():
        sample = None
        for vshort, year, _, plev in todo:
            droot = pathlib.Path(
                cfg["lumi"]["daily_dir"].format(var=vshort, plev=plev)
                if plev is not None else
                cfg["lumi"]["daily_dir"].format(var=vshort)
            )
            cand = droot / (f"{vshort}_{plev}_{year}_daily.nc"
                            if plev is not None else f"{vshort}_{year}_daily.nc")
            if cand.exists():
                sample = cand
                break
        if sample is None:
            logger.error("\n\n !!! No daily file found to generate regridding weights; exiting. !!! \n")
            sys.exit(1)

        logger.info("\nGenerating bilinear weights once using %s\n", sample)
        cdo_utils.generate_regridding_weights(sample, weights_file, cfg["grid_file"])
    else:
        logger.info("\nUsing existing weights file %s\n", weights_file)
    

    # Setup the number of workers
    if args.n_workers:
        n_workers = args.n_workers
    elif 'SLURM_CPUS_PER_TASK' in os.environ:
        n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        n_workers = cfg.get('n_workers', multiprocessing.cpu_count())



    logger.info("\nProcessing %d tasks with %d workers.\n", len(todo), n_workers)
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.map(process_year, todo)

    logger.info("\n\nLumi pipeline completed successfully.\n")

if __name__ == "__main__":
    main()
    








# def process_year(args):
#     '''
#         Process a single year for a specific variable.
#         This function performs the following steps:
#         1. Convert hourly data to daily statistics.
#         2. Regrid the daily data to the DANRA grid.
#         3. Split the regridded data into NPZ files.
#     '''
#     vshort, yr = args
#     raw_nc = pathlib.Path(cfg['lumi']['raw_dir'].format(var=vshort)) / f"{vshort}_{yr}.nc"
#     daily_nc = pathlib.Path(cfg['lumi']['daily_dir'].format(var=vshort)) / f"{vshort}_{yr}_daily.nc"
#     npz_dir = pathlib.Path(cfg['lumi']['npz_dir'].format(var=vshort)) / str(yr)

#     # 1. day-stats
#     cdo_utils.convert_to_daily_stat(raw_nc, cfg['variables_by_short']['vshort']['daily_stat'], daily_nc)
#     raw_nc.unlink()  # Remove the hourly raw file after processing

#     # 2. regrid 
#     if not pathlib.Path(cfg['weights_file']).exists():
#         cdo_utils.generate_regridding_weights(daily_nc, cfg['weights_file'], cfg['grid_file'])
#     rg_nc = daily_nc.with_suffix("_DG.nc")
#     cdo_utils.regrid_to_danra(daily_nc, rg_nc, cfg['weights_file'], cfg['grid_file'])
#     daily_nc.unlink()  # Remove the daily file after regridding

#     # 2. split to npz
#     cdo_utils.convert_daily_to_npz(rg_nc, npz_dir, yr, vshort)
#     rg_nc.unlink()  # Remove the regridded file after conversion




# if __name__ == "__main__":
#     # build a reverse lookup once
#     cfg['variables_by_short'] = {v['short']: v for v in cfg['variables'].values()}
#     todo = [(v['short'], yr) for v in cfg['variables'].values()
#             for yr in range(cfg['years'][0], cfg['years'][1] + 1)]

#     with multiprocessing.Pool(processes=cfg.get('n_workers', 4)) as pool:
#         pool.map(process_year, todo)
