#!/usr/bin/env -S uv run --script
# SBATCH --name=disruption-db
# SBATCH --partition=long
# SBATCH --mem=512g
# SBATCH --ntasks=4
# SBATCH --begin=teatime

"""
DisruptionDB: large-scale disruption workflow for Alcator C-Mod, DIII-D, and MAST
"""

import glob
import os
import subprocess
import time
import warnings

import numpy as np
import xarray as xr
import zarr
from disruption_py.config import config
from disruption_py.core.utils.misc import get_rss
from disruption_py.machine.tokamak import Tokamak, resolve_tokamak_from_environment
from disruption_py.workflow import run
from tqdm.auto import tqdm


def main():
    """
    main driver
    """

    # configuration

    tokamak = resolve_tokamak_from_environment()

    warnings.filterwarnings(
        "ignore",
        message="Consolidated metadata is currently not part in the Zarr format 3 specification.*",
    )

    # physics

    query = "select shot from disruption_efit where rc=0;"
    if tokamak is Tokamak.CMOD:
        # static 1 kHz efit tree
        efit_tree = "efit21"
        shots = query
        # execution on MDSplus server allows high throughput
        processes = 8
    elif tokamak is Tokamak.D3D:
        # dynamic 1 kHz efit tree
        efit_tree = "disruption"
        config(tokamak).efit.runtag = "DISPY"
        shots = query
        # slow and steady wins the race
        processes = 4
    elif tokamak is Tokamak.MAST:
        # single efit tree
        efit_tree = None
        shots = "https://mastapp.site/parquet/level2/shots.parquet"
        # avoid 503 "slow down" or 429 "rate limit" errors
        processes = 1
    else:
        raise NotImplementedError(tokamak.name)

    # inputs/output

    shards = f"{tokamak.value}/"
    output = f"{tokamak.value}.zarr"

    print(f"Inputs: {os.path.realpath(shards)}")
    print(f"Output: {os.path.realpath(output)}")

    if os.path.exists(output):
        raise FileExistsError(output)

    print(f"DisruptionDB: {tokamak.value}")
    print("Memory: RSS = {:,.1f} MB, MaxRSS = {:,.1f} MB".format(*get_rss()))

    # data

    if os.path.exists(shards):

        # lazy load
        print("Loading cached files...")
        tt = -time.time()
        results = {
            f: xr.open_dataset(f) for f in tqdm(glob.glob(f"{tokamak.value}/*.nc"))
        }
        tt += time.time()
        print(f"Loaded {len(results):,} cached files in {tt / 60:.1f} m.")

    else:

        # execution
        print("Executing DisruptionPy...")
        tt = -time.time()
        results = run(
            tokamak=tokamak,
            methods=None,
            shots=shots,
            efit_tree=efit_tree,
            time_base="efit",
            output=shards,
            processes=processes,
            log_level="error",
        )
        tt += time.time()
        print(f"Executed DisruptionPy in {tt / 60:.1f} m.")

    print("Memory: RSS = {:,.1f} MB, MaxRSS = {:,.1f} MB".format(*get_rss()))

    # concatenation

    results = list(results.values())
    data_vars = list(results[0].data_vars)

    print(
        f"Concatenating {len(data_vars)} data variables from {len(results):,} files..."
    )

    tt = -time.time()

    with tqdm(total=len(data_vars) * len(results)) as pbar:

        for d in data_vars:

            tqdm.write(d)

            for step in ["load", "conc", "sort", "save"]:

                mem, _ = get_rss()
                t = -time.time()

                if step == "load":

                    data = []
                    attrs = results[0][d].attrs
                    for result in results:
                        if d in result:
                            data.append(result[d].load())
                            del result[d]
                        else:
                            tqdm.write(
                                "Missing variable! "
                                f"{d} from {result.encoding['source']}"
                            )
                            data.append(xr.DataArray(result.idx * np.nan))
                        pbar.update(1)

                elif step == "conc":

                    out = xr.Dataset({d: xr.concat(data, dim="idx")}, attrs=attrs)

                elif step == "sort":

                    out = out.sortby(["shot", "time"])

                elif step == "save":

                    out.to_zarr(store=output, mode="a", consolidated=True)

                t += time.time()
                rss, _ = get_rss()

                tqdm.write(
                    f"\t{step:4s} = {t:6.2f} s / {rss-mem:+7,.1f} MB @ {rss:7,.1f} MB"
                )

    tt += time.time()
    print(
        f"Concatenated {len(data_vars)} data variables "
        f"from {len(results):,} files in {tt / 60:.1f} m."
    )
    print("Memory: RSS = {:,.1f} MB, MaxRSS = {:,.1f} MB".format(*get_rss()))

    # metadata

    xr.Dataset(attrs=results[0].attrs).to_zarr(
        store=output, mode="a", consolidated=True
    )
    zarr.consolidate_metadata(output)

    # summary

    ds = xr.open_zarr(output)

    memory = sum(v.nbytes for v in [ds] + list(ds.coords.values())) / 1024**2
    print(f"Dataset: {memory:,.0f} MB in memory")

    disk, *_ = subprocess.check_output(["du", "-s", output], text=True).split()
    print(f"Zarr: {float(disk) / 1024:,.0f} MB on disk")

    # netcdf

    nc = f"{tokamak.value}.nc"
    if os.path.exists(nc):
        raise FileExistsError(nc)

    ds.to_netcdf(nc, encoding={k: {"zlib": True, "complevel": 1} for k in ds})
    disk = os.path.getsize(nc) / 1024**2
    print(f"NetCDF: {float(disk):,.0f} MB on disk")

    return ds


if __name__ == "__main__":
    print(main())
