import zarr
import h5py
import os
import numpy as np
from astropy.timeseries import LombScargle
from tqdm import tqdm
import argparse

def save_lightcurves(path, total_time=1600, cadence=0.020833, max_planets=8, max_period=400, min_period=1, resolution=1500):
    for folder in ["power", "periods"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    frequency = np.linspace(1 / max_period, 1 / min_period, resolution)

    with h5py.File(path, "r") as file:
        keys = [f"{iteration}/{system}" for iteration in file.keys() for system in file[iteration].keys()]
        for i in tqdm(range(len(keys))):
            data = file[keys[i]]
            time = data["time"]
            flux_noise = data["flux_with_noise"]
            periods = []
            for j in range(max_planets):
                try:
                    periods.append(data[f"planets/planet_{j}/period"][()])
                except:
                    pass

            # Normalize periods to range [0, 1]
            periods = np.array(periods, dtype=np.float32)
            periods = np.sort(periods)[::-1]
            periods /= max_period
            periods = np.pad(periods, (0, (max_planets - len(periods))))
            zarr.save(f"periods/{i}_periods.zip", periods)

            # Initialize normalized flux array
            timestamps = np.arange(0, total_time, cadence)
            flux_noise = np.array(flux_noise, dtype=np.float32)
            flux = np.ones(len(timestamps), dtype=np.float32)
            time_index = {t: j for j, t in enumerate(timestamps)}
            for t, f in zip(time, flux_noise):
                if t in time_index:
                    flux[time_index[t]] = f

            # Normalize flux to range [0, 1]
            min_flux = np.min(flux)
            max_flux = np.max(flux)
            if max_flux - min_flux > 0:  # Avoid division by zero
                flux = (flux - min_flux) / (max_flux - min_flux)

            power = LombScargle(timestamps, flux).power(frequency)
            power = power[:,None] # Input size of 1
            zarr.save(f"power/{i}_power.zip", power)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for exoplanet detection.")
    parser.add_argument("-path", type=str, default="LightCurves.hdf5", help="Path to HDF5 lightcurve file.", required=True)
    parser.add_argument("--total_time", type=int, default=1600, help="Total observation time in days.")
    parser.add_argument("--cadence", type=float, default=0.0208333, help="Time between observations in days.")
    parser.add_argument("--max_planets", type=int, default=8, help="Maximum number of model output periods.")
    parser.add_argument("--max_period", type=int, default=400, help="Maximum period to normalise data.")
    parser.add_argument("--min_period", type=int, default=1, help="Minimum period to normalise data.")
    parser.add_argument("--resolution", type=int, default=1500, help="Resolution of Lomb-Scargle periodogram.")

    args = vars(parser.parse_args())
    save_lightcurves(**args)
