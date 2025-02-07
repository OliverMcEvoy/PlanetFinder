import multiprocessing.pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import scipy.signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from lightkurve import search_lightcurve
from scipy.optimize import minimize
import multiprocessing
from scipy.signal import medfilt
from lightkurve.lightcurve import TessLightCurve
from scipy.interpolate import interp1d
import SyntheticLightCurveGeneration as lightcurve_generator
from scipy.optimize import differential_evolution
from tqdm import tqdm
import importlib
import scipy.optimize as opt
import glob
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution, minimize, basinhopping
import matplotlib.ticker as ticker
from IPython.display import display, Math


def fetch_kepler_data_and_stellar_info(target, filter_type="savgol"):
    search_result = search_lightcurve(target, mission="Kepler")
    lc_collection = search_result.download_all()

    time, flux, error = np.array([]), np.array([]), np.array([])
    quart = 0
    for lc in lc_collection:
        print(
            f"Downloading light curve segment {quart + 1} of {len(lc_collection)}",
            end="\r",
        )
        quart += 1

        lc_data = lc.remove_nans()
        tmptime = lc_data.time.value
        tmpflux = lc_data.flux.value
        tmperror = lc_data.flux_err.value

        array_size = len(tmpflux)
        window_length = min(51, array_size - (array_size % 2 == 0))

        if window_length > 2:
            if filter_type == "savgol":
                normaliser = savgol_filter(
                    tmpflux, window_length=window_length, polyorder=3
                )
            elif filter_type == "medfilt":
                tmpflux = tmpflux.astype(np.float64)  # Convert to f64 for medfilt
                normaliser = medfilt(tmpflux, kernel_size=51)
        else:
            normaliser = np.ones_like(tmpflux)

        time = np.append(time, tmptime)
        flux = np.append(flux, tmpflux / normaliser)
        error = np.append(error, tmperror / normaliser)

    df = pd.DataFrame({"time": time, "flux": flux, "error": error})
    mean_flux = np.mean(flux)
    std_flux = np.std(flux)
    df = df[
        (df["flux"] <= mean_flux + 3 * std_flux)
        & (df["flux"] >= mean_flux - 8 * std_flux)
    ]

    if len(lc_collection) > 0:
        star_data = lc_collection[0].meta
        stellar_params = {
            "stellar_radius": star_data.get("RADIUS", np.nan),
            "temperature": star_data.get("TEFF", np.nan),
        }
    else:
        stellar_params = None

    return df, stellar_params


def summarize_results(results_list, stellar_info):
    print("\nDetected Planet Candidates:")
    print("-" * 40)

    if stellar_info:
        stellar_radius = stellar_info["stellar_radius"]
        print(f"Stellar Radius: {stellar_radius} Solar Radii")
        print(f"Stellar Temperature: {stellar_info['temperature']} K")
        print("-" * 40)

    for i, result in enumerate(results_list):

        print()
        print("-" * 40)
        print(f"Candidate {i+1}:")
        print(f"  Initial Period = {result['candidate_period']:.2f} days")
        print(f"  Refined Period = {result['refined_period']:.2f} days")
        print(f"  Transit Depth  = {result['depth']:.2e}")
        print(f"  Transit Duration = {result['duration']:.2f} days")
        print(f"  Power = {result['power']:.2f}")
        print("-" * 40)
        if stellar_info:
            depth = result["depth"]
            planet_radius = estimate_planet_radius(depth, stellar_radius)
            print(
                f"Best Transit Candidate: Period = {result['refined_period']:.2f} days, Depth = {depth:.2e}"
            )
            print(f"Estimated Planet Radius: {planet_radius:.3f} Solar Radii")
            earth_radius_in_terms_of_stellar = 0.009168
            jupiter_radius_in_terms_of_stellar = 0.10045
            print(
                f"Estimated Planet Radius: {planet_radius / earth_radius_in_terms_of_stellar :.3f} Earth Radii"
            )
            print(
                f"Estimated Planet Radius: {planet_radius / jupiter_radius_in_terms_of_stellar :.3f} Jupiter Radii"
            )

def estimate_planet_radius(transit_depth, stellar_radius):
    """
    Estimate the radius of a planet based on the transit depth and stellar radius.
    @params
    transit_depth: float -> the depth of the transit.
    stellar_radius: float -> the radius of the star.
    @returns
    float -> the estimated radius of the planet.
    """
    if transit_depth > 1:
        raise ValueError("Transit depth is not physical.")
    planet_radius = np.sqrt(transit_depth) * stellar_radius
    return planet_radius
