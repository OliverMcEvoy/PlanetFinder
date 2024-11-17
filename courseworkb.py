import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import scipy.signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from lightkurve import search_lightcurve

def fetch_kepler_data_and_stellar_info(target):
    search_result = search_lightcurve(target, mission="Kepler")
    lc_collection = search_result.download_all()

    time, flux, error = np.array([]), np.array([]), np.array([])
    for lc in lc_collection:
        lc_data = lc.normalize().remove_nans()
        tmptime = lc_data.time.value
        tmpflux = lc_data.flux.value
        tmperror = lc_data.flux_err.value

        array_size = len(tmpflux)
        window_length = min(501, array_size - (array_size % 2 == 0))

        if window_length > 2:
            interp_savgol = savgol_filter(tmpflux, window_length=window_length, polyorder=3)
        else:
            interp_savgol = np.ones_like(tmpflux)

        time = np.append(time, tmptime)
        flux = np.append(flux, tmpflux / interp_savgol)
        error = np.append(error, tmperror)

    df = pd.DataFrame({"time": time, "flux": flux, "error": error})
    mean_flux = np.mean(flux)
    std_flux = np.std(flux)
    df = df[(df["flux"] <= mean_flux + 2 * std_flux) & (df["flux"] >= mean_flux - 8 * std_flux)]

    # Fetch stellar information from the light curve metadata
    if len(lc_collection) > 0:
        star_data = lc_collection[0].meta
        stellar_params = {
            "stellar_radius": star_data.get("RADIUS", np.nan),  # Stellar radius in solar radii
            "temperature": star_data.get("TEFF", np.nan),   # Stellar effective temperature
        }
    else:
        stellar_params = None

    return df, stellar_params

# Find transit peaks (Lomb-Scargle or Fourier Spectrum)
def find_transits(x, y):
    freqs = np.linspace(1/(x.iloc[-1] - x.iloc[0]), 1/(x.iloc[1] - x.iloc[0]), 15000)
    lomb = scipy.signal.lombscargle(x, y, freqs, precenter=True)
    period = 1 / freqs
    return period, lomb

def run_bls_analysis(time, flux, error, min_period, max_period, duration_range=(0.01, 0.1)):
    bls = BoxLeastSquares(time, flux, dy=error)
    periods = np.linspace(min_period, max_period, 5000)
    durations = np.linspace(duration_range[0], duration_range[1], 100)
    results = bls.power(periods, durations)

    # Find the best period based on maximum power
    max_power_idx = np.argmax(results.power)
    best_period = results.period[max_power_idx]
    best_duration = results.duration[max_power_idx]
    best_transit_time = results.transit_time[max_power_idx]

    # Debugging: Print intermediate results to verify values
    print(f"Best Period: {best_period}, Best Duration: {best_duration}, Best Transit Time: {best_transit_time}")

    # Validate inputs before calling `bls.model()`
    if not (np.isfinite(best_period) and np.isfinite(best_duration) and np.isfinite(best_transit_time)):
        raise ValueError("Invalid input detected for BoxLeastSquares.model(). Ensure periods, durations, and transit times are finite.")

    # Generate the best transit model
    best_transit_model = bls.model(time, best_period, best_duration, best_transit_time)
    return results, results.power, results.period, best_period, best_transit_model

def estimate_planet_radius(transit_depth, stellar_radius):

    planet_radius = np.sqrt(transit_depth) * stellar_radius 
    return planet_radius


# Analyze peaks with BLS
def analyze_peaks_with_bls(time, flux, error, peak_periods, duration_range=(0.01, 0.1)):
    results_list = []
    for period in peak_periods:
        print(f"Analyzing candidate period: {period:.2f} days")

        # Skip periods that are too short for a valid transit duration
        if period < duration_range[1]:
            print(f"Skipping period {period:.2f} days as it's shorter than the maximum transit duration.")
            continue

        results, _, _, best_period, best_transit_model = run_bls_analysis(
            time,
            flux,
            error,
            min_period=period * 0.9,
            max_period=period * 1.1,
            duration_range=duration_range
        )
        
        if best_period is not None:
            results_list.append({
                "candidate_period": period,
                "refined_period": best_period,
                "transit_model": best_transit_model,
                "power": max(results.power),
                "duration": results.duration[np.argmax(results.power)],
                "depth": results.depth[np.argmax(results.power)],
            })
    return results_list


# Visualize phase-folded light curves
def plot_phase_folded_light_curves(time, flux, results_list):
    for i, result in enumerate(results_list):
        period = result["refined_period"]
        model_flux = result["transit_model"]
        phase = (time % period) / period
        plt.figure(figsize=(10, 5))
        plt.scatter(phase, flux, s=1, label="Flux")
        plt.plot(phase, model_flux, color="red", label=f"Planet {i+1} Transit Model")
        plt.xlabel("Phase")
        plt.ylabel("Normalized Flux")
        plt.title(f"Phase-Folded Light Curve for Candidate Planet {i+1}")
        plt.legend()
        plt.show()

# Summarize results
def summarize_results(results_list,stellar_info):
      print("\nDetected Planet Candidates:")
      print("-" * 40)
      for i, result in enumerate(results_list):
            
            print(f"Candidate {i+1}:") 
            print(f"  Initial Period = {result['candidate_period']:.2f} days")
            print(f"  Refined Period = {result['refined_period']:.2f} days")
            print(f"  Transit Depth  = {result['depth']:.2e}")
            print(f"  Transit Duration = {result['duration']:.2f} days")
            print(f"  Power = {result['power']:.2f}")
            print("-" * 40)
            if stellar_info:
                  stellar_radius = stellar_info["stellar_radius"]
                  depth = result["depth"]
                  planet_radius = estimate_planet_radius(depth, stellar_radius)
                  print(f"Best Transit Candidate: Period = {result['refined_period']:.2f} days, Depth = {depth:.2e}")
                  print(f"Estimated Planet Radius: {planet_radius:.2f} Earth Radii")

def run_lomb_scargle_analysis(kepler_dataframe):
      print("Running Lomb-Scargle Periodogram Analysis...")
      period, lomb2 = find_transits(kepler_dataframe["time"], kepler_dataframe["flux"])
      
      # Calculate height parameter as the 99th percentile of the power values
      height_threshold = np.percentile(lomb2, 99)
      
      # Find initial peaks to calculate the median period difference
      initial_peaks = find_peaks(lomb2, height=height_threshold)
      initial_peak_pos = period[initial_peaks[0]]
      
      # Calculate distance parameter as a fraction of the median period difference
      if len(initial_peak_pos) > 1:
            median_period_diff = np.median(np.diff(initial_peak_pos))
            distance_threshold = max(median_period_diff / 2, 1)  # Ensure distance is at least 1
      else:
            distance_threshold = 50  # Default value if not enough peaks are found
      
      # Adjust the height and distance parameters to reduce the number of peaks
      peaks = find_peaks(lomb2, height=height_threshold, distance=distance_threshold)
      peak_pos = period[peaks[0]]
      print("Lomb-Scargle Periodogram analysis done")

      # Visualize spectrum and peaks
      plt.figure(figsize=(10, 6))
      plt.plot(period, lomb2, label="Lomb-Scargle Power")
      plt.scatter(peak_pos, lomb2[peaks[0]], color="red", label="Detected Peaks")
      plt.xscale("log")
      plt.xlabel("Period (days)")
      plt.ylabel("Power")
      plt.title("Lomb-Scargle Periodogram")
      plt.legend()
      plt.show()

      return peak_pos

if __name__ == "__main__":
      print("Running Kepler Light Curve Analysis...")
      target = "Kepler-8"
      kepler_dataframe, stellar_info = fetch_kepler_data_and_stellar_info(target)
      print(f"Kepler Light Curve Data for {target} Fetched and Normalised")
      print(f"Stellar Information: {stellar_info}")

      peak_pos = run_lomb_scargle_analysis(kepler_dataframe)

      print("Detected Peaks:")
      print("-" * 40)
      print("Period (days)")
      print("-" * 40)
      for period in peak_pos:
            print(f"{period:.2f}")
      
      # Refine peaks using BLS
      results_list = analyze_peaks_with_bls(
            kepler_dataframe["time"].values,
            kepler_dataframe["flux"].values,
            kepler_dataframe["error"].values,
            peak_pos
      )
      print("BLS Analysis Complete!")

      # Visualize phase-folded light curves
      #plot_phase_folded_light_curves(kepler_dataframe["time"].values, kepler_dataframe["flux"].values, results_list)


      # Stellar information

      # Summarize results
      summarize_results(results_list,stellar_info)

