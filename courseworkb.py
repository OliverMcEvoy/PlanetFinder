import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import scipy.signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from lightkurve import search_lightcurve
import multiprocessing

def fetch_kepler_data_and_stellar_info(target):
    search_result = search_lightcurve(target, mission="Kepler")
    lc_collection = search_result.download_all()

    time, flux, error = np.array([]), np.array([]), np.array([])
    quart = 0
    for lc in lc_collection:
        print(f"Downloading light curve segment {quart + 1} of {len(lc_collection)}", end='\r')
        quart += 1

        lc_data = lc.remove_nans()
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
        error = np.append(error, tmperror / interp_savgol)

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


def run_bls_analysis(time, flux, error, min_period, max_period, duration_range=(0.01, 1)):
    bls = BoxLeastSquares(time, flux, dy=error)
    periods = np.linspace(min_period, max_period, 10000)
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

def analyze_period(period, time, flux, error, duration_range):
    try:
        print(f"Analyzing period {period:.2f} days...")

        # Ensure period is a scalar
        if isinstance(period, np.ndarray):
            period = period.item()

        # Skip periods that are too short for a valid transit duration.
        if period < duration_range[1]:
            print(f"Skipping period {period:.2f} days as it's shorter than the maximum transit duration.")
            return None

        results, _, _, best_period, best_transit_model = run_bls_analysis(
            time,
            flux,
            error,
            min_period=period * 0.9,
            max_period=period * 1.1,
            duration_range=duration_range
        )
        
        if best_period is not None:
            return {
                "candidate_period": period,
                "refined_period": best_period,
                "transit_model": best_transit_model,
                "power": max(results.power),
                "duration": results.duration[np.argmax(results.power)],
                "depth": results.depth[np.argmax(results.power)],
            }
        return None
    except Exception as e:
        return {"error": str(e), "period": period}


def analyze_peaks_with_bls(time, flux, error, peak_periods, duration_range=(0.01, 0.1)):

    with multiprocessing.Pool() as pool:
        results_list = pool.starmap(analyze_period, [(period, time, flux, error, duration_range) for period in peak_periods])
    
    # Filter out None results and handle errors
    final_results = []
    for result in results_list:
        if result is None:
            continue
        if "error" in result:
            print(f"Error analyzing period {result['period']}: {result['error']}")
        else:
            final_results.append(result)
    
    return final_results

def remove_exact_duplicates(results_list, decimal_places=3):
    unique_results = []
    unique_periods = set()
    
    for result in results_list:
        period = round(result["refined_period"], decimal_places)
        if period not in unique_periods:
            unique_periods.add(period)
            unique_results.append(result)
    
    return unique_results

def remove_duplicate_periods(results_list, decimal_places=3, percentage_threshold=0.05):
    # Remove exact duplicates
    unique_results = remove_exact_duplicates(results_list, decimal_places)
    # Sort results by refined_period
    unique_results = sorted(unique_results, key=lambda x: x["refined_period"])

    final_results = []
    final_periods = set()
    
    for result in unique_results:
        print(f"Checking period {result['refined_period']:.2f} days...")
        period = round(result["refined_period"], decimal_places)
        is_unique = True
        
        for final_period in final_periods:
            ratio = period / final_period
            rounded_ratio = round(ratio, decimal_places)
            lower_bound = (1 - percentage_threshold) * rounded_ratio
            upper_bound = (1 + percentage_threshold) * rounded_ratio
            
            if lower_bound < ratio < upper_bound:
                is_unique = False
                break
        
        if is_unique:
            print(f"Unique period found: {period:.2f} days")
            final_periods.add(period)
            final_results.append(result)

    return final_results


def plot_phase_folded_light_curves(time, flux, results_list):
    for i, result in enumerate(results_list):
        period = result["refined_period"]
        model_flux = result["transit_model"]
        duration = result["duration"]  # Assuming duration is provided in the result
        phase = ((time % period) / period - 0.5) % 1

        # Find the phase of the minimum flux in the model
        min_flux_indices = np.where(model_flux == np.min(model_flux))[0]
        min_flux_phase = np.mean(phase[min_flux_indices])

        # Adjust phase to center the transit
        phase = (phase - min_flux_phase + 0.5) % 1 - 0.5

        # Calculate the phase range based on the duration of the transit
        phase_range = duration / period

        # Filter the phase and flux to be within the calculated phase range
        mask = (phase >= -0.5) & (phase <= 0.5)
        filtered_phase = phase[mask]
        filtered_flux = flux[mask]
        filtered_model_flux = model_flux[mask]

        plt.figure(figsize=(12, 6))
        plt.scatter(filtered_phase, filtered_flux, s=10, label="Flux", color='blue', alpha=0.6)
        plt.scatter(filtered_phase, filtered_model_flux, color="orange", label=f"Planet {i+1} Transit Model")
        plt.xlabel("Phase", fontsize=14)
        plt.ylabel("Normalized Flux", fontsize=14)
        plt.title(f"Phase-Folded Light Curve for Candidate Planet {i+1}", fontsize=16)
        plt.yscale("log")

        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def plot_light_curve(time,flux,flux_error):
    plt.figure(figsize=(10, 6))
    plt.errorbar(time, flux, yerr=flux_error, fmt='o', color='red', markersize=2)
    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Flux")
    plt.title("Kepler Light Curve")
    plt.show()

def summarize_results(results_list,stellar_info):
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
                print(f"Best Transit Candidate: Period = {result['refined_period']:.2f} days, Depth = {depth:.2e}")
                print(f"Estimated Planet Radius: {planet_radius:.3f} Solar Radii")
                earth_radius_in_terms_of_stellar = 0.009168
                jupiter_radius_in_terms_of_stellar = 0.10045
                print(f"Estimated Planet Radius: {planet_radius / earth_radius_in_terms_of_stellar :.3f} Earth Radii")
                print(f"Estimated Planet Radius: {planet_radius / jupiter_radius_in_terms_of_stellar :.3f} Jupiter Radii")

# Find transit peaks (Lomb-Scargle Periodogram)
def find_transits(time, flux, resolution):
    
    preiod_range = (3, 30)
    period = np.linspace(preiod_range[0], preiod_range[1], resolution)
    frequency = np.linspace((1/45.),(1/0.007), resolution)

    # First Lomb-Scargle periodogram
    power_lomb_1 = scipy.signal.lombscargle(time, flux, frequency, precenter=True, normalize=False)

    plt.figure(figsize=(10, 6))
    plt.plot(period, power_lomb_1, label="First Lomb-Scargle Periodogram")
    plt.show()

    print('computing second periodogram')
    # Second Lomb-Scargle periodogram on the power spectrum
    power_lomb_2 = scipy.signal.lombscargle(frequency, power_lomb_1, period, precenter=True, normalize=False)
    
    return period, power_lomb_2

def run_lomb_scargle_analysis(kepler_dataframe,resolution=5000):
      print("Running Lomb-Scargle Periodogram Analysis...")
      period, lomb2 = find_transits(kepler_dataframe["time"], kepler_dataframe["flux"],resolution)
      
      # Calculate height parameter as the 99th percentile of the power values.
      height_threshold = np.percentile(lomb2, 98)
      
      # Find initial peaks to calculate the median period difference.
      initial_peaks = find_peaks(lomb2, height=height_threshold)
      initial_peak_pos = period[initial_peaks[0]]
      
      # Calculate distance parameter as a fraction of the median period difference.
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
      plt.plot(period, lomb2, label="Lomb-Scargle Periodogram")
      plt.plot(peak_pos, lomb2[peaks[0]], "x", label="Detected Peaks", color="red")
      plt.xscale("log")
      plt.yscale("log")
      plt.xlabel("Period (days)")
      plt.ylabel("Power")
      plt.title("Lomb-Scargle Periodogram")
      plt.legend()
      plt.show()

      return peak_pos

if __name__ == "__main__":
    print("Running Kepler Light Curve Analysis...")
    target = "Kepler-12"
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

    # Summarize results
    summarize_results(results_list,stellar_info)
    plot_phase_folded_light_curves(kepler_dataframe["time"].values, kepler_dataframe["flux"].values, results_list)

