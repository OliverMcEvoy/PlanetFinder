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
from scipy.signal  import medfilt
from lightkurve.lightcurve import TessLightCurve
import glob

def fetch_kepler_data_and_stellar_info(target,filter_type = 'savgol'):
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
        window_length = min(51, array_size - (array_size % 2 == 0))

        if window_length > 2:
            if filter_type == 'savgol':
                normaliser = savgol_filter(tmpflux, window_length=window_length, polyorder=3)
            elif filter_type == 'medfilt':
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
    df = df[(df["flux"] <= mean_flux + 3 * std_flux) & (df["flux"] >= mean_flux - 8 * std_flux)]

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

def loadDataFromFitsFiles(FolderPath, filter_type='savgol', randomise=False):
    '''
    Function to load data from multiple fits files in a folder
    @params
    FolderPath: str -> the path to the folder containing the fits files
    @returns
    time: array -> array containing the time values of the kepler dataframe.
    flux: array -> array containing the corresponding flux values for each time.
    error: array -> array containing the error values for each flux value.
    '''
    time, flux, error = np.array([]), np.array([]), np.array([])
    for lcfile in glob.glob(FolderPath + '/*.fits'):
        with fits.open(lcfile) as lc:
            lc_data = lc[1].data  # Assuming the data is in the first extension
            tmptime = np.array(lc_data['TIME'])
            tmpflux = np.array(lc_data['PDCSAP_FLUX'])
            tmperror = np.array(lc_data['PDCSAP_FLUX_ERR'])

            print(len(tmptime), len(tmpflux), len(tmperror))

            # Remove NaNs
            mask = ~np.isnan(tmpflux) & ~np.isnan(tmperror)
            tmptime = tmptime[mask]
            tmpflux = tmpflux[mask]
            tmperror = tmperror[mask]


            if randomise:
                np.random.shuffle(tmptime)

            time = np.append(time, tmptime)
            flux = np.append(flux, tmpflux)
            error = np.append(error, tmperror)
    
    array_size = len(flux)
    window_length = min(51, array_size - (array_size % 2 == 0))


    if filter_type == 'savgol':
        interp_savgol = savgol_filter(flux, window_length=window_length, polyorder=3)
    elif filter_type == 'medfilt':
        flux = flux.astype(np.float64) 
        interp_savgol = medfilt(flux, kernel_size=51)

    flux = flux / interp_savgol
    error = error / interp_savgol


    # Store the data in a dataframe.
    df = pd.DataFrame({"time": time, "flux": flux, "error": error})
    mean_flux = np.mean(flux)
    std_flux = np.std(flux)
    df = df[(df["flux"] <= mean_flux + 2 * std_flux) & (df["flux"] >= mean_flux - 8 * std_flux)]
    
    return df 


def fetch_kepler_data_and_stellar_info_normalise_entire_curve(target, filter_type = 'savgol',randomise = False):
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


        if randomise:
            np.random.shuffle(tmpflux)

        time = np.append(time, lc_data.time.value)
        flux = np.append(flux, lc_data.flux.value)
        error = np.append(error, lc_data.flux_err.value)

    array_size = len(flux)
    window_length = min(51, array_size - (array_size % 2 == 0))

    if filter_type == 'savgol':
        interp_savgol = savgol_filter(flux, window_length=window_length, polyorder=3)
    elif filter_type == 'medfilt':
        flux = flux.astype(np.float64) 
        interp_savgol = medfilt(flux, kernel_size=51)

    flux = flux/interp_savgol
    error = error/interp_savgol


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


def run_bls_analysis(time, flux, error, resolution,min_period, max_period, duration_range=(0.01, 1)):
    bls = BoxLeastSquares(time, flux, dy=error)
    periods = np.linspace(min_period, max_period, resolution)
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

    # Check if the power is significant enough to indicate a planet
    if results.power[max_power_idx] < 0.1:  # Threshold can be adjusted based on requirements
        print("No significant transit signal detected.")
        return None, None, None, None, None

    # Generate the best transit model
    best_transit_model = bls.model(time, best_period, best_duration, best_transit_time)
    return results, results.power, results.period, best_period, best_transit_model

def estimate_planet_radius(transit_depth, stellar_radius):
    planet_radius = np.sqrt(transit_depth) * stellar_radius 
    return planet_radius

def analyze_period(period, time, flux, error, resolution,duration_range):
    try:
        print(f"Analyzing period {period:.2f} days...")

        # Ensure period is a scalar
        if isinstance(period, np.ndarray):
            period = period.item()

        # Skip periods that are too short for a valid transit duration.
        if period < duration_range[1]:
            print(f"Skipping period {period:.2f} days as it's shorter than the maximum transit duration.")
            return None

        results, power, periods, best_period, best_transit_model = run_bls_analysis(
            time,
            flux,
            error,
            resolution,
            min_period=period * 0.9,
            max_period=period * 1.1,
            duration_range=duration_range
        )
        
        if best_period is not None:
            return {
                "candidate_period": period,
                "refined_period": best_period,
                "transit_model": best_transit_model,
                "power": max(power),
                "duration": results.duration[np.argmax(power)],
                "depth": results.depth[np.argmax(power)],
            }
        else:
            print(f"No valid period found for {period:.2f} days.")
            return None
    except Exception as e:
        return {"error": str(e), "period": period}


def analyze_peaks_with_bls(time, flux, error, peak_periods, resolution=10000,duration_range=(0.01, 0.25)):

    with multiprocessing.Pool() as pool:
        results_list = pool.starmap(analyze_period, [(period, time, flux, error,resolution, duration_range,) for period in peak_periods])
    
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

def remove_exact_duplicates(results_list, duplicates_percentage_threshold=0.05, complex_result_list = True):
    unique_results = []
    unique_periods = set()
    
    for result in results_list:
        
        if complex_result_list:
            period = result["refined_period"]
        else:
            period = result
            
        is_unique = True
        for unique_period in unique_periods:
            lower_bound = (1 - duplicates_percentage_threshold) * unique_period
            upper_bound = (1 + duplicates_percentage_threshold) * unique_period


            if lower_bound < period < upper_bound:
                is_unique = False
                break
        if is_unique:
            print(f"Adding period {period:.3f} days to the unique results.")
            unique_periods.add(period)
            unique_results.append(result)
    
    return unique_results

def remove_duplicate_periods(results_list, duplicates_percentage_threshold=0.05, percentage_threshold=0.05, power_threshold=0.1):
    # Remove exact duplicates
    unique_results = remove_exact_duplicates(results_list, duplicates_percentage_threshold)
    # Sort results by refined_period
    unique_results = sorted(unique_results, key=lambda x: x["refined_period"])

    final_results = []
    final_periods = set()
    final_powers = []

    for result in unique_results:
        period = result["refined_period"]
        power = result["power"]
        is_unique = True
        
        for i, final_period in enumerate(final_periods):
            ratio = period / final_period
            lower_bound = (1 - percentage_threshold) * ratio
            upper_bound = (1 + percentage_threshold) * ratio

            if lower_bound < ratio < upper_bound:
                final_power = final_powers[i]
                power_ratio = power / final_power
                power_lower_bound = 1 - power_threshold
                power_upper_bound = 1 + power_threshold

                if power_lower_bound < power_ratio < power_upper_bound:
                    is_unique = False
                    break

                for j in range(i + 1, len(final_powers)):
                    combined_power = final_power + final_powers[j]
                    combined_power_ratio = power / combined_power
                    if power_lower_bound < combined_power_ratio < power_upper_bound:
                        is_unique = False
                        break
                if not is_unique:
                    break
        
        if is_unique:
            print(f"Adding period {period:.3f} days to the final results.")
            final_periods.add(period)
            final_powers.append(power)
            final_results.append(result)

    return final_results


def plot_phase_folded_light_curves(kepler_dataframe, results_list):
    time, flux, error = kepler_dataframe["time"].values, kepler_dataframe["flux"].values, kepler_dataframe["error"].values
    for i, result in enumerate(results_list):
        period = result["refined_period"]
        model_flux = result["transit_model"]
        duration = result["duration"]
        phase = ((time % period) / period - 0.5) % 1

        # Find the phase of the minimum flux in the model
        min_flux_indices = np.where(model_flux == np.min(model_flux))[0]
        min_flux_phase = np.mean(phase[min_flux_indices])

        # Adjust phase to center the transit
        phase = (phase - min_flux_phase + 0.5) % 1 - 0.5

        # Calculate the phase range based on the duration of the transit
        phase_range_for_plot = duration / period

        # Filter the phase and flux to be within the calculated phase range
        mask = (phase >= -phase_range_for_plot) & (phase <= phase_range_for_plot)
        filtered_phase = phase[mask]
        filtered_flux = flux[mask]
        filtered_error = error[mask]
        filtered_model_flux = model_flux[mask]

        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Plot unfiltered data
        axs[0].errorbar(phase, flux, yerr=error, fmt='o', color='black', alpha=0.5, label="Unfiltered Flux", linestyle='none')
        axs[0].errorbar(phase, model_flux, fmt='s', color='red', alpha=0.5, label="Unfiltered Transit Model", linestyle='none')
        axs[0].set_xlabel("Phase", fontsize=14)
        axs[0].set_ylabel("Normalized Flux", fontsize=14)
        axs[0].set_title(f"Unfiltered Phase-Folded Light Curve for Candidate Planet {i+1}", fontsize=16)
        axs[0].legend(fontsize=12)
        axs[0].grid(True) #tmp to assist with debugging

        # Plot filtered data
        axs[1].errorbar(filtered_phase, filtered_flux, yerr=filtered_error, fmt='o', color='black', alpha=0.5, label="Filtered Flux", linestyle='none')
        axs[1].errorbar(filtered_phase, filtered_model_flux, fmt='s', color='red', alpha=0.5, label="Filtered Transit Model", linestyle='none', markersize=2)
        axs[1].set_xlabel("Phase", fontsize=14)
        axs[1].set_ylabel("Normalized Flux", fontsize=14)
        axs[1].set_title(f"Filtered Phase-Folded Light Curve for Candidate Planet {i+1}", fontsize=16)
        axs[1].legend(fontsize=12)
        axs[1].grid(True) #tmp to assist with debugging

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

def compute_lombscargle(args):
    time, flux, frequency_chunk = args
    return scipy.signal.lombscargle(time, flux, frequency_chunk, precenter=True, normalize=False)

# Find transit peaks (Lomb-Scargle Periodogram)
def find_transits(time, flux, resolution,period_range, list_of_random_lightcurves ):
    '''
    @params
    time: array -> array containing the time values of the kepler dataframe.
    flux: array -> array containing the corrosponding flux values for each time.
    resolution: int -> the number of points to use in the periodogram.

    find the transit peaks, the expected inputs 
    '''
    period = np.linspace(period_range[0], period_range[1], resolution)

    frequency_range = (time[1]-time[0], time[len(time)] -time[0])
    frequency = np.linspace((1/frequency_range[1]), (1/frequency_range[0]), resolution)

    # Split frequency array into chunks for parallel processing
    num_chunks = 12  # Number of processes
    frequency_chunks = np.array_split(frequency, num_chunks)
    period_chunks = np.array_split(period, num_chunks)

    # Use multiprocessing to compute the first Lomb-Scargle periodogram
    with multiprocessing.Pool() as pool:
        power_lomb_1_chunks = pool.map(compute_lombscargle, [(time, flux, chunk) for chunk in frequency_chunks])
    
    # Combine the results from each chunk
    power_lomb_1 = np.concatenate(power_lomb_1_chunks)

    plt.figure(figsize=(12, 6))
    plt.title("First Lomb-Scargle Periodogram")
    plt.plot(period, power_lomb_1, label="First Lomb-Scargle Periodogram")
    plt.show()

    print('computing second periodogram')
    # Second Lomb-Scargle periodogram on the power spectrum
    with multiprocessing.Pool() as pool:
        power_lomb_2_chunks = pool.map(compute_lombscargle, [(frequency, power_lomb_1, chunk) for chunk in period_chunks])

    power_lomb_2_regular = np.concatenate(power_lomb_2_chunks)

    plt.figure(figsize=(12, 6))
    plt.title("Second Lomb-Scargle Periodogram")
    plt.plot(period, power_lomb_2_regular, label="Second Lomb-Scargle Periodogram")

    if list_of_random_lightcurves:

        list_of_random_lightcurves_lombed = []
        list_of_difference_between_rand_and_regular = []
        power_lomb_difference_squared = np.zeros_like(power_lomb_2_regular)

        print('List of random light curves present, computing random light curves')
        i =0

        for lightcurve in list_of_random_lightcurves:
            
            print(f"Computing random light curve {i} of {len(list_of_random_lightcurves)}", end='\r')

            time, flux = lightcurve['time'], lightcurve['flux']
            with multiprocessing.Pool() as pool:
                power_lomb_1_chunks = pool.map(compute_lombscargle, [(time, flux, chunk) for chunk in frequency_chunks])
            power_lomb_1 = np.concatenate(power_lomb_1_chunks)
            with multiprocessing.Pool() as pool:
                power_lomb_2_chunks = pool.map(compute_lombscargle, [(frequency, power_lomb_1, chunk) for chunk in period_chunks])
            power_lomb_2 = np.concatenate(power_lomb_2_chunks)
            list_of_random_lightcurves_lombed.append(power_lomb_2)
            difference_squared = (power_lomb_2_regular-power_lomb_2)**2
            list_of_difference_between_rand_and_regular.append(difference_squared)
            power_lomb_difference_squared += difference_squared
            i+=1

        list_of_random_lightcurves_lombed_averaged = np.mean(list_of_random_lightcurves_lombed, axis=0)

        plt.figure(figsize=(12, 6))
        plt.title("Random Light Curves Lomb-Scargle Periodogram difference squared")
        plt.plot(period, power_lomb_difference_squared, label="Random Light Curves")

        plt.figure(figsize=(12, 6))
        plt.title("Random Light Curves Lomb-Scargle Periodogram")
        plt.plot(period, list_of_random_lightcurves_lombed_averaged, label="Random Light Curves")
        plt.show()

        power_lomb_2 = power_lomb_2_regular - list_of_random_lightcurves_lombed_averaged

        plt.figure(figsize=(12, 6))
        plt.title("Random Light Curves Subtracted Lomb-Scargle Periodogram")
        plt.plot(period, power_lomb_2, label="Random Light Curves Subtracted")
        plt.show()

    else:
        power_lomb_2 = power_lomb_2_regular


    return period, power_lomb_2

def find_transits_adjusted(time, flux, resolution, period_range, list_of_random_lightcurves):
    '''
    @params
    time: array -> array containing the time values of the kepler dataframe.
    flux: array -> array containing the corresponding flux values for each time.
    resolution: int -> the number of points to use in the periodogram.

    find the transit peaks, the expected inputs 
    '''
    period = np.linspace(period_range[0], period_range[1], resolution)

    frequency_range = (time[1] - time[0], time[len(time)] - time[0])
    frequency = np.linspace((1 / frequency_range[1]), (1 / frequency_range[0]), resolution)

    # Split frequency array into chunks for parallel processing
    num_chunks = 12  # Number of processes
    frequency_chunks = np.array_split(frequency, num_chunks)
    period_chunks = np.array_split(period, num_chunks)

    # Use multiprocessing to compute the first Lomb-Scargle periodogram
    with multiprocessing.Pool() as pool:
        power_lomb_1_chunks = pool.map(compute_lombscargle, [(time, flux, chunk) for chunk in frequency_chunks])

    # Combine the results from each chunk
    power_lomb_1_regular = np.concatenate(power_lomb_1_chunks)

    if list_of_random_lightcurves:

        list_of_random_lightcurves_lombed = []
        list_of_difference_between_rand_and_regular = []
        power_lomb_difference_squared = np.zeros_like(power_lomb_1_regular)

        print('List of random light curves present, computing random light curves')
        i = 0

        for lightcurve in list_of_random_lightcurves:

            print(f"Computing random light curve {i} of {len(list_of_random_lightcurves)}", end='\r')

            time, flux = lightcurve['time'], lightcurve['flux']
            with multiprocessing.Pool() as pool:
                power_lomb_1_chunks = pool.map(compute_lombscargle, [(time, flux, chunk) for chunk in frequency_chunks])
            power_lomb_1 = np.concatenate(power_lomb_1_chunks)
            list_of_random_lightcurves_lombed.append(power_lomb_1)
            difference_squared = (power_lomb_1_regular - power_lomb_1) ** 2
            list_of_difference_between_rand_and_regular.append(difference_squared)
            power_lomb_difference_squared += difference_squared
            i += 1

        list_of_random_lightcurves_lombed_averaged = np.mean(list_of_random_lightcurves_lombed, axis=0)

        plt.figure(figsize=(12, 6))
        plt.title("Random Light Curves Lomb-Scargle Periodogram difference squared")
        plt.plot(period, power_lomb_difference_squared, label="Random Light Curves")

        plt.figure(figsize=(12, 6))
        plt.title("Random Light Curves Lomb-Scargle Periodogram")
        plt.plot(period, list_of_random_lightcurves_lombed_averaged, label="Random Light Curves")
        plt.show()

        power_lomb_1 = power_lomb_1_regular - list_of_random_lightcurves_lombed_averaged

        plt.figure(figsize=(12, 6))
        plt.title("Random Light Curves Subtracted Lomb-Scargle Periodogram")
        plt.plot(period, power_lomb_1, label="Random Light Curves Subtracted")
        plt.show()

    else:
        power_lomb_1 = power_lomb_1_regular
        power_lomb_difference_squared = None

    plt.figure(figsize=(12, 6))
    plt.title("First Lomb-Scargle Periodogram")
    plt.plot(period, power_lomb_1, label="First Lomb-Scargle Periodogram")
    plt.show()

    print('computing second periodogram')
    # Second Lomb-Scargle periodogram on the power spectrum
    with multiprocessing.Pool() as pool:
        power_lomb_2_chunks = pool.map(compute_lombscargle, [(frequency, power_lomb_1, chunk) for chunk in period_chunks])

    power_lomb_2_regular = np.concatenate(power_lomb_2_chunks)

    plt.figure(figsize=(12, 6))
    plt.title("Second Lomb-Scargle Periodogram")
    plt.plot(period, power_lomb_2_regular, label="Second Lomb-Scargle Periodogram")

    if power_lomb_difference_squared is not None:
        print('computing second periodogram for difference squared')
        with multiprocessing.Pool() as pool:
            power_lomb_2_diff_chunks = pool.map(compute_lombscargle, [(frequency, power_lomb_difference_squared, chunk) for chunk in period_chunks])

        power_lomb_2_difference_squared = np.concatenate(power_lomb_2_diff_chunks)

        plt.figure(figsize=(12, 6))
        plt.title("Second Lomb-Scargle Periodogram (Difference Squared)")
        plt.plot(period, power_lomb_2_difference_squared, label="Second Lomb-Scargle Periodogram (Difference Squared)")
        plt.show()

    return period, power_lomb_2_regular

def run_lomb_scargle_analysis(kepler_dataframe, resolution=5000,period_range=(1, 30),list_of_random_lightcurves = False,different = False, power_filter = False):
    print("Running Lomb-Scargle Periodogram Analysis...")

    # Compute the Lomb-Scargle periodogram
    if not different:
        period, lomb2 = find_transits(kepler_dataframe["time"], kepler_dataframe["flux"], resolution,period_range,list_of_random_lightcurves)
    else:
        period, lomb2 =find_transits_adjusted(kepler_dataframe["time"], kepler_dataframe["flux"], resolution,period_range,list_of_random_lightcurves)
    
    # Compute the gradient and the second derivative (gradient of the gradient)
    gradient = np.gradient(lomb2, period)
    second_derivative = np.gradient(gradient, period)
    
    # Calculate rolling mean and standard deviation of the gradient
    window_size = 50  # Adjust window size as needed
    rolling_std = pd.Series(gradient).rolling(window=window_size).std().fillna(0)
    
    # Find the point where the gradient stabilizes
    stabilization_index = np.argmax(rolling_std < np.mean(rolling_std))
    gradient_threshold = np.abs(gradient[stabilization_index])
    
    # Determine the second derivative threshold algorithmically
    second_derivative_threshold = np.mean(np.abs(second_derivative)) + 2 * np.std(np.abs(second_derivative))
    
    print(f"Gradient Threshold: {gradient_threshold:.2e}, Second Derivative Threshold: {second_derivative_threshold:.2e}")

    # Plot the gradient
    plt.figure(figsize=(10, 6))
    plt.plot(period, gradient, label="Gradient of Power")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(gradient_threshold, color='red', linestyle='--', label='Gradient Threshold')
    plt.xlabel("Period (days)")
    plt.ylabel("Gradient")
    plt.title("Gradient of Lomb-Scargle Power vs Period")
    plt.legend()
    plt.show()

    # Plot the second derivative
    plt.figure(figsize=(10, 6))
    plt.plot(period, second_derivative, label="Second Derivative of Power")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel("Period (days)")
    plt.ylabel("Second Derivative")
    plt.title("Second Derivative of Lomb-Scargle Power vs Period")
    plt.legend()
    plt.show()
    
    # Determine regions where both gradient and second derivative are small
    smooth_region_indices = np.where(
        (np.abs(gradient) < gradient_threshold) &
        (np.abs(second_derivative) < second_derivative_threshold)
    )[0]

    # Ensure there are multiple consecutive points in the smooth region
    if len(smooth_region_indices) > 1:
        smooth_start_index = smooth_region_indices[0]  # Start of smooth region
        period_threshold = period[smooth_start_index]
    else:
        period_threshold = np.min(period)  # Fallback if no smooth region is found

    print(f"Excluding peaks before period = {period_threshold:.2f} days")
    
    # Determine peak detection parameters algorithmically
    height = np.median(lomb2) + 0 * np.std(lomb2)
    distance = resolution // 1000
    prominence = np.median(lomb2) + 0 * np.std(lomb2)
    
    # Find initial peaks using scipy's find_peaks with algorithmically determined parameters
    peaks, _ = find_peaks(lomb2,height=height, distance=distance, prominence=prominence)
    peak_pos = period[peaks]
    peak_powers = lomb2[peaks]
    
    # Exclude peaks in the low-period region based on the threshold
    if(power_filter):
        valid_peaks = (peak_pos >= period_threshold) & (peak_powers > power_filter) 
    else:
        valid_peaks = peak_pos >= period_threshold
    peak_pos = peak_pos[valid_peaks]
    peak_powers = peak_powers[valid_peaks]
    
    print("Lomb-Scargle Periodogram analysis done")

    # Visualize the Lomb-Scargle periodogram and detected peaks
    plt.figure(figsize=(10, 6))
    plt.plot(period, lomb2, label="Lomb-Scargle Periodogram")
    plt.plot(peak_pos, peak_powers, "x", label="Detected Peaks", color="red")
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

