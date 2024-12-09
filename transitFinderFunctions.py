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
from scipy.interpolate import interp1d
import PlanetGenerationExtreme as pg
from scipy.optimize import differential_evolution
from tqdm import tqdm
import importlib
import scipy.optimize as opt
importlib.reload(pg)
import glob
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution, minimize, basinhopping


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

def getRandomisedData(path,interations, filter_type):

    list_of_random_lightcurves = []
    for i in range (interations):
        list_of_random_lightcurves.append(loadDataFromFitsFiles(path, filter_type=filter_type, randomise=True))

    return list_of_random_lightcurves    

def loadDataFromFitsFiles(path, filter_type='savgol', randomise=False):
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
    for lcfile in glob.glob(path + '/*.fits'):
        with fits.open(lcfile) as lc:
            lc_data = lc[1].data  # Assuming the data is in the first extension
            tmptime = np.array(lc_data['TIME'])
            tmpflux = np.array(lc_data['PDCSAP_FLUX'])
            tmperror = np.array(lc_data['PDCSAP_FLUX_ERR'])

            mask = ~np.isnan(tmpflux) | ~np.isnan(tmperror)
            tmptime = tmptime[mask]
            tmpflux = tmpflux[mask]
            tmperror = tmperror[mask]

            if randomise:
                np.random.shuffle(tmptime)
                print(tmptime)

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

def run_bls_analysis(time, flux, error, resolution, min_period, max_period, duration_range=(0.01, 0.25)):
    """
    Run the Box Least Squares analysis to detect transits.
    """
    bls = BoxLeastSquares(time, flux, dy=error)
    periods = np.linspace(min_period, max_period, resolution)
    durations = np.linspace(duration_range[0], duration_range[1], 200)
    results = bls.power(periods, durations)

    # Find the best period based on maximum power
    max_power_idx = np.argmax(results.power)
    best_period = results.period[max_power_idx]
    best_duration = results.duration[max_power_idx]
    best_transit_time = results.transit_time[max_power_idx]

    # Debugging: Print intermediate results to verify values
    print(f"Best Period: {best_period}, Best Duration: {best_duration}, Best Transit Time: {best_transit_time}, Max Power: {results.power[max_power_idx]}")

    # Validate inputs before calling `bls.model()`
    if not (np.isfinite(best_period) and np.isfinite(best_duration) and np.isfinite(best_transit_time)):
        raise ValueError("Invalid input detected for BoxLeastSquares.model(). Ensure periods, durations, and transit times are finite.")

    # Generate the best transit model
    best_transit_model = bls.model(time, best_period, best_duration, best_transit_time)

    # Calculate transit depth for validation
    in_transit = flux[best_transit_model < 1]
    out_of_transit = flux[best_transit_model >= 1]
    calculated_depth = 1 - np.mean(in_transit) / np.mean(out_of_transit)
    print(f"Calculated Transit Depth: {calculated_depth:.6f}")

    return results, results.power, results.period, best_period, best_transit_model

def estimate_planet_radius(transit_depth, stellar_radius):
    """
    Estimate the radius of a planet based on the transit depth and stellar radius.
    """
    if transit_depth > 1:
        raise ValueError("Transit depth must be a fractional value (e.g., 0.01 for 1%).")
    planet_radius = np.sqrt(transit_depth) * stellar_radius
    return planet_radius

def analyze_period(period, time, flux, error, resolution, duration_range, allowed_deviation):
    """
    Analyze a specific period to refine its transit model and parameters.
    """
    try:
        # Ensure period is a scalar
        if isinstance(period, np.ndarray):
            period = period.item()

        # Skip periods that are too short for a valid transit duration
        if period < duration_range[1]:
            print(f"Skipping period {period:.2f} days as it's shorter than the maximum transit duration.")
            return None

        results, power, periods, best_period, best_transit_model = run_bls_analysis(
            time,
            flux,
            error,
            resolution,
            min_period=period * (1 - allowed_deviation),
            max_period=period * (1 + allowed_deviation),
            duration_range=duration_range
        )

        # Filter out shallow transits
        depth = results.depth[np.argmax(power)]
        if depth < 0.0001:
            print(f"Skipping shallow transit with depth {depth:.6f} at period {period:.2f} days.")
            return None
        
        print(f"transit depth: {depth}")

        return {
            "candidate_period": period,
            "refined_period": best_period,
            "transit_model": best_transit_model,
            "power": max(power),
            "duration": results.duration[np.argmax(power)],
            "depth": depth,
        }
    except Exception as e:
        return {"error": str(e), "period": period}

def analyze_peaks_with_bls(kepler_dataframe, peak_periods, resolution=10000, duration_range=(0.01, 0.5), allowed_deviation=0.05):
    """
    Analyze multiple period candidates using the BLS algorithm.
    """
    time = kepler_dataframe["time"].values
    flux = kepler_dataframe["flux"].values
    error = kepler_dataframe["error"].values

    with multiprocessing.Pool() as pool:
        results_list = list(
            tqdm(
                pool.starmap(
                    analyze_period, 
                    [(period, time, flux, error, resolution, duration_range, allowed_deviation) for period in peak_periods]
                ),
                total=len(peak_periods),
                desc="Analyzing Periods"
            )
        )
    
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
            unique_periods.add(period)
            unique_results.append(result)
    
    return unique_results

def remove_duplicate_periods(results_list, duplicates_percentage_threshold=0.05, repeat_transit_threshold=0.05, power_threhsold_for_repeat_periods=0.1):
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
            lower_bound = (1 - repeat_transit_threshold) * ratio
            upper_bound = (1 + repeat_transit_threshold) * ratio

            if lower_bound < ratio < upper_bound:
                final_power = final_powers[i]
                power_ratio = power / final_power
                power_lower_bound = 1 - power_threhsold_for_repeat_periods
                power_upper_bound = 1 + power_threhsold_for_repeat_periods

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


def calculate_fit_for_period(result, time, flux, error, total_time, star_radius, cadence, method='minimize'):
    period = result["refined_period"]
    bls_model_flux = result["transit_model"]
    transit_duration = result["duration"]

    filtered_phase = phase_fold(time, period,bls_model_flux=bls_model_flux)
    min_flux_phase = filtered_phase[np.argmin(bls_model_flux)]

    planet_radius = estimate_planet_radius(result["depth"], star_radius)

    differential_options = {
        'maxiter': 15,
        'popsize': 10,
        'disp': False,
        'tol': 0.000001,
    }

    minimize_options = {
        'maxiter': 100,
        'disp': True,
        'ftol': 0.0000001,

    }

    #can be done as M of sun same as our own
    guess_for_sma = (period/365)**(2/3)

    planet_radius = planet_radius
    bounds = [(guess_for_sma * 0.25, guess_for_sma*2), (0.7, 1), (0.2, 0.6), (planet_radius*0.25, planet_radius*2)]
    initial_guess = [guess_for_sma , 0.85, 0.25, planet_radius]

    print(f"initial guess: {initial_guess}")

    if method == 'minimize':
        result = minimize(lad, initial_guess, args=(period, total_time, filtered_phase, star_radius, flux, cadence, error, transit_duration), 
                          method='SLSQP', bounds=bounds, options=minimize_options)
        best_fit_params = result.x
        print('done minimising')
    elif method == 'differential_evolution':
        result = differential_evolution(lad, bounds, args=(period, total_time, filtered_phase, star_radius, flux, cadence, error, transit_duration), **differential_options)
        best_fit_params = result.x
        print('done differential evolution')
    elif method == 'Nelder-Mead':
        result = minimize(lad, initial_guess, args=(period, total_time, filtered_phase, star_radius, flux, cadence, error, transit_duration), 
                          method='Nelder-Mead', bounds=bounds, options=minimize_options)
        best_fit_params = result.x
    else:
        raise ValueError(f"Invalid optimization method: {method}")

    best_fit_a = best_fit_params[0]
    best_fit_u1 = best_fit_params[1]
    best_fit_u2 = best_fit_params[2]
    best_fit_radius = best_fit_params[3]

    planets = [
        {
            'period': period,
            'rp': best_fit_radius,
            'a': best_fit_a,
            'incl': np.pi / 2,
            'transit_midpoint': period/2
        }
    ]
    pg_time, best_fit_model_lightcurve, _ = pg.generate_multi_planet_light_curve(planets, total_time, star_radius, 0, snr_threshold=5, u1=best_fit_u1, u2=best_fit_u2, cadence=cadence, simulate_gap_in_data=False)

    #best_fit_model_lightcurve = best_fit_model_lightcurve / np.median(best_fit_model_lightcurve)

    pg_generated_phase = phase_fold(pg_time, period, best_fit_model_lightcurve)
    pg_nodel_lightcurve_projected_onto_kepler_phase = interoplate_phase_folded_light_curve(filtered_phase, pg_generated_phase, best_fit_model_lightcurve)
    #best_fit_model_lightcurve = best_fit_model_lightcurve / medfilt(best_fit_model_lightcurve, kernel_size=51)

    final_chi2 = lad(best_fit_params, period, total_time, filtered_phase, star_radius, flux, cadence, error, transit_duration)

    print(f"Final chi2: {final_chi2}")
    print(f"Best fit parameters: {best_fit_params}")

    return {
        'filtered_phase': filtered_phase,
        'flux': flux,
        'bls_model_flux': bls_model_flux,
        'pg_model_phase': filtered_phase,
        'best_fit_model_lightcurve': pg_nodel_lightcurve_projected_onto_kepler_phase,
        'final_chi2': final_chi2,
        'best_fit_params': best_fit_params,
        'planet_radius': planet_radius,
        'method': method,
        'transit_duration': transit_duration,
        'period': period
    }


def calculate_best_fit_parameters(kepler_dataframe, results_list):
    time, flux, error = kepler_dataframe["time"].values, kepler_dataframe["flux"].values, kepler_dataframe["error"].values
    star_radius = 1
    cadence = 0.02
    total_time = time[-1]

    #methods = ['minimize', 'differential_evolution', 'Nelder-Mead']
    methods = ['minimize','Nelder-Mead']
    all_results = []

    with multiprocessing.Pool() as pool:
        for method in methods:
            results = pool.starmap(calculate_fit_for_period, [(result, time, flux, error, total_time, star_radius, cadence, method) for result in results_list])
            all_results.append(results)

    return all_results

def plot_phase_folded_light_curves(all_results):
    num_candidates = len(all_results[0])
    fig, axs = plt.subplots(num_candidates, 4, figsize=(20, 6 * num_candidates))

    #methods = ['minimize', 'differential_evolution', 'Nelder-Mead']
    methods = ['minimize','Nelder-Mead']

    if num_candidates == 1:
        axs = [axs]

    for j in range(num_candidates):
        # Left plot: Filtered Flux and BLS Model Flux (first method)
        result = all_results[0][j]
        window_size = result['transit_duration'] * 2  / result['period']
        mask = (result['filtered_phase'] >= 0.5 - window_size) & (result['filtered_phase'] <= 0.5 + window_size)
        axs[j][0].errorbar(result['filtered_phase'][mask], result['flux'][mask], fmt='o', color='black', alpha=0.5, label="Filtered Flux", linestyle='none')
        axs[j][0].errorbar(result['filtered_phase'][mask], result['bls_model_flux'][mask], fmt='o', color='green', alpha=0.5, label="Filtered Transit Model", linestyle='none')
        axs[j][0].set_title(f"Filtered Phase-Folded Light Curve for Candidate Planet {j + 1}", fontsize=16)
        axs[j][0].set_ylabel('Normalized Flux')
        axs[j][0].legend()

        # Right plots: Best Fit Model Light Curve for each method
        for i, method_results in enumerate(all_results):
            print(f"Plotting best fit model for method {methods[i]}")
            result = method_results[j]
            axs[j][i+1].errorbar(result['filtered_phase'][mask], result['flux'][mask], fmt='o', color='black', alpha=0.5, label="Filtered Flux", linestyle='none')
            axs[j][i+1].errorbar(result['filtered_phase'][mask], result['bls_model_flux'][mask], fmt='o', color='green', alpha=0.5, label="Filtered Transit Model", linestyle='none')
            axs[j][i+1].set_title(f"Filtered Phase-Folded Light Curve for Candidate Planet {j + 1}", fontsize=16)
            axs[j][i+1].set_ylabel('Normalized Flux')
            axs[j][i+1].legend()
            axs[j][i + 1].errorbar(result['filtered_phase'][mask], result['best_fit_model_lightcurve'][mask], fmt='o', color='blue', alpha=0.5, label=f"Best Fit Model ({methods[i]})", linestyle='none')
            axs[j][i + 1].set_title(f"Best Fit Model Light Curve for Candidate Planet {j + 1} ({methods[i]})", fontsize=16)
            axs[j][i + 1].legend()

    fig.text(0.5, 0.04, 'Phase', ha='center', fontsize=14)
    plt.tight_layout()
    plt.show()

def phase_fold(time, period, flux=None, bls_model_flux=None):
    phase = (time % period) / period
    if bls_model_flux is not None:
        # Identify indices where the flux is at the lower level (in transit)
        in_transit = bls_model_flux < np.max(bls_model_flux)
        # Find the times corresponding to the transit
        transit_times = time[in_transit]
        # Calculate the midpoint of the transit times
        if len(transit_times) > 0:
            transit_midpoint = np.mean(transit_times % period) / period
            phase = (phase - transit_midpoint + 0.5) % 1
    elif flux is not None:
        min_flux_phase = (time[np.argmin(flux)] % period) / period
        phase = (phase - min_flux_phase + 0.5) % 1
    else:
        phase = (phase + 0.5) % 1  # Shift phase to center the transit
    return phase
def get_filtered_phase_flux(time, flux, model_flux, period, duration, error=None):
    phase = ((time % period) / period - 0.5) % 1
    min_flux_indices = np.where(model_flux == np.min(model_flux))[0]
    min_flux_phase = np.mean(phase[min_flux_indices])
    
    phase = phase_fold(time, period, min_flux_phase)
    
    phase_range_for_plot = duration / period
    mask = (phase >= -phase_range_for_plot) & (phase <= phase_range_for_plot)
    filtered_phase = phase[mask]
    filtered_flux = flux[mask]
    filtered_model_flux = model_flux[mask]

    if error is None:
        return filtered_phase, filtered_flux, filtered_model_flux

    filtered_error = error[mask]
    return filtered_phase, filtered_flux, filtered_error, filtered_model_flux

def lad(params, period, total_time, kepler_phase, star_radius, flux, cadence, error, transit_duration):
    planets = [
        {
            'period': period,
            'rp': params[3],
            'a': params[0],
            'incl': np.pi / 2,
            'transit_midpoint': period/2
        }
    ]
    u1 = params[1]
    u2 = params[2]
    observation_noise = 0

    time, pg_model_lightcurve, _ = pg.generate_multi_planet_light_curve(planets, total_time, star_radius, observation_noise, snr_threshold=0, u1=u1, u2=u2, cadence=cadence, simulate_gap_in_data=False)
    #pg_model_lightcurve = pg_model_lightcurve / np.median(pg_model_lightcurve)

    pg_generated_phase = phase_fold(time, period,pg_model_lightcurve)
    pg_nodel_lightcurve_projected_onto_kepler_phase = interoplate_phase_folded_light_curve(kepler_phase, pg_generated_phase, pg_model_lightcurve)

    transit_midpoint = period/2
    window_size = transit_duration * 2 / period
    
    window_mask = (kepler_phase >= (0.5 - window_size)) & (kepler_phase <= (0.5 + window_size))

    # plt.plot(kepler_phase, flux)
    # plt.plot(kepler_phase, pg_nodel_lightcurve_projected_onto_kepler_phase)
    # plt.show()

    #time, full_pg_flux = interpolate_lightcurve(phase, pg_model_lightcurve, flux[window_mask], total_time)

    lad_value = np.sum(((flux[window_mask] - pg_nodel_lightcurve_projected_onto_kepler_phase[window_mask])**2)/error[window_mask])
    return lad_value

def interoplate_phase_folded_light_curve(kepler_phase, pg_generated_phase, pg_model_lightcurve):
    interpolated_values = np.zeros_like(kepler_phase)
    
    for i, kp in enumerate(kepler_phase):
        closest_index = np.argmin(np.abs(pg_generated_phase - kp))
        interpolated_values[i] = pg_model_lightcurve[closest_index]
    
    return interpolated_values

def interpolate_lightcurve(time, pg_model_lightcurve, bls_model_flux, total_time):
    # Create a time array matching bls_model_flux length
    length_of_bls_model_flux = len(bls_model_flux)
    time_array = np.linspace(0, total_time, length_of_bls_model_flux)
    full_pg_flux = np.ones(length_of_bls_model_flux, dtype=np.float32)

    # Interpolate pg_model_lightcurve onto time_array
    interpolation_function = interp1d(
        time, pg_model_lightcurve, kind='linear', bounds_error=False, fill_value=1.0
    )
    full_pg_flux = interpolation_function(time_array)

    return time_array, full_pg_flux
def plot_light_curve(time,flux,flux_error=None):
    plt.figure(figsize=(10, 6))

    if flux_error is not None:
        plt.errorbar(time, flux, yerr=flux_error, fmt='o', color='red', markersize=2)

    else:
        plt.plot(time, flux, color='red')
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

    # Second Lomb-Scargle periodogram on the power spectrum
    with multiprocessing.Pool() as pool:
        power_lomb_2_chunks = pool.map(compute_lombscargle, [(frequency, power_lomb_1, chunk) for chunk in period_chunks])

    power_lomb_2_regular = np.concatenate(power_lomb_2_chunks)


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


        power_lomb_2 = power_lomb_2_regular - list_of_random_lightcurves_lombed_averaged

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
        list_of_difference= []

        print('List of random light curves present, computing random light curves')
        i = 0

        for lightcurve in list_of_random_lightcurves:

            print(f"Computing random light curve {i} of {len(list_of_random_lightcurves)}", end='\r')

            time, flux = lightcurve['time'], lightcurve['flux']
            with multiprocessing.Pool() as pool:
                power_lomb_1_chunks = pool.map(compute_lombscargle, [(time, flux, chunk) for chunk in frequency_chunks])
            power_lomb_shuffled = np.concatenate(power_lomb_1_chunks)
            list_of_random_lightcurves_lombed.append(power_lomb_shuffled)
            list_of_difference.append((power_lomb_1_regular - power_lomb_shuffled))

            i += 1

        #list_of_random_lightcurves_lombed_averaged = np.mean(list_of_random_lightcurves_lombed, axis=0)
        #power_lomb_1 = power_lomb_1_regular - list_of_random_lightcurves_lombed_averaged
        power_lomb_1 = np.mean(list_of_difference, axis=0)

    else:
        power_lomb_1 = power_lomb_1_regular

    # Second Lomb-Scargle periodogram on the power spectrum
    with multiprocessing.Pool() as pool:
        power_lomb_2_chunks = pool.map(compute_lombscargle, [(frequency, power_lomb_1, chunk) for chunk in period_chunks])

    power_lomb_2 = np.concatenate(power_lomb_2_chunks)


    return period, power_lomb_2

def run_lomb_scargle_analysis(kepler_dataframe, resolution=5000,period_range=(1, 30),list_of_random_lightcurves = False,different = False, power_filter = False):
    print("Running Lomb-Scargle Periodogram Analysis...")

    # Compute the Lomb-Scargle periodogram
    if not different:
        period, lomb2 = find_transits(kepler_dataframe["time"], kepler_dataframe["flux"], resolution,period_range,list_of_random_lightcurves)
    else:
        period, lomb2 =find_transits_adjusted(kepler_dataframe["time"], kepler_dataframe["flux"], resolution,period_range,list_of_random_lightcurves)

    
    # Determine peak detection parameters algorithmically
    height = np.median(lomb2) + 3 * np.std(lomb2)
    distance = resolution // 10000
    prominence = np.median(lomb2) + 3 * np.std(lomb2)
    
    # Find initial peaks using scipy's find_peaks with algorithmically determined parameters
    peaks, _ = find_peaks(lomb2,height=height, distance=distance, prominence=prominence)
    peak_pos = period[peaks]
    peak_powers = lomb2[peaks]
    
    # Exclude peaks in the low-period region based on the threshold
    if(power_filter):
        valid_peaks =(peak_powers > power_filter) 
    else:
        valid_peaks = peak_pos 

    peak_pos = peak_pos[valid_peaks]
    peak_powers = peak_powers[valid_peaks]
    
    print("Lomb-Scargle Periodogram analysis done")

    # Visualize the Lomb-Scargle periodogram and detected peaks
    plt.figure(figsize=(12, 6))
    plt.scatter(period, lomb2, label="Lomb-Scargle Periodogram", color = "black" ,linewidths= 0, s=1)
    plt.plot(peak_pos, peak_powers, "x", label="Detected Peaks", color="red")
    plt.xlabel("Period (days)")
    plt.yscale("log")
    plt.ylabel("Power")
    plt.title("Lomb-Scargle Periodogram")
    plt.legend()
    plt.show()

    return peak_pos , peak_powers



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

