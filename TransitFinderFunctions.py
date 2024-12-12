import multiprocessing.pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from scipy.signal import find_peaks, medfilt, savgol_filter, lombscargle
from scipy.optimize import differential_evolution, minimize
import SyntheticLightCurveGeneration as lightcurve_generator
from scipy.optimize import differential_evolution
from tqdm import tqdm
import glob
import matplotlib.ticker as ticker
from IPython.display import display, Math


def loadDataFromFitsFiles(path, filter_type="savgol", randomise=False):
    """
    Function to load data from multiple fits files in a folder
    @params
    FolderPath: str -> the path to the folder containing the fits files
    @returns (all of the returns of this function are in a dataframe)
    time: array -> array containing the time values of the kepler dataframe.
    flux: array -> array containing the corresponding flux values for each time.
    error: array -> array containing the error values for each flux value.
    """
    time, flux, error = np.array([]), np.array([]), np.array([])
    for lcfile in glob.glob(path + "/*.fits"):
        with fits.open(lcfile) as lc:
            lc_data = lc[1].data
            tmptime = np.array(lc_data["TIME"])
            tmpflux = np.array(lc_data["PDCSAP_FLUX"])
            tmperror = np.array(lc_data["PDCSAP_FLUX_ERR"])

            # Bit wise operation to remove nans.
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

    # Determine the window length, which is the minimum of 51 or the array size -1 if the array is even or array size -1 if odd .
    array_size = len(flux)
    window_length = min(51, array_size - (array_size % 2 == 0))

    if filter_type == "savgol":
        normalising_filter = savgol_filter(
            flux, window_length=window_length, polyorder=3
        )
    elif filter_type == "medfilt":
        flux = flux.astype(np.float64)
        normalising_filter = medfilt(flux, kernel_size=51)

    flux = flux / normalising_filter
    error = error / normalising_filter

    # Filter random outliers, a less strict filter for outliers is used for the data below the mean to ensure that transits are not filtered.
    df = pd.DataFrame({"time": time, "flux": flux, "error": error})
    mean_flux = np.mean(flux)
    std_flux = np.std(flux)
    df = df[
        (df["flux"] <= mean_flux + 2 * std_flux)
        & (df["flux"] >= mean_flux - 8 * std_flux)
    ]

    return df


def analyze_period_with_bls(
    time,
    flux,
    error,
    period,
    resolution,
    duration_range,
    allowed_deviation,
    n_bootstrap,
):
    """
    Analyze a single period candidate using the BLS algorithm and bootstrap resampling.
    @params
    time: array -> array containing the time values of the kepler dataframe.
    flux: array -> array containing the corresponding flux values for each time.
    error: array -> array containing the error values for each flux value.
    period: float -> the period to analyze.
    resolution: int -> the resolution of the BLS algorithm.
    duration_range: tuple -> the range of durations to search for transits.
    allowed_deviation: float -> the allowed deviation from the period.
    n_bootstrap: int -> the number of bootstrap resamples to perform.
    @returns
    dict -> a dictionary containing the results of the BLS analysis.
    """
    # Define the minimum and maximum periods to analyse.
    min_period = period * (1 - allowed_deviation)
    max_period = period * (1 + allowed_deviation)

    bls = BoxLeastSquares(time, flux, dy=error)
    periods = np.linspace(min_period, max_period, resolution)
    durations = np.linspace(duration_range[0], duration_range[1], 200)
    results = bls.power(periods, durations)

    # Get the results based off max power.
    max_power_idx = np.argmax(results.power)
    best_period = results.period[max_power_idx]
    best_duration = results.duration[max_power_idx]
    best_transit_time = results.transit_time[max_power_idx]

    best_transit_model = bls.model(time, best_period, best_duration, best_transit_time)

    # Perform bootstrap resampling to estimate uncertainties.
    # This is done by resampling the time, flux, and error arrays with replacement.
    # Note: This is a computationally intensive process, but it results in accurate estimates fo the uncertainties.

    bootstrap_periods = []
    bootstrap_durations = []
    bootstrap_transit_times = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(time), len(time), replace=True)
        bootstrap_time = time[indices]
        bootstrap_flux = flux[indices]
        bootstrap_error = error[indices]

        bootstrap_bls = BoxLeastSquares(
            bootstrap_time, bootstrap_flux, dy=bootstrap_error
        )
        bootstrap_results = bootstrap_bls.power(periods, durations)

        bootstrap_max_power_idx = np.argmax(bootstrap_results.power)
        bootstrap_periods.append(bootstrap_results.period[bootstrap_max_power_idx])
        bootstrap_durations.append(bootstrap_results.duration[bootstrap_max_power_idx])
        bootstrap_transit_times.append(
            bootstrap_results.transit_time[bootstrap_max_power_idx]
        )

    period_uncertainty = np.std(bootstrap_periods)
    duration_uncertainty = np.std(bootstrap_durations)
    transit_time_uncertainty = np.std(bootstrap_transit_times)

    # Calculate the depth of the transit.
    # If it is too shallow filter the period out.
    depth = results.depth[max_power_idx]
    if depth < 0.000001:
        print(
            f"Skipping shallow transit with depth {depth:.6f} at period {period:.2f} days."
        )
        return None

    return {
        "candidate_period": period,
        "refined_period": best_period,
        "transit_model": best_transit_model,
        "power": max(results.power),
        "duration": best_duration,
        "depth": depth,
        "period_uncertainty": period_uncertainty,
        "duration_uncertainty": duration_uncertainty,
        "transit_time_uncertainty": transit_time_uncertainty,
    }


def analyze_peaks_with_bls(
    kepler_dataframe,
    peak_periods,
    resolution=5000,
    duration_range=(0.01, 0.5),
    allowed_deviation=0.05,
    n_bootstrap=100,
):
    """
    Analyze multiple period candidates using the BLS algorithm.
    @params
    kepler_dataframe: DataFrame -> the kepler dataframe to analyze.
    peak_periods: array -> an array of period candidates to analyze.
    resolution: int -> the resolution of the BLS algorithm.
    duration_range: tuple -> the range of durations to search for transits.
    allowed_deviation: float -> the allowed deviation from the period.
    n_bootstrap: int -> the number of bootstrap resamples to perform.
    @returns
    list -> a list of dictionaries containing the results of the BLS analysis.
    """
    time = kepler_dataframe["time"].values
    flux = kepler_dataframe["flux"].values
    error = kepler_dataframe["error"].values

    with multiprocessing.Pool() as pool:
        results_list = list(
            tqdm(
                pool.starmap(
                    analyze_period_with_bls,
                    [
                        (
                            time,
                            flux,
                            error,
                            period,
                            resolution,
                            duration_range,
                            allowed_deviation,
                            n_bootstrap,
                        )
                        for period in peak_periods
                    ],
                ),
                total=len(peak_periods),
                desc="Analyzing Periods",
            )
        )

    final_results = []
    for result in results_list:
        if result is None:
            continue
        if "error" in result:
            print(f"Error analyzing period {result['period']}: {result['error']}")
        else:
            final_results.append(result)

    return final_results


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


def remove_exact_duplicates(
    results_list, duplicates_percentage_threshold=0.05, complex_result_list=True
):
    """
    Remove exact duplicates,
    The tolerance of the duplication removal can be tuned to be as sentive or as strict as needed.
    @params
    results_list: list -> a list of dictionaries containing the results of the BLS analysis.
    duplicates_percentage_threshold: float -> the percentage threshold for duplicates.
    complex_result_list: bool -> a boolean to determine if the results list is a complex result list.
    @returns
    list -> a list of dictionaries containing the results of the BLS analysis with duplicates removed.
    """
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


def remove_duplicate_periods(
    results_list,
    duplicates_percentage_threshold=0.05,
    repeat_transit_threshold=0.05,
    power_threhsold_for_repeat_periods=0.1,
):
    """
    Remove the dupliate periods as a result of the BLS analysis.
    The tolerance of the duplication removal can be tuned to be as sentive or as strict as needed.
    @params
    results_list: list -> a list of dictionaries containing the results of the BLS analysis.
    duplicates_percentage_threshold: float -> the percentage threshold for duplicates.
    repeat_transit_threshold: float -> the threshold for repeat transits.
    power_threhsold_for_repeat_periods: float -> the power threshold for repeat periods.
    @returns
    list -> a list of dictionaries containing the results of the BLS analysis with duplicates removed.
    """

    # Remove exact duplicates
    unique_results = remove_exact_duplicates(
        results_list, duplicates_percentage_threshold
    )
    # Sort results by refined_period
    unique_results = sorted(unique_results, key=lambda x: x["refined_period"])

    final_results = []
    final_periods = set()
    final_powers = []

    # I am sorry about the amount of nesting in this monstrosity but this I think the code is readable enough for its purpose.
    # It essientially performs checks to see if the period is a repeat transit.
    for result in unique_results:
        period = result["refined_period"]
        power = result["power"] / result["refined_period"]  # to normalize the power
        is_unique = True

        for i, final_period in enumerate(final_periods):
            ratio = period / final_period
            lower_bound = (1 - repeat_transit_threshold) * ratio
            upper_bound = (1 + repeat_transit_threshold) * ratio

            # Check if the period is a repeat transit
            if lower_bound < ratio < upper_bound:
                final_power = final_powers[i]
                power_ratio = power / final_power
                power_lower_bound = 1 - power_threhsold_for_repeat_periods
                power_upper_bound = 1 + power_threhsold_for_repeat_periods

                # Check if the power is significantly different
                if power_lower_bound < power_ratio < power_upper_bound:
                    is_unique = False
                    break

                # Check if the period is a repeat transit with a combination of the previous periods
                for j in range(i + 1, len(final_powers)):
                    combined_power = final_power + final_powers[j]
                    combined_power_ratio = power / combined_power
                    if power_lower_bound < combined_power_ratio < power_upper_bound:
                        is_unique = False
                        break
                if not is_unique:
                    break

        # If the period is unique add it to the final results
        if is_unique:
            print(f"Adding period {period:.3f} days to the final results.")
            final_periods.add(period)
            final_powers.append(power)
            final_results.append(result)

    return final_results


def calculate_fit_for_period(
    result,
    time,
    flux,
    error,
    total_time,
    star_radius,
    cadence,
    method="minimize",
    minimize_options=None,
    nelder_mead_options=None,
    differential_options=None,
):
    """
    Calculate the best fit parameters for a given period. using 3 different optimization methods.
    The optimization methods present are:
    - minimize
    - differential_evolution
    - Nelder-Mead
    @params
    result: dict -> a dictionary containing the results of the BLS analysis.
    time: array -> array containing the time values of the kepler dataframe.
    flux: array -> array containing the corresponding flux values for each time.
    error: array -> array containing the error values for each flux value.
    total_time: float -> the total time of the kepler dataframe.
    star_radius: float -> the radius of the star.
    cadence: float -> the cadence of the kepler dataframe.
    method: str -> the optimization method to use, in the situation only 1 or 2 methods are deisred.
    minimize_options: dict -> the options for the minimize optimization method.
    nelder_mead_options: dict -> the options for the Nelder-Mead optimization method.
    differential_options: dict -> the options for the differential_evolution optimization method.
    """
    # Extract the necessary parameters from the BLS analysis
    period = result["refined_period"]
    bls_model_flux = result["transit_model"]
    transit_duration = result["duration"]

    # Phase fold the light curve

    kepler_phase = phase_fold(time, period, bls_model_flux=bls_model_flux)
    planet_radius = estimate_planet_radius(result["depth"], star_radius)

    # Set default options incase a user does not pass them in
    if minimize_options is None:
        minimize_options = {
            "maxiter": 10,
            "disp": False,
            "ftol": 1e-4,
        }

    if nelder_mead_options is None:
        nelder_mead_options = {
            "maxiter": 10,
            "disp": False,
            "fatol": 1e-4,
        }

    if differential_options is None:
        differential_options = {
            "maxiter": 15,
            "popsize": 10,
            "disp": False,
            "tol": 1e-6,
        }

    # Guess the semi major axis, limb darkening coefficients, and planet radius.
    # The SMA guess is an ok guess, maybe its something to improve on for future iterations.
    # The planet radius relies on an accurate Transit depth from the BLS analysis.
    guess_for_sma = (period / 365) ** (2 / 3)
    bounds = [
        (guess_for_sma * 0.5, guess_for_sma * 2),
        (0.7, 0.95),
        (0.2, 0.6),
        (planet_radius * 0.5, planet_radius * 2.5),
    ]
    initial_guess = [guess_for_sma, 0.85, 0.25, planet_radius]

    # Define the args to be passed into the optimization function, these are to be kept constant.
    # I tried to vary the period but then the minimization function would missbehave and give unphysical results.
    # Maybe it could be revisiited with a lower tolerance.
    args = (
        period,
        total_time,
        kepler_phase,
        star_radius,
        flux,
        cadence,
        error,
        transit_duration,
    )

    if method == "minimize":
        result = minimize(
            calculate_reduced_chi_squared,
            initial_guess,
            args=args,
            method="L-BFGS-B",
            bounds=bounds,
            options=minimize_options,
        )
        best_fit_params = result.x

    elif method == "differential_evolution":
        result = differential_evolution(
            calculate_reduced_chi_squared, bounds, args=args, **differential_options
        )
        best_fit_params = result.x

    elif method == "Nelder-Mead":
        result = minimize(
            calculate_reduced_chi_squared,
            initial_guess,
            args=args,
            method="Nelder-Mead",
            options=nelder_mead_options,
        )
        best_fit_params = result.x
    else:
        raise ValueError(f"Invalid optimization method: {method}")

    # Extract best-fit parameters
    best_fit_a = best_fit_params[0]
    best_fit_u1 = best_fit_params[1]
    best_fit_u2 = best_fit_params[2]
    best_fit_radius = best_fit_params[3]

    # Generate the best-fit model light curve
    planets = [
        {
            "period": period,
            "rp": best_fit_radius,
            "a": best_fit_a,
            "incl": np.pi / 2,
            "transit_midpoint": period / 2,
        }
    ]
    synthetic_time, best_fit_synthetic_lightcurve, _ = (
        lightcurve_generator.generate_multi_planet_light_curve(
            planets,
            total_time,
            star_radius,
            0,
            snr_threshold=5,
            u1=best_fit_u1,
            u2=best_fit_u2,
            cadence=cadence,
            simulate_gap_in_data=False,
        )
    )

    synthetic_phase = phase_fold(synthetic_time, period, best_fit_synthetic_lightcurve)
    best_fit_synthetic_lightcurve_interpolated = interoplate_phase_folded_light_curve(
        kepler_phase,
        synthetic_phase,
        best_fit_synthetic_lightcurve,
    )

    final_chi2 = calculate_reduced_chi_squared(
        best_fit_params,
        period,
        total_time,
        kepler_phase,
        star_radius,
        flux,
        cadence,
        error,
        transit_duration,
    )

    return {
        "filtered_phase": kepler_phase,
        "flux": flux,
        "bls_model_flux": bls_model_flux,
        "best_fit_model_lightcurve": best_fit_synthetic_lightcurve_interpolated,
        "final_chi2": final_chi2,
        "best_fit_params": best_fit_params,
        "planet_radius": planet_radius,
        "method": method,
        "transit_duration": transit_duration,
        "period": period,
    }


def calculate_best_fit_parameters(
    kepler_dataframe,
    results_list,
    minimize_options=None,
    nelder_mead_options=None,
    differential_options=None,
    methods=["minimize", "differential_evolution", "Nelder-Mead"],
):
    """
    This function sets up the some multiprocessing to run the optimization functions in parallel.
    @params
    kepler_dataframe: DataFrame -> the kepler dataframe to analyze.
    results_list: list -> a list of dictionaries containing the results of the BLS analysis.
    minimize_options: dict -> the options for the minimize optimization method.
    nelder_mead_options: dict -> the options for the Nelder-Mead optimization method.
    differential_options: dict -> the options for the differential_evolution optimization method.
    methods: list -> a list of optimization methods to use.
    @returns
    list -> a list of dictionaries containing the results of the fitting analysis.
    """
    time = kepler_dataframe["time"].values
    flux = kepler_dataframe["flux"].values
    error = kepler_dataframe["error"].values
    star_radius = 1
    cadence = 0.0208333
    total_time = time[-1]
    all_results = []

    # Prepare arguments for each method
    pool_args = []
    for method in methods:
        method_options = {
            "minimize": minimize_options,
            "Nelder-Mead": nelder_mead_options,
            "differential_evolution": differential_options,
        }

        for result in results_list:
            pool_args.append(
                (
                    result,
                    time,
                    flux,
                    error,
                    total_time,
                    star_radius,
                    cadence,
                    method,
                    minimize_options if method == "minimize" else None,
                    (
                        differential_options
                        if method == "differential_evolution"
                        else None
                    ),
                    nelder_mead_options if method == "Nelder-Mead" else None,
                )
            )

    # Run optimizations in parallel
    # Gotta love a wee bit of multiprocessing, dont gotta worry about the rooom being too cold anymore.
    with multiprocessing.Pool() as pool:
        results = pool.starmap(calculate_fit_for_period, pool_args)

    # Organize results by method, some finish earlier when multiproccessing is used.
    index = 0
    for method in methods:
        method_results = results[index : index + len(results_list)]
        all_results.append(method_results)
        index += len(results_list)

    return all_results


def phase_fold(time, period, flux=None, bls_model_flux=None):
    """
    Phase fold the light curve.
    It also centers the light curve based on ethier the flux or the bls_flux provided,
    bls_flux is required for the data given as part of the coursework, but the regular flux can be used for the syntehtic lightcurve.
    @params
    time: array -> array containing the time values of the kepler dataframe.
    period: float -> the period to phase fold the light curve.
    flux: array -> array containing the corresponding flux values for each time.
    bls_model_flux: array -> array containing the BLS model flux values.
    @returns
    array -> an array containing the phase folded light curve.

    """
    phase = (time % period) / period
    if bls_model_flux is not None:
        # Identify indices where the flux is at the lower level (in transit)
        in_transit = bls_model_flux < np.max(bls_model_flux)
        # Find the times corresponding to the transit
        transit_times = time[in_transit]
        # Calculate the midpoint of the transit times
        if len(transit_times) > 0:
            transit_midpoint = np.mean(transit_times % period) / period
            phase = phase - transit_midpoint + 0.5
    elif flux is not None:
        min_flux_phase = (time[np.argmin(flux)] % period) / period
        phase = phase - min_flux_phase + 0.5
    else:
        phase = phase + 0.5
    return phase


def calculate_reduced_chi_squared(
    params,
    period,
    total_time,
    kepler_phase,
    star_radius,
    flux,
    cadence,
    error,
    transit_duration,
):
    """
    Calculate the reduced chi squared value for a generated syntehtic light curve and the actual data
    @params
    params: array -> an array containing the parameters to optimize.
    period: float -> the period to optimize.
    total_time: float -> the total time of the kepler dataframe.
    kepler_phase: array -> an array containing the phase folded light curve.
    star_radius: float -> the radius of the star.
    flux: array -> array containing the corresponding flux values for each time.
    cadence: float -> the cadence of the kepler dataframe.
    error: array -> array containing the error values for each flux value.
    transit_duration: float -> the duration of the transit.
    @returns
    float -> the reduced chi squared value.
    """

    # Set up the synethic light curve genertor with the parameters to optimize.
    planets = [
        {
            "period": period,
            "rp": params[3],
            "a": params[0],
            "incl": np.pi / 2,
            "transit_midpoint": period / 2,
        }
    ]
    u1 = params[1]
    u2 = params[2]
    observation_noise = 0

    # Generate the syntehtic light curve
    # The name is slightly missleading as it can generate a lightcurve for a single planet as well.
    time, pg_model_lightcurve, _ = (
        lightcurve_generator.generate_multi_planet_light_curve(
            planets,
            total_time,
            star_radius,
            observation_noise,
            snr_threshold=0,
            u1=u1,
            u2=u2,
            cadence=cadence,
            simulate_gap_in_data=False,
        )
    )

    # Fold the light curve and interpolate it onto the kepler phase
    pg_generated_phase = phase_fold(time, period, pg_model_lightcurve)
    pg_nodel_lightcurve_projected_onto_kepler_phase = (
        interoplate_phase_folded_light_curve(
            kepler_phase, pg_generated_phase, pg_model_lightcurve
        )
    )

    # Determine the window on which to optimize around
    transit_midpoint = period / 2
    window_size = transit_duration * 1.2 / period

    window_mask = (kepler_phase >= (0.5 - window_size)) & (
        kepler_phase <= (0.5 + window_size)
    )

    chi_2 = np.sum(
        (
            (
                flux[window_mask]
                - pg_nodel_lightcurve_projected_onto_kepler_phase[window_mask]
            )
            ** 2
        )
        / (error[window_mask] ** 2)
    )
    reduced_chi_2 = chi_2 / (len(flux[window_mask]) - len(params))
    return reduced_chi_2


def interoplate_phase_folded_light_curve(
    kepler_phase, pg_generated_phase, pg_model_lightcurve
):
    """
    Interpolate the phase folded light curve onto the kepler phase.
    This function is the result of many many iterions of this function, I am not proud of the amount of time it took to get this function to this point.
    But I am proud of the result, the beauty of this function lies in its simiplicity.
    @params
    kepler_phase: array -> an array containing the phase folded light curve.
    pg_generated_phase: array -> an array containing the phase folded light curve of the generated light curve.
    pg_model_lightcurve: array -> an array containing the model light curve of the generated light curve.
    @returns
    array -> an array containing the interpolated light curve.
    """
    interpolated_values = np.zeros_like(kepler_phase)

    for i, kp in enumerate(kepler_phase):
        closest_index = np.argmin(np.abs(pg_generated_phase - kp))
        interpolated_values[i] = pg_model_lightcurve[closest_index]

    return interpolated_values


def plot_phase_folded_light_curves(all_results):
    """
    Plot the phase folded light curves for each method.
    Its a bit janky and difficult to read, not proud of the presentation of this code but I think the output is pretty.
    @params
    all_results: list -> a list of dictionaries containing the results of the fitting analysis.
    """
    num_candidates = len(all_results[0])
    fig, axs = plt.subplots(4, num_candidates, figsize=(4 * num_candidates, 12))

    methods = ["minimize", "differential_evolution", "Nelder-Mead"]
    colors = ["orangered", "limegreen", "darkviolet"]

    if num_candidates == 1:
        axs = np.array([axs]).T

    for j in range(num_candidates):
        # Top plot: Filtered Flux and BLS Model Flux (first method)

        period = all_results[0][j]["period"]
        result = all_results[0][j]
        window_size = result["transit_duration"] * 1.2 / result["period"]
        mask = (result["filtered_phase"] >= 0.5 - window_size) & (
            result["filtered_phase"] <= 0.5 + window_size
        )
        axs[0][j].errorbar(
            result["filtered_phase"][mask],
            result["flux"][mask],
            fmt="o",
            color="black",
            alpha=0.7,
            linestyle="none",
        )
        axs[0][j].errorbar(
            result["filtered_phase"][mask],
            result["bls_model_flux"][mask],
            fmt="^",
            color="teal",
            alpha=0.5,
            linestyle="none",
            markersize=4,
        )
        axs[0][j].set_title(
            f"Phase-Folded Light Curve\nPeriod of {period:.3f} days", fontsize=12
        )
        axs[0][j].set_ylabel("Normalised Flux , BLS Model")

        axs[0][j].tick_params(direction="in", which="both")
        axs[0][j].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        axs[0][j].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0][j].yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        axs[0][j].yaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Bottom plots: Best Fit Model Light Curve for each method
        for i, method_results in enumerate(all_results):
            result = method_results[j]

            best_fit_params = result["best_fit_params"]
            period = result["period"]

            planets = [
                {
                    "period": period,
                    "rp": best_fit_params[3],
                    "a": best_fit_params[0],
                    "incl": np.pi / 2,
                    "transit_midpoint": period / 2,
                }
            ]
            pg_time, best_fit_model_lightcurve, _ = (
                lightcurve_generator.generate_multi_planet_light_curve(
                    planets,
                    1600,
                    1,
                    0,
                    snr_threshold=1,
                    u1=best_fit_params[1],
                    u2=best_fit_params[2],
                    cadence=0.0005,
                    simulate_gap_in_data=False,
                )
            )

            pg_generated_phase = phase_fold(pg_time, period, best_fit_model_lightcurve)

            pg_mask = (pg_generated_phase >= 0.5 - window_size) & (
                pg_generated_phase <= 0.5 + window_size
            )

            axs[i + 1][j].errorbar(
                result["filtered_phase"][mask],
                result["flux"][mask],
                fmt="o",
                color="black",
                alpha=0.7,
                linestyle="none",
                markersize=5,
            )
            axs[i + 1][j].set_ylabel(f"Normalised Flux, {methods[i]}  ")
            axs[i + 1][j].legend()

            axs[i + 1][j].errorbar(
                pg_generated_phase[pg_mask],
                best_fit_model_lightcurve[pg_mask],
                fmt="^",
                color=colors[i],
                alpha=1,
                linestyle="none",
                markersize=4,
            )
            axs[i + 1][j].legend(loc="upper right")

            axs[i + 1][j].tick_params(direction="in", which="both")
            axs[i + 1][j].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            axs[i + 1][j].xaxis.set_minor_locator(ticker.AutoMinorLocator())
            axs[i + 1][j].yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            axs[i + 1][j].yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Hide y-axis labels for all plots except those in the first column.
    for ax in axs[:, 1:].flatten():
        ax.set_ylabel("")
        ax.yaxis.set_ticklabels([])

    # Hide x-axis labels for all plots except those in the last row.
    for ax in axs[:-1, :].flatten():
        ax.set_xlabel("")
        ax.xaxis.set_ticklabels([])

    # Add x-axis label to the bottom row of plots.
    for ax in axs[-1, :]:
        ax.set_xlabel("Phase")

    # Figure description.
    fig.text(
        0.5,
        0.01,
        "Fig.1. The fitted parameters for each of the detected periods and for each of the minimisation methods, It can be seen some fit better than others e.g . However it is apparent from the graph that the method of lightcurve generation to minimise the reduced $chi^2$ seems to work quite well",
        ha="center",
        fontsize=12,
    )

    plt.tight_layout()
    plt.show()


def print_best_fit_parameters(all_results, bls_analysis):
    """
    A Table of the results
    @params
    all_results: list -> a list of dictionaries containing the results of the fitting analysis.
    bls_analysis: list -> a list of dictionaries containing the results of the BLS analysis.
    """
    columns = ["Period", "\alpha", "$R_p$", "chi2"]
    data = []
    period_groups = {}
    chi2_list = {}
    u1_values = []
    u2_values = []

    # Get Period Uncertainitys
    period_uncertainties = {}
    for result in bls_analysis:
        period_floor = np.floor(result["candidate_period"])
        period_uncertainties[period_floor] = result["period_uncertainty"]

    # Group results by period
    for method_results in all_results:
        for result in method_results:
            period = result["period"]
            best_fit_params = result["best_fit_params"]
            chi2 = result["final_chi2"]
            if period not in period_groups:
                period_groups[period] = []
                chi2_list[period] = []
            period_groups[period].append(best_fit_params)
            chi2_list[period].append(chi2)
            u1_values.append(best_fit_params[1])
            u2_values.append(best_fit_params[2])

    # Calculate average and standard deviation for each group
    for period, params_list in period_groups.items():
        params_array = np.array(params_list)
        avg_params = np.mean(params_array, axis=0)
        std_params = np.std(params_array, axis=0)
        chi2_np = np.array(chi2_list[period])
        best_chi2 = np.min(chi2_np)
        std_chi2 = np.std(chi2_np)
        period_floor = np.floor(period)
        data.append(
            [
                f"{period:.5f} ± {period_uncertainties[period_floor]:.5f}",
                f"{avg_params[0]:.4f} ± {std_params[0]:.4f}",
                f"{avg_params[3]:.4f} ± {std_params[3]:.4f}",
                f"{best_chi2:.4f}",
            ]
        )

    df = pd.DataFrame(data, columns=columns)

    def make_pretty(styler):
        styler.set_caption("Table: Best Fit Parameters for Each Planet")
        styler.format(precision=3, thousands=",", decimal=".")
        return styler

    styled_df = df.style.pipe(make_pretty)
    display(styled_df)

    # Calculate and print total average and standard deviation for u1 and u2
    u1_np = np.array(u1_values)
    u2_np = np.array(u2_values)
    avg_u1 = np.mean(u1_np)
    std_u1 = np.std(u1_np)
    avg_u2 = np.mean(u2_np)
    std_u2 = np.std(u2_np)
    display(
        Math(
            r"Limb darkening Coefficient $u_1$: {:.3f} $\pm$ {:.3f}".format(
                avg_u1, std_u1
            )
        )
    )
    display(
        Math(
            r"Limb darkening Coefficient $u_2$: {:.3f} $\pm$ {:.3f}".format(
                avg_u2, std_u2
            )
        )
    )


# Below this point are the functions used to find the intial peaks,
# It is not directly refrenced in the report hence I have not included it in the main body of the code.
# I experimented with a couple of different methods to find the peaks,
# Whats below seems to work ok
# But I have not taken too much time to optimize it, or encourage readability.
# So read below this point at your own risk!


def getRandomisedData(path, interations, filter_type):
    """
    Function to load data from multiple fits files in a folder
    However the twist with this function is the data is randomised!
    This is to allow a altered form of the lomb-scargle periodogram to be tested where the data is randomised and then the periodogram is run on the randomised data
    The difference between the randomised lomb-scargle periodogram and the normal lomb-scargle periodogram is then calculated and it helps identify the intial peaks
    @params
    FolderPath: str -> the path to the folder containing the fits files
    @returns (all of the returns of this function are in a dataframe )
    time: array -> array containing the time values of the kepler dataframe.
    flux: array -> array containing the corresponding flux values for each time.
    error: array -> array containing the error values for each flux value.
    """

    list_of_random_lightcurves = []
    for i in range(interations):
        list_of_random_lightcurves.append(
            loadDataFromFitsFiles(path, filter_type=filter_type, randomise=True)
        )

    return list_of_random_lightcurves


def compute_lombscargle(args):
    time, flux, frequency_chunk = args
    return lombscargle(time, flux, frequency_chunk, precenter=True, normalize=False)


# Find transit peaks (Lomb-Scargle Periodogram)
def find_transits(time, flux, resolution, period_range, list_of_random_lightcurves):
    """
    @params
    time: array -> array containing the time values of the kepler dataframe.
    flux: array -> array containing the corrosponding flux values for each time.
    resolution: int -> the number of points to use in the periodogram.

    find the transit peaks, the expected inputs
    """
    period = np.linspace(period_range[0], period_range[1], resolution)

    frequency_range = (time[1] - time[0], time[len(time)] - time[0])
    frequency = np.linspace(
        (1 / frequency_range[1]), (1 / frequency_range[0]), resolution
    )

    # Split frequency array into chunks for parallel processing
    num_chunks = 12  # Number of processes
    frequency_chunks = np.array_split(frequency, num_chunks)
    period_chunks = np.array_split(period, num_chunks)

    # Use multiprocessing to compute the first Lomb-Scargle periodogram
    with multiprocessing.Pool() as pool:
        power_lomb_1_chunks = pool.map(
            compute_lombscargle, [(time, flux, chunk) for chunk in frequency_chunks]
        )

    # Combine the results from each chunk
    power_lomb_1 = np.concatenate(power_lomb_1_chunks)

    # Second Lomb-Scargle periodogram on the power spectrum
    with multiprocessing.Pool() as pool:
        power_lomb_2_chunks = pool.map(
            compute_lombscargle,
            [(frequency, power_lomb_1, chunk) for chunk in period_chunks],
        )

    power_lomb_2_regular = np.concatenate(power_lomb_2_chunks)

    if list_of_random_lightcurves:

        list_of_random_lightcurves_lombed = []
        list_of_difference_between_rand_and_regular = []
        power_lomb_difference_squared = np.zeros_like(power_lomb_2_regular)

        print("List of random light curves present, computing random light curves")
        i = 0

        for lightcurve in list_of_random_lightcurves:

            print(
                f"Computing random light curve {i} of {len(list_of_random_lightcurves)}",
                end="\r",
            )

            time, flux = lightcurve["time"], lightcurve["flux"]
            with multiprocessing.Pool() as pool:
                power_lomb_1_chunks = pool.map(
                    compute_lombscargle,
                    [(time, flux, chunk) for chunk in frequency_chunks],
                )
            power_lomb_1 = np.concatenate(power_lomb_1_chunks)
            with multiprocessing.Pool() as pool:
                power_lomb_2_chunks = pool.map(
                    compute_lombscargle,
                    [(frequency, power_lomb_1, chunk) for chunk in period_chunks],
                )
            power_lomb_2 = np.concatenate(power_lomb_2_chunks)
            list_of_random_lightcurves_lombed.append(power_lomb_2)
            difference_squared = (power_lomb_2_regular - power_lomb_2) ** 2
            list_of_difference_between_rand_and_regular.append(difference_squared)
            power_lomb_difference_squared += difference_squared
            i += 1

        list_of_random_lightcurves_lombed_averaged = np.mean(
            list_of_random_lightcurves_lombed, axis=0
        )

        power_lomb_2 = power_lomb_2_regular - list_of_random_lightcurves_lombed_averaged

    else:
        power_lomb_2 = power_lomb_2_regular

    return period, power_lomb_2


def find_transits_adjusted(
    time, flux, resolution, period_range, list_of_random_lightcurves
):
    """
    @params
    time: array -> array containing the time values of the kepler dataframe.
    flux: array -> array containing the corresponding flux values for each time.
    resolution: int -> the number of points to use in the periodogram.

    find the transit peaks, the expected inputs
    """
    period = np.linspace(period_range[0], period_range[1], resolution)

    frequency_range = (time[1] - time[0], time[len(time)] - time[0])
    frequency = np.linspace(
        (1 / frequency_range[1]), (1 / frequency_range[0]), resolution
    )

    # Split frequency array into chunks for parallel processing
    num_chunks = 12  # Number of processes
    frequency_chunks = np.array_split(frequency, num_chunks)
    period_chunks = np.array_split(period, num_chunks)

    # Use multiprocessing to compute the first Lomb-Scargle periodogram
    with multiprocessing.Pool() as pool:
        power_lomb_1_chunks = pool.map(
            compute_lombscargle, [(time, flux, chunk) for chunk in frequency_chunks]
        )

    # Combine the results from each chunk
    power_lomb_1_regular = np.concatenate(power_lomb_1_chunks)

    if list_of_random_lightcurves:

        list_of_random_lightcurves_lombed = []
        list_of_difference = []

        print("List of random light curves present, computing random light curves")
        i = 0

        for lightcurve in list_of_random_lightcurves:

            print(
                f"Computing random light curve {i} of {len(list_of_random_lightcurves)}",
                end="\r",
            )

            time, flux = lightcurve["time"], lightcurve["flux"]
            with multiprocessing.Pool() as pool:
                power_lomb_1_chunks = pool.map(
                    compute_lombscargle,
                    [(time, flux, chunk) for chunk in frequency_chunks],
                )
            power_lomb_shuffled = np.concatenate(power_lomb_1_chunks)
            list_of_random_lightcurves_lombed.append(power_lomb_shuffled)
            list_of_difference.append((power_lomb_1_regular - power_lomb_shuffled))

            i += 1

        # list_of_random_lightcurves_lombed_averaged = np.mean(list_of_random_lightcurves_lombed, axis=0)
        # power_lomb_1 = power_lomb_1_regular - list_of_random_lightcurves_lombed_averaged
        power_lomb_1 = np.mean(list_of_difference, axis=0)

    else:
        power_lomb_1 = power_lomb_1_regular

    # Second Lomb-Scargle periodogram on the power spectrum
    with multiprocessing.Pool() as pool:
        power_lomb_2_chunks = pool.map(
            compute_lombscargle,
            [(frequency, power_lomb_1, chunk) for chunk in period_chunks],
        )

    power_lomb_2 = np.concatenate(power_lomb_2_chunks)

    return period, power_lomb_2


def run_lomb_scargle_analysis(
    kepler_dataframe,
    resolution=5000,
    period_range=(1, 30),
    list_of_random_lightcurves=False,
    different=False,
    power_filter=False,
):
    print("Running Lomb-Scargle Periodogram Analysis...")

    # Compute the Lomb-Scargle periodogram
    if not different:
        period, lomb2 = find_transits(
            kepler_dataframe["time"],
            kepler_dataframe["flux"],
            resolution,
            period_range,
            list_of_random_lightcurves,
        )
    else:
        period, lomb2 = find_transits_adjusted(
            kepler_dataframe["time"],
            kepler_dataframe["flux"],
            resolution,
            period_range,
            list_of_random_lightcurves,
        )

    # Determine peak detection parameters algorithmically
    height = np.median(lomb2) + 3 * np.std(lomb2)
    distance = resolution // 10000
    prominence = np.median(lomb2) + 3 * np.std(lomb2)

    # Find initial peaks using scipy's find_peaks with algorithmically determined parameters
    peaks, _ = find_peaks(
        lomb2, height=height, distance=distance, prominence=prominence
    )
    peak_pos = period[peaks]
    peak_powers = lomb2[peaks]

    # Exclude peaks in the low-period region based on the threshold
    if power_filter:
        valid_peaks = peak_powers > power_filter
    else:
        valid_peaks = peak_pos

    peak_pos = peak_pos[valid_peaks]
    peak_powers = peak_powers[valid_peaks]

    print("Lomb-Scargle Periodogram analysis done")

    # Visualize the Lomb-Scargle periodogram and detected peaks
    plt.figure(figsize=(12, 6))
    plt.scatter(
        period,
        lomb2,
        label="Lomb-Scargle Periodogram",
        color="black",
        linewidths=0,
        s=1,
    )
    plt.plot(peak_pos, peak_powers, "x", label="Detected Peaks", color="red")
    plt.xlabel("Period (days)")
    plt.yscale("log")
    plt.ylabel("Power")
    plt.title("Lomb-Scargle Periodogram")
    plt.legend()
    plt.show()

    return peak_pos, peak_powers
