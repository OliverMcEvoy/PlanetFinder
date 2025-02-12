import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation, get_sun
from astropy import units as u
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import h5py
import argparse
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import scipy.signal
import multiprocessing

def calculate_keplerian_orbit(period, transit_midpoint, semi_major_axis, time_array):
    '''
    Calculate the x and y values of the orbit.
    @params:
    period: float, the period of the planet.
    transit_midpoint: float, the transit midpoint of the planet.
    semi_major_axis: float, the semi-major axis of the planet.
    time_array: numpy.ndarray, the time array.
    @returns:
    x_value_of_orbit: numpy.ndarray, the x value of the orbit.
    y_value_of_orbit: numpy.ndarray, the y value of the orbit.
    '''
    orbit = np.pi * 2 * (time_array + transit_midpoint) / (period)  #
    x_value_of_orbit = semi_major_axis * np.cos(orbit)
    y_value_of_orbit = semi_major_axis * np.sin(orbit)
    return x_value_of_orbit, y_value_of_orbit


def calculate_limb_darkened_light_curve(
    light_curve,
    x_value_of_orbit,
    y_value_of_orbit,
    planet_radius,
    limb_darkening_u1,
    limb_darkening_u2,
    star_radius,
):
    '''
    Calculate the limb darkened light curve.
    @params:
    light_curve: numpy.ndarray, the light curve.
    x_value_of_orbit: numpy.ndarray, the x value of the orbit.
    y_value_of_orbit: numpy.ndarray, the y value of the orbit.
    planet_radius: float, the radius of the planet.
    limb_darkening_u1: float, the first limb darkening coefficient.
    limb_darkening_u2: float, the second limb darkening coefficient.
    star_radius: float, the radius of the star.
    @returns:
    light_curve: numpy.ndarray, the limb darkened light curve.
    '''
    star_radius_in_au = star_radius * (1 / 215)  # convert Stellar radius to AU
    planet_radius_in_au = planet_radius * (1 / 215)  # convert planet radius to AU

    transiting = abs(y_value_of_orbit) < (
        star_radius_in_au - planet_radius_in_au
    )  # this checks if the planet is passing in front of the star
    partial_transit = (
        abs(y_value_of_orbit) < (star_radius_in_au + planet_radius_in_au)
    ) & (
        abs(y_value_of_orbit) > (star_radius_in_au - planet_radius_in_au)
    )  # this checks if the planet is passing in front of the star

    infront_of_star = x_value_of_orbit > 0
    inside_transit = transiting * infront_of_star
    partial_transit = partial_transit * infront_of_star
    all_transits = inside_transit + partial_transit

    transit_depth = (planet_radius / star_radius) ** 2

    length_of_planet_transitioning = (
        star_radius_in_au + planet_radius_in_au - abs(y_value_of_orbit[partial_transit])
    ) / star_radius_in_au
    partial_transit_depth = (
        planet_radius_in_au / star_radius_in_au
    ) ** 2 * length_of_planet_transitioning

    average_intensity = (2 + limb_darkening_u2 * (1 - limb_darkening_u1)) / (
        2 + limb_darkening_u2
    )

    normalised_distance_from_center_of_star_inside_transit = np.sqrt(
        np.maximum(
            0.0000001, 1 - (y_value_of_orbit[inside_transit] / (star_radius_in_au)) ** 2
        )
    )
    limb_darkening_effect = 1 - limb_darkening_u1 * (
        1 - normalised_distance_from_center_of_star_inside_transit**limb_darkening_u2
    )
    normalised_limb_darkening_effect_inside_transit = (
        limb_darkening_effect * average_intensity
    )

    normalised_distance_from_center_of_star_partial_transit = np.sqrt(
        np.maximum(
            0.00000001,
            1 - (y_value_of_orbit[partial_transit] / (star_radius_in_au)) ** 2,
        )
    )
    limb_darkening_effect = 1 - limb_darkening_u1 * (
        1 - normalised_distance_from_center_of_star_partial_transit**limb_darkening_u2
    )
    normalised_limb_darkening_effect_partial_transit = (
        limb_darkening_effect * average_intensity
    )

    light_curve[inside_transit] = light_curve[inside_transit] - (
        transit_depth * normalised_limb_darkening_effect_inside_transit
    )
    light_curve[partial_transit] = light_curve[partial_transit] - (
        partial_transit_depth * normalised_limb_darkening_effect_partial_transit
    )
    return light_curve


def generate_multi_planet_light_curve(
    planets,
    total_time,
    star_radius=1.0,
    observation_noise=0.001,
    snr_threshold=5,
    u1=0.3,
    u2=0.2,
    cadence=0.0208333,
    simulate_gap_in_data=True,
):
    '''
    Generate a light curve for a multi-planetary system.
    @params:
    planets: list, a list of dictionaries containing the parameters of the planets.
    total_time: float, the total observation time in days.
    star_radius: float, the radius of the star.
    observation_noise: float, the observation noise.
    snr_threshold: float, the signal-to-noise ratio threshold for detection.
    u1: float, the first limb darkening coefficient.
    u2: float, the second limb darkening coefficient.
    cadence: float, the time interval between data points in days.
    simulate_gap_in_data: bool, whether to simulate gaps in the data.
    @returns:
    time_array: numpy.ndarray, the time array.
    flux_with_noise: numpy.ndarray, the light curve with noise.
    combined_light_curve: numpy.ndarray, the light curve no noise.
    '''
    time_array = np.arange(0, total_time, cadence)
    light_curve = np.ones_like(time_array)
    star_radius_squared = star_radius**2

    for planet in planets:
        period = planet["period"]
        planet_radius = planet["rp"]  # * (1/215) # convert from stellar radii to AU
        semi_major_axis = planet["a"]
        inclination = planet["incl"]
        transit_midpoint = planet["transit_midpoint"]

        x_value_of_orbit, y_value_of_orbit = calculate_keplerian_orbit(
            period, transit_midpoint, semi_major_axis, time_array
        )
        combined_light_curve = calculate_limb_darkened_light_curve(
            light_curve,
            x_value_of_orbit,
            y_value_of_orbit,
            planet_radius,
            u1,
            u2,
            star_radius,
        )

    flux_with_noise = combined_light_curve * (
        1 + np.random.normal(0, observation_noise*2, len(time_array))
    )

    if simulate_gap_in_data:
        # Introduce random gaps and fill them with 1
        num_gaps = np.random.randint(1, 4)
        for _ in range(num_gaps):
            gap_start = np.random.uniform(0, total_time - 50)
            gap_end = gap_start + np.random.uniform(0, 100)
            gap_mask = (time_array >= gap_start) & (time_array <= gap_end)
            flux_with_noise[gap_mask] = 1
            combined_light_curve[gap_mask] = 1

    return time_array, flux_with_noise, combined_light_curve


def limb_darken_values():
    u1 = np.random.uniform(0.8, 0.9)
    u2 = np.random.uniform(0.2, 0.3)
    # if 1- (u1 + u2) <0 :
    #     return limb_darken_values()
    return u1, u2


def generate_random_planet_systems(
    num_systems, max_planets_per_system, total_time, force_max_planets=False
):
    '''
    Generate random planetary systems.
    @params:
    num_systems: int, the number of planetary systems to generate.
    max_planets_per_system: int, the maximum number of planets per system.
    total_time: float, the total observation time in days.
    force_max_planets: bool, whether to force the maximum number of planets per system.
    @returns:
    systems: list, a list of dictionaries containing the parameters of the planetary systems.
    '''
    systems = []
    observation_noise = np.random.uniform(0.00015, 0.00025)
    for _ in range(num_systems):

        if force_max_planets:
            num_planets = max_planets_per_system
        else:
            num_planets = np.random.randint(2, max_planets_per_system + 1)

        planets = []

        star_radius = 1
        total_time = total_time

        for _ in range(num_planets):

            #Generate a random period distribution centered around 40 days
            standard_deviation = (np.log(70)-np.log(5))/2
            mean_period = np.random.uniform(35, 55)
            period = np.random.lognormal(mean=np.log(mean_period), sigma=standard_deviation)

            planet_radius = np.random.uniform(0.01, 0.04)
            semi_major_axis = (period / 365) ** (2 / 3) * np.random.uniform(0.75, 1.25)
            # varing other parameters has the same effect as eccentricity for this model so it does not make sense to vary it.
            # Well... a minor effect is that the transit duration is longer for higher eccentricities
            # But this is something I come back to later.
            eccentricity = 0  # np.random.uniform(0, 0.3)
            inclination = np.pi / 2
            transit_midpoint = np.random.uniform(0, period)

            planets.append(
                {
                    "period": period,
                    "rp": planet_radius,
                    "a": semi_major_axis,
                    "e": eccentricity,  # Note not currently used
                    "incl": inclination,
                    "transit_midpoint": transit_midpoint,
                }
            )

        u1, u2 = limb_darken_values()

        systems.append(
            {
                "planets": planets,
                "star_radius": star_radius,
                "observation_noise": observation_noise,
                "total_time": total_time,
                "u1": u1,
                "u2": u2,
            }
        )
    return systems


# def save(args):
#     with h5py.file ...:
#         flux = list -> str
#         flux = df['flux'].apply(lambda x: json.dumps(x))


def process_system(system, snr_threshold, total_time, cadence,lomb_scargle=False,double_lomb_scargle=False,plot=False):
    '''
    Generate a light curve for a given planetary system and return the number of detectable planets.
    @params:
    system: dict, a dictionary containing the parameters of the planetary system.
    snr_threshold: float, the signal-to-noise ratio threshold for detection.
    total_time: float, the total observation time in days.
    cadence: float, the time interval between data points in days.
    @returns:
    time_array: numpy.ndarray, the time array.
    flux_with_noise: numpy.ndarray, the light curve with noise.
    combined_light_curve: numpy.ndarray, the light curve no noise.
    total_time: float, the total observation time in days.
    star_radius: float, the radius of the star.
    observation_noise: float, the observation noise.
    u1: float, the first limb darkening coefficient.
    u2: float, the second limb darkening coefficient.
    planets: list, a list of dictionaries containing the parameters of the planets.
    num_detectable_planets: int, the number of detectable planets.
    total_planets: int, the total number of planets.
    '''
    time_array, flux_with_noise, combined_light_curve = (
        generate_multi_planet_light_curve(
            system["planets"],
            total_time=total_time,
            star_radius=system["star_radius"],
            observation_noise=system["observation_noise"],
            snr_threshold=snr_threshold,
            u1=system["u1"],
            u2=system["u2"],
            cadence=cadence,
        )
    )

    detectable_planets = []
    standard_deviation = np.std(flux_with_noise)
    for planet in system["planets"]:
        snr = (planet["rp"] * 2 / standard_deviation) * (
            np.floor(total_time / planet["period"]) ** 0.5
        )
        if snr >= snr_threshold and planet["period"] < 400:
            detectable_planets.append(planet)

    num_detectable_planets = len(detectable_planets)
    total_planets = len(system["planets"])

    if not lomb_scargle and not double_lomb_scargle:

        return (
            time_array,
            flux_with_noise,
            combined_light_curve,
            system["total_time"],
            system["star_radius"],
            system["observation_noise"],
            system["u1"],
            system["u2"],
            detectable_planets,
            num_detectable_planets,
            total_planets,
        )
    # Run the lomb scargle analysis
    xaxis , power = run_lomb_scargle_analysis(time_array, flux_with_noise,resolution=20000,period_range=(1,400) ,plot=plot,double_lomb_scargle=double_lomb_scargle)
    return (
        xaxis,
        power,
        system["star_radius"],
        system["planets"],
        num_detectable_planets,
        total_planets,
    )


def plot_light_curve(time_array, flux_with_noise, combined_light_curve):
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, flux_with_noise, label="Light Curve with Noise", color="red")
    if combined_light_curve is not False:
        plt.plot(time_array, combined_light_curve, label="Light Curve", color="blue")
    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.legend()
    plt.title("Light Curves")
    plt.show()

def compute_lombscargle(args):
    time, flux, frequency_chunk = args
    lombscargle = LombScargle(time, flux)
    power = lombscargle.power(frequency_chunk)
    return power

# Find transit peaks (Lomb-Scargle Periodogram)
def find_transits(time, flux, resolution, period_range, double_lomb_scargle=False, plot=False):
    '''
    @params
    time: array -> array containing the time values of the kepler dataframe.
    flux: array -> array containing the corresponding flux values for each time.
    resolution: int -> the number of points to use in the periodogram.

    find the transit peaks, the expected inputs 
    '''
    period = np.linspace(period_range[0], period_range[1], resolution)

    frequency_range = (time[1] - time[0], time[-1] - time[0])
    frequency = np.linspace((1 / frequency_range[1]), (1 / frequency_range[0]), resolution)

    # Compute the first Lomb-Scargle periodogram
    power_lomb_1 = compute_lombscargle((time, flux, frequency))

    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(frequency, power_lomb_1, label="First Lomb-Scargle Periodogram")
        plt.show()

    if double_lomb_scargle is False:
        return frequency, power_lomb_1

    # Second Lomb-Scargle periodogram on the power spectrum
    power_lomb_2 = compute_lombscargle((frequency, power_lomb_1, period))

    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(period, power_lomb_2, label="Second Lomb-Scargle Periodogram")
        plt.show()

    return period, power_lomb_2

def run_lomb_scargle_analysis(time, flux, resolution=5000,period_range=(1, 30), plot = True, double_lomb_scargle=False):
    xaxis, power = find_transits(time,flux, resolution,period_range,double_lomb_scargle=double_lomb_scargle, plot=plot)


    if plot:
        # Visualize the Lomb-Scargle periodogram and detected peaks
        plt.figure(figsize=(10, 6))
        plt.plot(xaxis, power, label="Lomb-Scargle Periodogram")
        plt.xlabel("Period (days)")
        plt.ylabel("Power")
        plt.title("Lomb-Scargle Periodogram")
        plt.legend()
        plt.show()

    return xaxis, power

def main():
    parser = argparse.ArgumentParser(
        description="Generate sample transits for multi-planetary systems."
    )
    parser.add_argument(
        "--num_systems",
        type=int,
        default=32,
        help="Number of planetary systems to generate.",
    )
    parser.add_argument(
        "--max_planets_per_system",
        type=int,
        default=6,
        help="Maximum number of planets per system.",
    )
    parser.add_argument(
        "--snr_threshold",
        type=float,
        default=5,
        help="Signal-to-noise ratio threshold for detection.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=64,
        help="Number of iterations for generating systems.",
    )
    parser.add_argument(
        "--total_time", type=float, default=365, help="Total observation time in days."
    )
    parser.add_argument(
        "--cadence",
        type=float,
        default=0.0208333,
        help="Time interval between data points in days.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="TrainingData/planet_systems.hdf5",
        help="Name of the output HDF5 file.",
    )
    parser.add_argument("--plot", action="store_true", help="Plot the light curves.")
    parser.add_argument("--single_lomb_scargle", action="store_true", help="Run the single lomb scargle analysis")
    parser.add_argument("--double_lomb_scargle", action="store_true", help="Run the double lomb scargle analysis")

    args = parser.parse_args()


    #run if not a lomb scargle required
    if args.single_lomb_scargle is False and args.double_lomb_scargle is False:
        with h5py.File(args.output_file, "w") as hdf5_file:
            for iteration in tqdm(
                range(args.num_iterations), desc="Generating planet systems"
            ):
                random_systems = generate_random_planet_systems(
                    args.num_systems, args.max_planets_per_system, args.total_time
                )

                with Pool(cpu_count()) as pool:
                    results = pool.starmap(
                        process_system,
                        [
                            (system, args.snr_threshold, args.total_time, args.cadence)
                            for system in random_systems
                        ],
                    )

                group = hdf5_file.create_group(f"iteration_{iteration}")
                for i, result in enumerate(results):
                    system_group = group.create_group(f"system_{i}")
                    system_group.create_dataset("time", data=result[0])
                    system_group.create_dataset("flux_with_noise", data=result[1])
                    system_group.create_dataset("num_detectable_planets", data=result[9])

                    planets_group = system_group.create_group("planets")
                    for j, planet in enumerate(result[8]):
                        planet_group = planets_group.create_group(f"planet_{j}")
                        planet_group.create_dataset("period", data=planet["period"])

            if args.plot:
                for result in results:
                    plot_light_curve(result[0], result[1], result[2])

            print("HDF5 file created successfully.")
    
    #Runs if a lomb scargle is required 

    with h5py.File(args.output_file, "w") as hdf5_file:
        print("lomb scargles generating...")
        for iteration in tqdm(
            range(args.num_iterations), desc="Generating planet systems using lomb scargle analysis"
        ):
            random_systems = generate_random_planet_systems(
                args.num_systems, args.max_planets_per_system, args.total_time
            )

            with Pool(cpu_count()) as pool:
                results = pool.starmap(
                    process_system,
                    [
                        (system, args.snr_threshold, args.total_time, args.cadence,args.single_lomb_scargle,args.double_lomb_scargle,args.plot)
                        for system in random_systems
                    ],
                )

            group = hdf5_file.create_group(f"iteration_{iteration}")
            for i, result in enumerate(results):
                system_group = group.create_group(f"system_{i}")
                system_group.create_dataset("xais", data=result[0])
                system_group.create_dataset("power", data=result[1])
                system_group.create_dataset("num_detectable_planets", data=result[4])

                planets_group = system_group.create_group("planets")
                #Use i for name to ensure no missing planets
                i = 0
                for j, planet in enumerate(result[3]):

                    #Filter out periods greater than 400 so they are not recorded
                    if planet["period"] < 400:
                        planet_group = planets_group.create_group(f"planet_{i}")
                        planet_group.create_dataset("period", data=planet["period"])
                        i += 1

        if args.plot:
            for result in results:
                plot_light_curve(result[0], result[1], False)

        print("HDF5 file created successfully.")




if __name__ == "__main__":
    main()
