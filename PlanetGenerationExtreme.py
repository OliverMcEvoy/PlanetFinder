import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation, get_sun
from astropy import units as u
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import h5py
import argparse
import matplotlib.pyplot as plt

def calculate_keplerian_orbit(period, transit_midpoint, semi_major_axis, inclination, time_array):
    times = Time(time_array, format='jd')
    mean_anomaly = 2 * np.pi * (times.jd - transit_midpoint) / period
    true_anomaly = mean_anomaly
    x = semi_major_axis * (np.cos(true_anomaly) - 1)
    y = semi_major_axis * np.sin(true_anomaly)
    z = y * np.sin(inclination)
    y = y * np.cos(inclination)
    positions = CartesianRepresentation(x * u.au, y * u.au, z * u.au)
    projected_distance = np.sqrt(positions.x**2 + positions.y**2).to(u.au).value
    
    return projected_distance

def calculate_limb_darkened_light_curve(projected_distance, planet_radius, limb_darkening_u1, limb_darkening_u2, star_radius):
    star_radius = star_radius * (1/215) #convert Steller adius to AU
    normalised_distance = projected_distance # /star_radius
    normalised_planet_radius = planet_radius / star_radius
    light_curve = np.ones_like(normalised_distance)
    inside_transit = normalised_distance < (1 + normalised_planet_radius)
    valid_normalised_distance = np.clip(normalised_distance, 0, 1)
    mu = np.sqrt(1 - valid_normalised_distance**2)
    intensity = 1 - limb_darkening_u1 * (1 - mu) - limb_darkening_u2 * (1 - mu)**2
    light_curve[inside_transit] -= (
        normalised_planet_radius**2 *
        (1 - limb_darkening_u1 * (1 - mu[inside_transit]) - limb_darkening_u2 * (1 - mu[inside_transit])**2)
    )
    return light_curve

def generate_multi_planet_light_curve(planets, total_time, star_radius=1.0, observation_noise=0.001, snr_threshold=5, u1=0.3, u2=0.2, cadence=0.0208333, simulate_gap_in_data = True):
    time_array = np.arange(0, total_time, cadence)
    combined_light_curve = np.ones_like(time_array)
    star_radius_squared = star_radius ** 2

    for planet in planets:
        period = planet['period']
        planet_radius = planet['rp'] #* (1/215) # convert from stellar radii to AU
        semi_major_axis = planet['a']
        inclination = planet['incl']
        transit_midpoint = planet['transit_midpoint']

        projected_distance = calculate_keplerian_orbit(period, transit_midpoint, semi_major_axis, inclination, time_array)
        light_curve_model = calculate_limb_darkened_light_curve(projected_distance, planet_radius, u1, u2, star_radius)

        combined_light_curve *= light_curve_model

    flux_with_noise = combined_light_curve + np.random.normal(0, observation_noise, len(time_array))

    if simulate_gap_in_data:
        # Introduce random gaps and fill them with 1
        num_gaps = np.random.randint(1, 4)
        for _ in range(num_gaps):
            gap_start = np.random.uniform(0, total_time - 50)
            gap_end = gap_start + np.random.uniform(0, 40)
            gap_mask = (time_array >= gap_start) & (time_array <= gap_end)
            flux_with_noise[gap_mask] = 1
            combined_light_curve[gap_mask] = 1

    return time_array, flux_with_noise, combined_light_curve

def limb_darken_values():
    u1 = np.random.uniform(0.1, 1)
    u2 = np.random.uniform(0.0, 0.5)
    if 1- (u1 + u2) >0 :
        return limb_darken_values()
    return u1, u2

def generate_random_planet_systems(num_systems, max_planets_per_system, total_time, force_max_planets=False):
    systems = []
    observation_noise = 0.0001#np.random.uniform(0.0002, 0.0004)
    for _ in range(num_systems):

        if force_max_planets:
            num_planets = max_planets_per_system
        else:
            num_planets = np.random.randint(1, max_planets_per_system + 1)

        planets = []

        star_radius = np.random.uniform(1.0, 2.0)
        total_time = total_time

        for _ in range(num_planets):
            period = np.random.uniform(1, 50)
            planet_radius = np.random.uniform(0.0001, 0.05)
            semi_major_axis = (period**2)**(1/3)
            eccentricity = np.random.uniform(0, 0.3)
            inclination = np.pi / 2
            transit_midpoint = np.random.uniform(0, period)

            planets.append({
                'period': period,
                'rp': planet_radius,
                'a': semi_major_axis,
                'e': eccentricity, #Note not currently used 
                'incl': inclination,
                'transit_midpoint': transit_midpoint
            })

        u1,u2 = limb_darken_values()

        systems.append({
            'planets': planets,
            'star_radius': star_radius,
            'observation_noise': observation_noise,
            'total_time': total_time,
            'u1': u1,
            'u2': u2
        })
    return systems


# def save(args):
#     with h5py.file ...:
#         flux = list -> str
#         flux = df['flux'].apply(lambda x: json.dumps(x))

        

def process_system(system, snr_threshold, total_time, cadence):
    time_array, flux_with_noise, combined_light_curve = generate_multi_planet_light_curve(
        system['planets'], total_time=total_time, star_radius=system['star_radius'], 
        observation_noise=system['observation_noise'], snr_threshold=snr_threshold, 
        u1=system['u1'], u2=system['u2'], cadence=cadence
    )

    detectable_planets = []
    for planet in system['planets']:
        snr = np.max(np.abs(combined_light_curve - 1)) / system['observation_noise']
        if snr >= snr_threshold:
            detectable_planets.append(planet)

    num_detectable_planets = len(detectable_planets)
    total_planets = len(system['planets'])

    return (time_array, flux_with_noise, combined_light_curve, system['total_time'], system['star_radius'], system['observation_noise'], system['u1'], system['u2'], system['planets'], num_detectable_planets, total_planets)

def plot_light_curve(time_array, flux_with_noise, combined_light_curve):
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, flux_with_noise, label='Light Curve with Noise', color='red')
    plt.xlabel('Time (days)')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('Light Curves')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate sample transits for multi-planetary systems.")
    parser.add_argument('--num_systems', type=int, default=32, help='Number of planetary systems to generate.')
    parser.add_argument('--max_planets_per_system', type=int, default=6, help='Maximum number of planets per system.')
    parser.add_argument('--snr_threshold', type=float, default=5, help='Signal-to-noise ratio threshold for detection.')
    parser.add_argument('--num_iterations', type=int, default=64, help='Number of iterations for generating systems.')
    parser.add_argument('--total_time', type=float, default=365, help='Total observation time in days.')
    parser.add_argument('--cadence', type=float, default=0.0208333, help='Time interval between data points in days.')
    parser.add_argument('--output_file', type=str, default='planet_systems.hdf5', help='Name of the output HDF5 file.')
    parser.add_argument('--plot', action='store_true', help='Plot the light curves.')

    args = parser.parse_args()

    with h5py.File(args.output_file, 'w') as hdf5_file:
        for iteration in tqdm(range(args.num_iterations), desc="Generating planet systems"):
            random_systems = generate_random_planet_systems(args.num_systems, args.max_planets_per_system, args.total_time)

            with Pool(cpu_count()) as pool:
                results = pool.starmap(process_system, [
                    (system, args.snr_threshold, args.total_time, args.cadence) for system in random_systems
                ])

            group = hdf5_file.create_group(f'iteration_{iteration}')
            for i, result in enumerate(results):
                system_group = group.create_group(f'system_{i}')
                system_group.create_dataset('time', data=result[0])
                system_group.create_dataset('flux_with_noise', data=result[1])
                system_group.create_dataset('num_detectable_planets', data=result[9])

                planets_group = system_group.create_group('planets')
                for j, planet in enumerate(result[8]):
                    planet_group = planets_group.create_group(f'planet_{j}')
                    planet_group.create_dataset('period', data=planet['period'])

        if args.plot:
            for result in results:
                plot_light_curve(result[0], result[1], result[2])

        print("HDF5 file created successfully.")

if __name__ == "__main__":
    main()