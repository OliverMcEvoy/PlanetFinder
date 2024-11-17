import torch
import numpy as np
import matplotlib.pyplot as plt
import exoplanet
from transitFinder import TransitModel, load_model
import random

# Define the function to generate the light curve
def generate_multi_planet_light_curve(planets, star_radius=1.0, observation_noise=0.001, total_time=365, snr_threshold=10, u1=0.3, u2=0.2, cadence=0.2):
    time = np.arange(0, total_time, cadence)
    planet_light_curves = np.zeros_like(time)
    individual_light_curves = []
    
    detected_count = 0
    star_radius_squared = star_radius ** 2

    for planet in planets:
        period = planet['period']
        rp = planet['rp'] * star_radius
        a = planet['a']
        incl = planet['incl']
        t0 = planet['transit_midpoint']
        
        orbit = exoplanet.orbits.KeplerianOrbit(period=period, t0=t0, a=a, incl=incl)
        light_curve_model = exoplanet.LimbDarkLightCurve([u1, u2]).get_light_curve(
            orbit=orbit, r=rp, t=time
        ).eval().flatten()

        planet_light_curves += light_curve_model
        
        transit_depth = (rp ** 2) / star_radius_squared
        snr = transit_depth / observation_noise

        if snr > snr_threshold:
            detected_count += 1

    flux_with_noise = planet_light_curves + np.random.normal(0, observation_noise, len(time))
    
    return time, flux_with_noise, detected_count

def normalize_light_curve(flux):
    flux_with_noise_magnitude = np.abs(flux).max()
    normalized_flux = flux / flux_with_noise_magnitude
    return normalized_flux

# Load the model and make predictions
def load_model_and_predict(model_path, planets):
    max_len = 1825  # Adjusted to match the model's expected input length
    # Generate the light curve
    time, flux_with_noise, detected_count = generate_multi_planet_light_curve(planets)
    
    # Normalize the light curve
    normalized_flux_with_noise = normalize_light_curve(flux_with_noise)
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    
    model = load_model(model_path, device)
    
    # Preprocess the generated light curve
    max_flux_len = max_len
    max_time_len = max_len
    
    flux_with_noise_tensor = torch.tensor(normalized_flux_with_noise, dtype=torch.float32).unsqueeze(0).to(device)
    time_tensor = torch.tensor(time, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Ensure the tensors have the correct shape
    if flux_with_noise_tensor.shape[1] != max_flux_len:
        flux_with_noise_tensor = torch.nn.functional.pad(flux_with_noise_tensor, (0, max_flux_len - flux_with_noise_tensor.shape[1]))
    if time_tensor.shape[1] != max_time_len:
        time_tensor = torch.nn.functional.pad(time_tensor, (0, max_time_len - time_tensor.shape[1]))
    
    # Make predictions
    with torch.no_grad():
        detected_count_pred = model(flux_with_noise_tensor, time_tensor)
    
    # Convert to integer and calculate confidence
    detected_count_pred_int = torch.round(detected_count_pred).item()
    confidence_percentage = (detected_count_pred.item() / 5) * 100  # Assuming max 5 detectable planets
    
    # Print the number of predicted planets vs actual planets
    print(f"Actual number of detected planets: {detected_count}")
    print(f"Predicted number of detected planets: {detected_count_pred_int}")
    print(f"Confidence percentage: {confidence_percentage:.2f}%")
    
    # Plot the actual individual light curves
    plt.figure(figsize=(12, 6))
    plt.plot(time, flux_with_noise, label='Flux with Noise')
    plt.plot(time, normalized_flux_with_noise, label='Normalized Flux with Noise')
    
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('Actual vs Predicted Individual Light Curves')
    plt.show()

def generate_random_planets(num_planets=2):
    planets = []
    for _ in range(num_planets):
        planet = {
            'period': random.uniform(1, 365),
            'rp': random.uniform(0.05, 0.5),
            'a': random.uniform(1, 150),
            'incl': np.pi / 2,  # Assuming inclination is always pi/2
            'transit_midpoint': random.uniform(0, 365)  # Assuming transit midpoint can be any value within a year
        }
        planets.append(planet)
    return planets

def main():
    planets = generate_random_planets()

    load_model_and_predict("transit_model_365_days.pth", planets)

if __name__ == "__main__":
    main()