import torch
import numpy as np
import matplotlib.pyplot as plt
import exoplanet
from transitFinder import TransitModel, plot_light_curves, load_model

# Define the function to generate the light curve
def generate_multi_planet_light_curve(planets, star_radius=1.0, observation_noise=0.001, total_time=365, snr_threshold=5, u1=0.3, u2=0.2, cadence=0.2):
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
            individual_light_curves.append(light_curve_model)
    
    flux_with_noise = planet_light_curves + np.random.normal(0, observation_noise, len(time))
    
    return time, flux_with_noise, planet_light_curves, detected_count, individual_light_curves

# Define the function to load the model and make predictions
def load_model_and_predict(model_path, planets):
    max_len = 16384
    # Generate the light curve
    time, flux_with_noise, detected_count, individual_light_curves = generate_multi_planet_light_curve(planets)
    
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
    
    flux_with_noise_tensor = torch.tensor(flux_with_noise, dtype=torch.float32).unsqueeze(0).to(device)
    time_tensor = torch.tensor(time, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Ensure the tensors have the correct shape
    if flux_with_noise_tensor.shape[1] != max_flux_len:
        flux_with_noise_tensor = torch.nn.functional.pad(flux_with_noise_tensor, (0, max_flux_len - flux_with_noise_tensor.shape[1]))
    if time_tensor.shape[1] != max_time_len:
        time_tensor = torch.nn.functional.pad(time_tensor, (0, max_time_len - time_tensor.shape[1]))
    
    # Make predictions
    with torch.no_grad():
        detected_count_pred= model(flux_with_noise_tensor, time_tensor)
    
    # Print the number of predicted planets vs actual planets
    print(f"Actual number of detected planets: {detected_count}")
    print(f"Predicted number of detected planets: {detected_count_pred.item()}")
    
    # Plot the actual individual light curves
    plt.figure(figsize=(12, 6))
    plt.plot(time, flux_with_noise, label=f'Actual Light Curve {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('Actual vs Predicted Individual Light Curves')
    plt.show()

def main():
    planets = [
        {'period': 10, 'rp': 0.1, 'a': 0.1, 'incl': np.pi/2, 'transit_midpoint': 5},
        {'period': 20, 'rp': 0.2, 'a': 0.2, 'incl': np.pi/2, 'transit_midpoint': 10},
    ]

    load_model_and_predict("transit_model.pth", planets)

if __name__ == "__main__":
    main()