import numpy as np
import torch
import matplotlib.pyplot as plt
from PlanetGenerationExtreme import generate_random_planet_systems, process_system
from transitFinderExtreme import TransitModel, load_model

def generate_light_curves(num_systems=10, max_planets_per_system=6, total_time=365, cadence=0.0208333):
    systems = generate_random_planet_systems(num_systems, max_planets_per_system, total_time,True)
    light_curves = []
    for system in systems:
        time_array, flux_with_noise, combined_light_curve, total_time, star_radius, observation_noise, u1, u2, planets, num_detectable_planets, total_planets = process_system(system, snr_threshold=1, total_time=total_time, cadence=cadence)
        light_curves.append((time_array, flux_with_noise, combined_light_curve, planets, num_detectable_planets, total_planets))

    return light_curves

def analyze_light_curves(light_curves, model, device):
    model.eval()
    results = []
    for time_array, flux_with_noise, combined_light_curve, planets, true_num_planets, total_planets in light_curves:
        flux_with_noise /= np.abs(flux_with_noise).max()
        flux_with_noise_tensor = torch.tensor(flux_with_noise, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_periods, predicted_num_planets = model(flux_with_noise_tensor)  # Model output: predicted periods and number of planets
            predicted_periods = predicted_periods.squeeze(0).cpu().numpy()
            predicted_num_planets = predicted_num_planets.item()

        # Extract actual periods from planets
        true_periods = [planet['period'] for planet in planets]
        
        # Print actual vs predicted periods and number of planets
        print(f"Actual periods: {true_periods}")
        print(f"Predicted periods: {predicted_periods}")
        print(f"Actual number of planets: {true_num_planets}")
        print(f"Predicted number of planets: {predicted_num_planets}")

        results.append((true_periods, predicted_periods, time_array, flux_with_noise, combined_light_curve, true_num_planets, predicted_num_planets))
    return results

def plot_light_curves(results):
    fig, axs = plt.subplots(len(results), 1, figsize=(15, 10))

    if len(results) == 1:
        axs = [axs]

    for i, (true_periods, predicted_periods, time_array, flux_with_noise, combined_light_curve, true_num_planets, predicted_num_planets) in enumerate(results):
        axs[i].plot(time_array, flux_with_noise, label='Normalized Light Curve', color='blue')
      #  axs[i].plot(time_array, combined_light_curve, label='No-noise Light Curve', color='blue')

        axs[i].set_xlabel('Time (days)')
        axs[i].set_ylabel('Flux')
        axs[i].legend()
        axs[i].set_title(f"True Planets: {true_num_planets}, Predicted Planets: {predicted_num_planets}")

        # Add text showing the actual and predicted periods
        actual_text = f"Actual periods: {', '.join([f'{p:.2f}' for p in true_periods])}"
        predicted_text = f"Predicted periods: {', '.join([f'{p:.2f}' for p in predicted_periods])}"

        axs[i].text(0.02, 0.98, actual_text, transform=axs[i].transAxes, fontsize=16, verticalalignment='top')
        axs[i].text(0.02, 0.92, predicted_text, transform=axs[i].transAxes, fontsize=16, verticalalignment='top')

    plt.tight_layout()
    plt.show()

def main():
    num_systems = 1
    max_planets_per_system = 2
    total_time = 1600

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("2_AlignedPeriodAndPlanetOnePercentTheGoodModeltransit_model_5_percent_weight_decay.pth", device)

    light_curves = generate_light_curves(num_systems, max_planets_per_system, total_time, cadence=0.02043357)
    results = analyze_light_curves(light_curves, model, device)
    plot_light_curves(results)

if __name__ == "__main__":
    main()