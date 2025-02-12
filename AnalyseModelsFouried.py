import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from SyntheticLightCurveGeneration import generate_random_planet_systems, process_system ,run_lomb_scargle_analysis
from RNN import load_model as load_model_RNN
from CNN import load_model as local_model_CNN
import TransitFinderFunctions as tff
import pandas as pd

# For future prooding keeping the load model functions seperate
def get_model_RNN( model_path, device):
    '''
    Load the RNN model
    @params
    model_name: str -> the name of the model to load
    model_path: str -> the path to the model file
    device: str -> the device to run the model on
    @returns
    model: torch.nn.Module -> the loaded model
    '''
    model = load_model_RNN(model_path, device)
    return model

def predict_model_output(flux_tensor, model):
    model.eval()
    with torch.no_grad():
        output , _= model(flux_tensor)
    return output


def generate_light_curves(
    num_systems=10, max_planets_per_system=6, total_time=365, cadence=0.0208333
):
    '''
    Generate synthetic light curves
    @params
    num_systems: int -> the number of systems to generate
    max_planets_per_system: int -> the maximum number of planets per system
    total_time: int -> the total time to generate light curves for
    cadence: float -> the cadence of the light curves
    @returns
    light_curve: dict -> the generated light curve
    '''
    system = generate_random_planet_systems(
        num_systems, max_planets_per_system, total_time, False
    )
    (
        time_array,
        flux_with_noise,
        combined_light_curve,
        total_time,
        _,
        _,
        _,
        _,
        planets,
        num_detectable_planets,
        total_planets,
    ) = process_system(
        system[0], snr_threshold=1, total_time=total_time, cadence=cadence
    )

    light_curve = {
        "time_array": time_array,
        "flux_with_noise": flux_with_noise,
        "combined_light_curve": combined_light_curve,
        "planets": planets,
        "true_num_planets": num_detectable_planets,
        "total_planets": total_planets,
    }
    return light_curve


def analyze_light_curves(light_curve,power, model, device):
    '''
    Analyze the light curves
    @params
    light_curve: dict -> the light curve to analyze
    model: torch.nn.Module -> the model to use for analysis
    device: str -> the device to run the model on
    @returns
    results: dict -> the results of the analysis
    '''
    model.eval()
    # Initialize normalized power array
    power = torch.tensor(power, dtype=torch.float32)

    # Normalize power to range [0, 1]
    min_power = power.min()
    max_power = power.max()
    if max_power - min_power > 0:  # Avoid division by zero
        power = (power - min_power) / (max_power - min_power)

    # Reshape power to the desired shape [batch_size, 5000, 1]
    power = power.unsqueeze(0).unsqueeze(2).repeat(128, 1, 1)


    print(power.shape)
    predicted_periods = predict_model_output(power, model)

    true_periods = [planet["period"] for planet in light_curve["planets"]]

    results = {
        "true_periods": true_periods,
        "predicted_periods": predicted_periods,
        "time_array": light_curve["time_array"],
        "flux_with_noise": light_curve["flux_with_noise"],
        "true_num_planets": light_curve["true_num_planets"],
    }
    return results


def plot_light_curves(results,power,time):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results["time_array"], results["flux_with_noise"], color="dodgerblue")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    ax.set_title("Synthetic Lightcurve")


    txt = " Fig.3 - Synthetic Lightcurve. \n This is a synthetic lightcurve generated using the attached script. \n Rerun the code for a new lightcurve."

    fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment="center", fontsize=10)

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().tick_params(axis='both', which='both', direction='in')
    plt.subplots_adjust(bottom=0.17)  # Adjust the bottom margin to add a gap
    plt.show()

    plt.plot(power, time)
    plt.xlabel("Period (days)")
    plt.ylabel("Power")
    plt.title("Lomb-Scargle Periodogram")
    plt.show()



def analyze_and_plot_kepler(kepler_dataframe, model, device):
    time_array = kepler_dataframe["time"].values
    flux_array = kepler_dataframe["flux"].values
    length_of_input = np.arange(0, 1600, 0.0208333)

    flux_tensor = (
        torch.tensor(
            np.interp(length_of_input, time_array, flux_array, left=1, right=1),
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .to(device)
    )

    predicted_periods, predicted_num_planets = predict_model_output(flux_tensor, model)

    results = {
        "predicted_num_planets": predicted_num_planets,
        "predicted_periods": predicted_periods,
    }
    return results

def print_comparison_results(ml_results):
    """
    Display a comparison of actual and predicted periods and number of planets for RNN and CNN models.
    @params
    ml_results: dict -> a dictionary containing the results of the model predictions.
    """

    period_data_real = []

    model_results = ml_results
    true_periods = model_results.get('true_periods', [])
    predicted_periods = model_results.get('predicted_periods', [[]])

    for i in range(max(len(true_periods), len(predicted_periods[0]))):
        true_period = true_periods[i] if i < len(true_periods) else np.nan
        predicted_period = (predicted_periods[0][i].item() * 400) if i < len(predicted_periods[0]) else None

        period_data_real.append([true_period, predicted_period])

    # Sort the period_data_real list by the predicted periods
    period_data_real.sort(key=lambda x: (x[1] if x[1] is not None else float('inf')))

    period_df_real = pd.DataFrame(period_data_real, columns=["Actual Period", "Predicted Period"])

    def make_pretty(styler):
        styler.format(precision=3, thousands=",", decimal=".")
        return styler

    styled_period_df_real = period_df_real.style.pipe(make_pretty)

    display(styled_period_df_real)
    print("Table 1. Comparison of Actual and Predicted Periods for Real Data")



def main(
    model_path="0best_model_stuf.pth",
    max_planets_in_system=5,
):
    '''
    The main function to run the analysis
    @params
    model_path: str -> the path to the model file.
    model_name: str -> the name of the model to load.
    comparison_model_path: str -> the path to the comparison model file.
    generate_light_curve: bool -> whether to generate a light curve or use Kepler data.
    max_planets_in_system: int -> the maximum number of planets in a system.
    @returns
    results: dict -> the results of the analysis.
    results_comparison: dict -> the results of the comparison analysis. (sometimes)
    '''
    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == 'cuda':
        torch.cuda.empty_cache()

    model = get_model_RNN( model_path, device)

    light_curve = generate_light_curves(
        num_systems=1,
        max_planets_per_system=max_planets_in_system,
        total_time=1600,
    )

    # use the fourier transform to get the lightcurve in the right format


    time , power = run_lomb_scargle_analysis(light_curve['time_array'], light_curve['flux_with_noise'],resolution=5000,period_range=(1,200) ,plot=True,double_lomb_scargle=False)

    results = analyze_light_curves(light_curve, power, model, device)

    plot_light_curves(results,power,time)
    results["is_kepler"] = False



    return results


if __name__ == "__main__":
    main()
