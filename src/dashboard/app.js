async function sendPrediction() {
    const payload = {
        temperature_2m: parseFloat(document.getElementById("temperature_2m").value),
        relative_humidity_2m: parseFloat(document.getElementById("relative_humidity_2m").value),
        dew_point_2m: parseFloat(document.getElementById("dew_point_2m").value),
        shortwave_radiation: parseFloat(document.getElementById("shortwave_radiation").value),
        apparent_temperature: parseFloat(document.getElementById("apparent_temperature").value),

        lag_1h: parseFloat(document.getElementById("lag_1h").value),
        lag_24h: parseFloat(document.getElementById("lag_24h").value),
        lag_168h: parseFloat(document.getElementById("lag_168h").value),

        Hour: parseInt(document.getElementById("Hour").value),
        DayOfWeek: parseInt(document.getElementById("DayOfWeek").value),
        DayOfYear: parseInt(document.getElementById("DayOfYear").value),
        IsWeekend: parseInt(document.getElementById("IsWeekend").value),

        model_choice: document.getElementById("model_choice").value
    };

    const resultDiv = document.getElementById("result");
    resultDiv.innerText = "Sending request to Voltcast API...";

    try {
        // Local backend for testing
        const response = await fetch(`/predict/forecast`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errText = await response.text();
            resultDiv.innerText = `Error ${response.status}: ${errText}`;
            return;
        }

        const data = await response.json();

        resultDiv.innerText =
            `Model: ${data.model_used} | ` +
            `Horizon: ${data.forecast_time_ahead} | ` +
            `Predicted Demand: ${data.Predicted_demand_mw} MW`;

    } catch (error) {
        console.error(error);
        resultDiv.innerText = "Error: Could not reach Voltcast API. Is the backend running locally?";
    }
}
