
const form = document.getElementById("prediction-form");
const button = document.getElementById("predict-button");
const message = document.getElementById("form-message");
const resultCard = document.getElementById("result-card");

const initialResultHTML = resultCard.innerHTML;

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    message.textContent = "";

    const payload = Object.fromEntries(
        new FormData(form).entries()
    );

    button.disabled = true;
    button.textContent = "Calculating...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                result.error || "Unable to generate prediction."
            );
        }

        resultCard.innerHTML = `
            <div class="result-icon">✓</div>

            <p class="eyebrow" style="color: #27815f;">
                PREDICTION SUCCESSFULLY GENERATED
            </p>

            <h2>Estimated property value</h2>

            <div class="result-price">
                ${Number(result.predicted_price).toLocaleString("en-IN", {
            maximumFractionDigits: 2
        })}
            </div>

            <p class="result-unit">
                Price in the same unit as your training dataset
            </p>

            <div class="result-divider"></div>

            <p class="result-foot">
            </p>

            <div class="result-divider"></div>

            <div class="prediction-disclaimer">
                <strong>Disclaimer</strong>
                <p>
                    This is an estimated property value generated
                    using a trained Linear Regression model and
                    historical training data. It is not an official
                    property valuation or a guaranteed market price.
                </p>
            </div>

            <button
                type="button"
                id="new-prediction-button"
                class="new-prediction-button"
            >
                ↻ New Prediction
            </button>
        `;

    } catch (error) {
        message.textContent = error.message;

    } finally {
        button.disabled = false;
        button.innerHTML = 'Predict house price <b>→</b>';
    }
});

resultCard.addEventListener("click", (event) => {

    if (event.target.closest("#new-prediction-button")) {

        form.reset();

        message.textContent = "";

        resultCard.innerHTML = initialResultHTML;

        form.querySelector("select, input").focus();
    }
});
