const form = document.getElementById("predictionForm");

const predictButton = document.getElementById("predictButton");

const resultSection = document.getElementById("resultSection");


// PUBLIC BACKEND URL
const API_URL = "https://universe-promotion-overlook.ngrok-free.dev/predict";


// All 23 model input features

const features = [

    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",

    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",

    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",

    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"

];


// Handle form submission

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    const inputData = {};

    // Collect all 23 input values

    for (let feature of features) {

        const input = document.getElementById(feature);

        if (!input || input.value.trim() === "") {

            alert("Please fill in all 23 input fields.");

            return;

        }

        inputData[feature] = Number(input.value);

        if (!Number.isFinite(inputData[feature])) {

            alert("Please enter valid numerical values.");

            return;

        }

    }


    // Disable button while prediction is running

    predictButton.disabled = true;

    predictButton.textContent = "Analyzing...";


    try {

        // Send data to PUBLIC Flask backend

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {

                "Content-Type": "application/json",

                "ngrok-skip-browser-warning": "true"

            },

            body: JSON.stringify(inputData)

        });


        const result = await response.json();


        if (!response.ok) {

            throw new Error(result.error || "Prediction failed.");

        }


        // Display prediction results

        displayResults(result);


    }

    catch (error) {

        console.error("Prediction Error:", error);

        alert(
            "Unable to connect to the prediction server.\n\n" +
            error.message +
            "\n\nPlease check whether Flask and ngrok are running."
        );

    }


    finally {

        predictButton.disabled = false;

        predictButton.textContent = "Analyze Credit Risk →";

    }

});



// Display results on website

function displayResults(result) {

    const probability = result.default_probability;

    const prediction = result.prediction;


    document.getElementById("probabilityValue").textContent =
        probability + "%";


    document.getElementById("predictionValue").textContent =
        result.result;


    document.getElementById("thresholdValue").textContent =
        (result.threshold * 100).toFixed(0) + "%";


    document.getElementById("modelValue").textContent =
        result.model;


    document.getElementById("probabilityFill").style.width =
        probability + "%";


    const badge = document.getElementById("resultBadge");


    if (prediction === 1) {

        badge.textContent = "Potential Default";

        badge.className = "result-badge danger";

    }

    else {

        badge.textContent = "No Default";

        badge.className = "result-badge safe";

    }


    // Show results

    resultSection.hidden = false;


    // Scroll to results

    resultSection.scrollIntoView({

        behavior: "smooth",

        block: "start"

    });

}



// Clear previous results when form is reset

form.addEventListener("reset", function () {

    resultSection.hidden = true;

});