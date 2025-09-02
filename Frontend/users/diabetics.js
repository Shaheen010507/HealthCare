
/*document.getElementById("diabetesForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    // Get values from form
    const data = {
        Pregnancies: parseInt(document.getElementById("pregnancies").value) || 0,
        Glucose: parseInt(document.getElementById("glucose").value) || 0,
        BloodPressure: parseInt(document.getElementById("bloodPressure").value) || 0,
        SkinThickness: parseInt(document.getElementById("skinThickness").value) || 0,
        Insulin: parseInt(document.getElementById("insulin").value) || 0,
        BMI: parseFloat(document.getElementById("bmi").value) || 0.0,
        DiabetesPedigreeFunction: parseFloat(document.getElementById("dpf").value) || 0.0,
        Age: parseInt(document.getElementById("age").value) || 0
    };

    // Basic check for required fields
    if (data.Glucose === 0 || data.Age === 0) {
        document.getElementById("result").textContent = "Please enter Glucose and Age.";
        return;
    }

    try {
        const response = await fetch("http://127.0.0.1:5000/predict_diabetes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.prediction !== undefined) {
            document.getElementById("result").textContent = 
                result.prediction === 1 ? "You may have diabetes." : "You are unlikely to have diabetes.";
        } else {
            document.getElementById("result").textContent = "Error: " + result.error;
        }
    } catch (error) {
        document.getElementById("result").textContent = "Failed to connect to server.";
    }
});*/
document.getElementById("diabetesForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const data = {
        Pregnancies: parseInt(document.getElementById("pregnancies").value) || 0,
        Glucose: parseInt(document.getElementById("glucose").value) || 0,
        BloodPressure: parseInt(document.getElementById("bloodPressure").value) || 0,
        SkinThickness: parseInt(document.getElementById("skinThickness").value) || 0,
        Insulin: parseInt(document.getElementById("insulin").value) || 0,
        BMI: parseFloat(document.getElementById("bmi").value) || 0.0,
        DiabetesPedigreeFunction: parseFloat(document.getElementById("dpf").value) || 0.0,
        Age: parseInt(document.getElementById("age").value) || 0
    };

    if (data.Glucose === 0 || data.Age === 0) {
        document.getElementById("result").textContent = "Please enter Glucose and Age.";
        return;
    }

    try {
        const response = await fetch("http://localhost:5000/predict_diabetes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        console.log("Server response:", result); // 👈 Debugging line

        if (result.prediction !== undefined) {
            document.getElementById("result").textContent =
                result.prediction === 1 ? "You may have diabetes." : "You are unlikely to have diabetes.";
        } else {
            document.getElementById("result").textContent = "Error: " + (result.error || "Unexpected response");
        }
    } catch (error) {
        console.error("Fetch error:", error); // 👈 Debugging line
        document.getElementById("result").textContent = "Failed to connect to server.";
    }
});
document.getElementById("batchUploadForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Please select a file first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("http://localhost:5000/predict_batch_diabetes", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        document.getElementById("batchResult").innerHTML =
          `<p style="color:red;">Error: ${error.message}</p>`;
    }
});

/* ---------------- Batch Upload ---------------- */
document.getElementById("batchUploadForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Please select a file first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("http://localhost:5000/predict_diabetes_batch", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        displayBatchResults(result);
    } catch (error) {
        document.getElementById("batchResult").innerHTML =
          `<p style="color:red;">Error: ${error.message}</p>`;
    }
});

function displayBatchResults(result) {
    if (result.error) {
        document.getElementById("batchResult").innerHTML =
          `<p style="color:red;">Error: ${result.error}</p>`;
        return;
    }

    let output = "<h3>Batch Prediction Results:</h3><ul>";
    result.predictions.forEach((pred, index) => {
        output += `<li>Patient ${index + 1}: ${pred.result}</li>`;
    });
    output += "</ul>";

    document.getElementById("batchResult").innerHTML = output;
}
