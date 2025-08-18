
document.getElementById("diabetesForm").addEventListener("submit", async function(e) {
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
});
