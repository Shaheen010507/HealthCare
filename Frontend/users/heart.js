document.getElementById("heartForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const data = {
        age: parseInt(document.getElementById("age").value),
        sex: parseInt(document.getElementById("sex").value),
        cp: parseInt(document.getElementById("cp").value),
        trestbps: parseInt(document.getElementById("trestbps").value),
        chol: parseInt(document.getElementById("chol").value),
        fbs: parseInt(document.getElementById("fbs").value),
        restecg: parseInt(document.getElementById("restecg").value),
        thalach: parseInt(document.getElementById("thalach").value),
        exang: parseInt(document.getElementById("exang").value),
        oldpeak: parseFloat(document.getElementById("oldpeak").value),
        slope: parseInt(document.getElementById("slope").value),
        ca: parseInt(document.getElementById("ca").value),
        thal: parseInt(document.getElementById("thal").value)
    };

    try {
        const response = await fetch("http://127.0.0.1:5000/predict_heart", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.prediction === 1) {
            document.getElementById("result").innerText = "⚠️ You may have heart disease.";
            document.getElementById("result").style.color = "red";
        } else {
            document.getElementById("result").innerText = "✅ You are unlikely to have heart disease.";
            document.getElementById("result").style.color = "green";
        }
    } catch (error) {
        document.getElementById("result").innerText = "❌ Error: " + error.message;
        document.getElementById("result").style.color = "red";
    }
});
/* ---------------- Batch CSV Upload ---------------- */
document.getElementById("batchHeartForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const fileInput = document.getElementById("heartFileInput");
    if (!fileInput.files.length) {
        alert("Please select a file first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("http://localhost:5000/predict_heart_batch", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        displayHeartBatchResults(result);
    } catch (error) {
        document.getElementById("batchHeartResult").innerHTML =
          `<p style="color:red;">Error: ${error.message}</p>`;
    }
});

function displayHeartBatchResults(result) {
    if (result.error) {
        document.getElementById("batchHeartResult").innerHTML =
          `<p style="color:red;">Error: ${result.error}</p>`;
        return;
    }

    let output = "<h3>Batch Prediction Results:</h3><ul>";
    result.predictions.forEach((pred, index) => {
        // Determine text and color based on prediction
        let text = pred.prediction === 1 ? "Heart Disease Detected" : "No Heart Disease";
        let color = pred.prediction === 1 ? "red" : "green";

        output += `<li style="color:${color}">Patient ${index + 1}: ${text}</li>`;
    });
    output += "</ul>";

    document.getElementById("batchHeartResult").innerHTML = output;
}
