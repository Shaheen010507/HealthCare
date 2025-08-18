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
