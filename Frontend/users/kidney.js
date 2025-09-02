

document.getElementById("kidneyForm").addEventListener("submit", function(e) {
  e.preventDefault();

  let formData = new FormData(e.target);
  let data = {};
  formData.forEach((value, key) => {
    data[key] = value;
  });

  console.log("Form Data:", data);

  // --- Dummy Prediction Logic (replace with API later) ---
  let riskScore = 0;

  if (data.bp < 90 || data.bp > 140) riskScore++;
  if (data.sg < 1.01) riskScore++;
  if (data.al > 2) riskScore++;
  if (data.su > 1) riskScore++;
  if (data.rbc == "1" || data.pc == "1") riskScore++;
  if (data.htn == "1" || data.dm == "1") riskScore++;
  if (data.ane == "1") riskScore++;

  let resultText = riskScore >= 3 ? "⚠️ High Risk of CKD" : "✅ Low Risk of CKD";

  document.getElementById("result").innerText = resultText;
});
/*---------------- Kidney Batch CSV Upload ---------------- */
/*document.getElementById("kidneyBatchForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const fileInput = document.getElementById("kidneyFileInput");
    if (!fileInput.files.length) {
        alert("Please select a CSV file first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("http://localhost:5000/predict_kidney_batch", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        displayKidneyResults(result);

    } catch (error) {
        document.getElementById("kidneyBatchResult").innerHTML =
            `<p style="color:red;">Error: ${error.message}</p>`;
    }
});

function displayKidneyResults(result) {
    if (!result.predictions) {
        document.getElementById("kidneyBatchResult").innerHTML =
            `<p style="color:red;">Server Error: ${result.error || "Unknown error"}</p>`;
        return;
    }

    let output = "<h4>Batch Prediction Results:</h4><ul>";
    result.predictions.forEach((pred, index) => {
        output += `<li>Patient ${index + 1}: ${pred.result}</li>`;
    });
    output += "</ul>";
    document.getElementById("kidneyBatchResult").innerHTML = output;
}
*/

document.getElementById("kidneyBatchForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const fileInput = document.getElementById("kidneyFileInput");
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("/predict_kidney_batch", {
            method: "POST",
            body: formData
        });

        const data = await response.json(); // ✅ Will fail if Flask sends HTML

        if (data.error) {
            alert("Error: " + data.error);
        } else {
            alert("Batch Prediction Results:\n" + data.batch_results.join("\n"));
        }
    } catch (err) {
        alert("Error parsing response: " + err);
    }
});
