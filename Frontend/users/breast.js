document.getElementById("breastForm").addEventListener("submit", async function(event) {
  event.preventDefault();

  // Collect form values
  let formData = {};
  new FormData(this).forEach((value, key) => {
    formData[key] = parseFloat(value); // convert to number
  });

  // Prepare features as a list for the backend
  let features = [
    formData.clump_thickness,
    formData.uniformity_cell_size,
    formData.uniformity_cell_shape,
    formData.marginal_adhesion,
    formData.single_epithelial_cell_size,
    formData.bare_nuclei,
    formData.bland_chromatin,
    formData.normal_nucleoli,
    formData.mitoses
  ];

  try {
    // Call Flask API
    let response = await fetch("http://127.0.0.1.:5000/predict_breast_cancer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: features })
    });

    let result = await response.json();
    let resultDiv = document.getElementById("result");
    resultDiv.style.display = "block";

    if (result.prediction === 4) { // Malignant
      resultDiv.textContent = "Prediction: Malignant (Cancer Detected)";
      resultDiv.style.color = "red";
    } else if (result.prediction === 2) { // Benign
      resultDiv.textContent = "Prediction: Benign (No Cancer Detected)";
      resultDiv.style.color = "green";
    } else {
      resultDiv.textContent = "Prediction: Unknown";
      resultDiv.style.color = "orange";
    }

  } catch (error) {
    alert("Error connecting to backend! Make sure Flask is running.");
    console.error(error);
  }
});
/*Batch Prediction*/

document.getElementById("breastBatchForm").addEventListener("submit", async function(event) {
  event.preventDefault();
  
  const fileInput = document.getElementById("csvFile");
  const file = fileInput.files[0];
  if (!file) return alert("Please select a CSV file");

  const reader = new FileReader();
  reader.onload = async function(e) {
    const text = e.target.result;
    const lines = text.trim().split("\n");
    const batch_features = lines.map(line => line.split(",").map(Number));

    try {
      const response = await fetch("http://127.0.0.1:5000/predict_breast_batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_features: batch_features })
      });

      const result = await response.json();
      const resultDiv = document.getElementById("batchResult");
      resultDiv.textContent = result.map((res, idx) => `Patient ${idx+1}: ${res}`).join("\n");

    } catch (err) {
      alert("Error connecting to backend!");
      console.error(err);
    }
  };
  reader.readAsText(file);
});
