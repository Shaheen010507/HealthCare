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
    let response = await fetch("http://192.168.1.15:5000/predict_breast_cancer", {
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
