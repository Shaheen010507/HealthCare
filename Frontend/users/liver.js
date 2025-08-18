const API_URL = "http://127.0.0.1:5000/predict_liver"; // change if backend URL differs

function showError(msg){
  const el = document.getElementById("result");
  el.style.color = "red";
  el.textContent = msg;
}

function showSuccess(msg){
  const el = document.getElementById("result");
  el.style.color = "green";
  el.textContent = msg;
}

document.getElementById("liverForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  // collect values
  const data = {
    age: Number(document.getElementById("age").value),
    gender: Number(document.getElementById("gender").value), // 1 or 0
    total_bilirubin: Number(document.getElementById("total_bilirubin").value),
    direct_bilirubin: Number(document.getElementById("direct_bilirubin").value),
    alkaline_phosphotase: Number(document.getElementById("alkaline_phosphotase").value),
    alamine_aminotransferase: Number(document.getElementById("alamine_aminotransferase").value),
    aspartate_aminotransferase: Number(document.getElementById("aspartate_aminotransferase").value),
    total_proteins: Number(document.getElementById("total_proteins").value),
    albumin: Number(document.getElementById("albumin").value),
    albumin_and_globulin_ratio: Number(document.getElementById("albumin_and_globulin_ratio").value)
  };

  // client-side validation (simple)
  if (!Number.isFinite(data.age) || data.age < 1 || data.age > 120) return showError("Age must be 1–120.");
  if (![0,1].includes(data.gender)) return showError("Select gender (Male/Female).");
  if (!(data.total_bilirubin >= 0)) return showError("Total bilirubin must be >= 0.");
  if (!(data.direct_bilirubin >= 0)) return showError("Direct bilirubin must be >= 0.");
  if (!(data.alkaline_phosphotase >= 0)) return showError("Alkaline phosphotase must be >= 0.");
  if (!(data.alamine_aminotransferase >= 0)) return showError("ALT must be >= 0.");
  if (!(data.aspartate_aminotransferase >= 0)) return showError("AST must be >= 0.");
  if (!(data.total_proteins >= 0)) return showError("Total proteins must be >= 0.");
  if (!(data.albumin >= 0)) return showError("Albumin must be >= 0.");
  if (!(data.albumin_and_globulin_ratio >= 0)) return showError("Albumin & globulin ratio must be >= 0.");

  // show loading
  const resultEl = document.getElementById("result");
  resultEl.style.color = "#333";
  resultEl.textContent = "Predicting…";

  try{
    const res = await fetch(API_URL, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(data)
    });

    const json = await res.json();

    if (!res.ok) {
      showError(json.error || "Server error");
      return;
    }

    if (json.prediction === 1){
      // disease
      resultEl.style.color = "red";
      resultEl.textContent = "⚠️ Likely liver disease — consult a doctor.";
    } else {
      resultEl.style.color = "green";
      resultEl.textContent = "✅ Low risk of liver disease.";
    }
  }catch(err){
    showError("Network / server error. Make sure backend is running.");
    console.error(err);
  }
});
