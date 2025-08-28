

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
