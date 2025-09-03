// Tab switching
const tabs = document.querySelectorAll(".tab-btn");
const contents = document.querySelectorAll(".tab-content");

tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    tabs.forEach(btn => btn.classList.remove("active"));
    contents.forEach(content => content.classList.remove("active"));

    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
  });
});

// Example API calls (update URLs with your Flask backend)
document.getElementById("diabetesForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const res = await fetch("http://127.0.0.1:5000/predict_diabetes", {
    method: "POST",
    body: formData
  });
  const result = await res.json();
  document.getElementById("diabetesResult").textContent = "Prediction: " + result.prediction;
});

// Repeat same for other forms (heart, liver, kidney, breast)...
