
// Handles navigation to different pages
function navigate(page) {
  switch (page) {
    case 'dashboard':
      window.location.href = 'dashboard.html';
      break;
    case 'profile':
      window.location.href = 'profile.html';
      break;
    case 'diabetes':
      window.location.href = 'diabetics.html';
      break;
    case 'heart':
      window.location.href = 'heart.html';
      break;
    case 'liver':
      window.location.href = 'liver.html';
      break;
    case 'kidney':
      window.location.href = 'kidney.html';
      break;
    case 'cancer':
      window.location.href = 'cancer.html';
      break;
    default:
      alert('Page not found!');
  }
}

// Logs out the user
function logout() {
  // Example Firebase logout
  firebase.auth().signOut().then(() => {
    alert("Logged out successfully!");
    window.location.href = 'index.html'; // Redirect to home page after logout
  }).catch((error) => {
    console.error("Logout error:", error);
    alert("Failed to log out. Please try again.");
  });
}
