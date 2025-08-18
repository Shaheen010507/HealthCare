function signup() {
  const username = document.getElementById("username").value;
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;
  const confirmPassword = document.getElementById("confirmPassword").value;

  // 1. Check if passwords match
  if (password !== confirmPassword) {
    document.getElementById("signup-msg").innerText = "Passwords do not match!";
    return;
  }

  // 2. Create user in Firebase
  firebase.auth().createUserWithEmailAndPassword(email, password)
    .then((userCredential) => {
      const user = userCredential.user;

      // 3. Optionally update display name
      return user.updateProfile({
        displayName: username
      });
    })
    .then(() => {
      // 4. Redirect to dashboard
      window.location.href = "users/user_page.html";
    })
    .catch((error) => {
      // 5. Show error
      document.getElementById("signup-msg").innerText = "Signup failed: " + error.message;
    });
}
function login() {
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  firebase.auth().signInWithEmailAndPassword(email, password)
    .then((userCredential) => {
      // Redirect to dashboard
      window.location.href = "users/user_page.html";
    })
    .catch((error) => {
      document.getElementById("login-msg").innerText = "Login failed: " + error.message;
    });
}
