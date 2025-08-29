/*function signup() {
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
// ----------------- Google Sign-In -----------------
function googleLogin() {
  const provider = new firebase.auth.GoogleAuthProvider();
  auth.signInWithPopup(provider)
    .then((result) => {
      const user = result.user;
      console.log("Google Login Success:", user);
      window.location.href = "users/user_page.html";
    })
    .catch((error) => {
      alert("Google Login Failed: " + error.message);
    });
}

// ----------------- Facebook Sign-In -----------------
function facebookLogin() {
  const provider = new firebase.auth.FacebookAuthProvider();
  auth.signInWithPopup(provider)
    .then((result) => {
      const user = result.user;
      console.log("Facebook Login Success:", user);
      window.location.href = "users/user_page.html";
    })
    .catch((error) => {
      alert("Facebook Login Failed: " + error.message);
    });
}
// ----------------- Optional: Auto redirect if already logged in -----------------
auth.onAuthStateChanged((user) => {
  if (user) {
    window.location.href = "users/user_page.html";
  }
});
*/

// ---------------- Initialize auth ----------------
const auth = firebase.auth();

// ---------------- Email/Password Signup ----------------
function signup() {
  const username = document.getElementById("username").value;
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;
  const confirmPassword = document.getElementById("confirmPassword").value;

  if (password !== confirmPassword) {
    document.getElementById("signup-msg").innerText = "Passwords do not match!";
    return;
  }

  auth.createUserWithEmailAndPassword(email, password)
    .then((userCredential) => {
      return userCredential.user.updateProfile({ displayName: username });
    })
    .then(() => {
      window.location.href = "users/user_page.html";
    })
    .catch((error) => {
      document.getElementById("signup-msg").innerText = "Signup failed: " + error.message;
    });
}

// ---------------- Email/Password Login ----------------
function login() {
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  auth.signInWithEmailAndPassword(email, password)
    .then(() => {
      window.location.href = "users/user_page.html";
    })
    .catch((error) => {
      document.getElementById("login-msg").innerText = "Login failed: " + error.message;
    });
}

// ---------------- Google Sign-In ----------------
function googleLogin() {
  const provider = new firebase.auth.GoogleAuthProvider();
  auth.signInWithPopup(provider)
    .then((result) => {
      window.location.href = "users/user_page.html";
    })
    .catch((error) => {
      alert("Google Login Failed: " + error.message);
    });
}





