async function showSpinner(message = "Loading…") {
  $("#spinner-text").text(message);
  $("#spinner-overlay").fadeIn(200);
}

async function updateSpinnerText(message) {
  $("#spinner-text").text(message);
}

async function hideSpinner() {
  $("#spinner-overlay").fadeOut(200);
}
