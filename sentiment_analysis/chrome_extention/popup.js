document.addEventListener("DOMContentLoaded", function () {
    const analyzeBtn = document.getElementById("analyzeBtn");
    const inputField = document.getElementById("inputText");
    const resultBox = document.getElementById("result");
  
    analyzeBtn.addEventListener("click", async () => {
      const userInput = inputField.value.trim();
  
      if (!userInput) {
        resultBox.textContent = "Please enter some text.";
        return;
      }
  
      resultBox.textContent = "Analyzing...";
  
      try {
        const response = await fetch("http://127.0.0.1:5000/analyze", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: userInput }),
        });
  
        const data = await response.json();
  
        if (data.error) {
          resultBox.textContent = "Error: " + data.error;
        } else {
          resultBox.innerHTML = `
            <strong>Sentiment:</strong> ${data.sentiment}<br/>
            <strong>Confidence:</strong> ${data.score}
          `;
        }
      } catch (err) {
        resultBox.textContent = "Could not reach the server. Is it running?";
        console.error(err);
      }
    });
  });
  