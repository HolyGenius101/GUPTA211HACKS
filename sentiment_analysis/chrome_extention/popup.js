document.getElementById("analyzeBtn").addEventListener("click", async () => {
    const text = document.getElementById("textInput").value.trim();
    const resultDiv = document.getElementById("result");
  
    if (!text) {
      resultDiv.textContent = "Please enter some text.";
      return;
    }
  
    try {
      const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
      });
  
      if (!response.ok) {
        throw new Error("Server error");
      }
  
      const data = await response.json();
      resultDiv.textContent = `Sentiment: ${data.sentiment} (Polarity: ${data.polarity.toFixed(2)})`;
    } catch (error) {
      resultDiv.textContent = "Error analyzing text. Is the server running?";
      console.error("Error:", error);
    }
  });
  