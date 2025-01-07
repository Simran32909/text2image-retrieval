document.getElementById('searchInput').addEventListener('input', async function() {
    const userInput = this.value;
    const suggestionsBox = document.getElementById('suggestionsBox');
    suggestionsBox.innerHTML = ''; // Clear previous suggestions

    if (!userInput.trim()) {
        suggestionsBox.style.display = 'none'; // Hide if input is empty
        return;
    }

    // Fetch suggestions from the backend
    const response = await fetch('/get_suggestions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userInput }),
    });
    const data = await response.json();

    if (data.suggestions && data.suggestions.length > 0) {
        suggestionsBox.style.display = 'block'; // Show the suggestions box
        data.suggestions.forEach((suggestion) => {
            const suggestionItem = document.createElement('div');
            suggestionItem.textContent = suggestion;
            suggestionItem.className = 'suggestion-item';
            suggestionItem.onclick = () => {
                // Fill the input with the clicked suggestion
                document.getElementById('searchInput').value = suggestion;
                suggestionsBox.innerHTML = ''; // Clear suggestions after selection
                suggestionsBox.style.display = 'none'; // Hide the suggestions box
            };
            suggestionsBox.appendChild(suggestionItem);
        });
    } else {
        suggestionsBox.style.display = 'none'; // Hide the suggestions box if no suggestions
    }
});

document.getElementById('retrieveBtn').addEventListener('click', async () => {
    const userInput = document.getElementById('searchInput').value;
    const resultsDiv = document.getElementById('results');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // Show the loading spinner
    loadingSpinner.style.display = 'block';

    resultsDiv.innerHTML = '<p>Loading...</p>';

    // Check if user input is empty
    if (!userInput.trim()) {
        alert('Please enter some text!');
        loadingSpinner.style.display = 'none'; // Hide the spinner in case of empty input
        return;
    }

    try {
        // Fetch results from the backend
        const response = await fetch('/retrieve_images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: userInput }),
        });

        // Hide the spinner once data is fetched
        loadingSpinner.style.display = 'none';

        const data = await response.json();

        // Clear the previous results
        resultsDiv.innerHTML = '';

        if (data.length > 0) {
            // Append results
            data.forEach((result) => {
                const img = document.createElement('img');
                img.src = result.image_url;
                img.style.maxWidth = '100%';
                img.style.margin = '10px 0';
                resultsDiv.appendChild(img);

                const caption = document.createElement('p');
                caption.textContent = `Similarity: ${result.similarity.toFixed(4)}`;
                resultsDiv.appendChild(caption);
            });
        } else {
            resultsDiv.textContent = 'No matching images found.';
        }
    } catch (error) {
        // Handle any errors during fetch
        loadingSpinner.style.display = 'none';
        resultsDiv.innerHTML = '<p>Something went wrong. Please try again later.</p>';
    }
});
