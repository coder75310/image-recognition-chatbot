// static/script.js

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const imageUpload = document.getElementById('imageUpload');
    const questionInput = document.getElementById('questionInput');
    const sendButton = document.getElementById('sendButton');
    const chatBox = document.getElementById('chatBox');
    const imagePreview = document.getElementById('imagePreview');
    const loadingIndicator = document.getElementById('loadingIndicator');

    let uploadedImageFile = null;

    // --- Event Listeners ---

    // Handle image selection
    imageUpload.addEventListener('change', () => {
        const file = imageUpload.files[0];
        if (file) {
            uploadedImageFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.alt = "Image preview";
            };
            reader.readAsDataURL(file);
            
            // Enable the input field and button
            questionInput.disabled = false;
            sendButton.disabled = false;
            questionInput.placeholder = "Great! Now ask a question.";
        }
    });

    // Handle form submission (sending question and image)
    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const question = questionInput.value.trim();
        if (!question || !uploadedImageFile) {
            return;
        }

        // Display user's question in the chat
        appendMessage('user', question);
        questionInput.value = ''; // Clear input field
        toggleLoading(true);

        // Create form data to send
        const formData = new FormData();
        formData.append('image', uploadedImageFile);
        formData.append('question', question);

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Network response was not ok');
            }

            const data = await response.json();
            // Display bot's answer
            appendMessage('bot', data.answer);

        } catch (error) {
            console.error('Error:', error);
            appendMessage('bot', 'Sorry, an error occurred: ${error.message}');
        } finally {
            toggleLoading(false);
        }
    });

    // --- Helper Functions ---

    /**
     * Appends a message to the chat box.
     * @param {string} sender - 'user' or 'bot'.
     * @param {string} text - The message content.
     */
    function appendMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', sender);

        const p = document.createElement('p');
        p.textContent = text;
        messageDiv.appendChild(p);
        
        chatBox.appendChild(messageDiv);
        // Scroll to the latest message
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    /**
     * Shows or hides the loading indicator.
     * @param {boolean} show - True to show, false to hide.
     */
    function toggleLoading(show) {
        if (show) {
            loadingIndicator.classList.remove('hidden');
            sendButton.disabled = true;
            questionInput.disabled = true;
        } else {
            loadingIndicator.classList.add('hidden');
            sendButton.disabled = false;
            questionInput.disabled = false;
        }
    }
});