const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatHistory = document.getElementById('chat-history');

// Generate a random user ID for the session
const userId = "user_" + Math.random().toString(36).substr(2, 9);

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;

    // 1. Add User Message
    addMessage(query, 'user');
    userInput.value = '';
    userInput.disabled = true;

    // 2. Show Loading Indicator
    const loadingId = addLoadingIndicator();

    try {
        // 3. Call API
        const response = await fetch('/api/v1/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                user_id: userId
            })
        });

        // 4. Remove Loading & Add System Response
        removeMessage(loadingId);

        if (response.ok) {
            const data = await response.json();
            addSystemResponse(data);
        } else {
            // Try to get error message from response
            let errorMessage = "Sorry, something went wrong. Please try again.";
            try {
                const errorData = await response.json();
                if (errorData.detail) {
                    errorMessage = `Error: ${errorData.detail}`;
                } else if (errorData.message) {
                    errorMessage = `Error: ${errorData.message}`;
                }
            } catch (e) {
                errorMessage = `Error: HTTP ${response.status} ${response.statusText}`;
            }
            addMessage(errorMessage, 'system');
        }

    } catch (error) {
        removeMessage(loadingId);
        let errorMessage = "Network error. Please check your connection.";
        if (error.message) {
            errorMessage = `Network error: ${error.message}`;
        }
        addMessage(errorMessage, 'system');
        console.error('Error:', error);
    } finally {
        userInput.disabled = false;
        userInput.focus();
    }
});

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    // Simple text replacement for newlines
    const formattedText = text.replace(/\n/g, '<br>');

    messageDiv.innerHTML = `
        <div class="message-content">
            ${formattedText}
        </div>
    `;
    chatHistory.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv.id = 'msg-' + Date.now();
}

function addLoadingIndicator() {
    const id = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = id;
    loadingDiv.className = 'message system';
    loadingDiv.innerHTML = `
        <div class="typing-indicator">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    chatHistory.appendChild(loadingDiv);
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

function addSystemResponse(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';

    // Extract response content (new API structure)
    const response = data.response || {};
    const content = response.content || data.answer || "No response content available";

    // Parse markdown-like bold/italics (simple implementation)
    let formattedContent = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');

    let html = `<div class="message-content">${formattedContent}`;

    // Add Safety Notices
    if (data.safety_assessment && data.safety_assessment.required_disclaimers && data.safety_assessment.required_disclaimers.length > 0) {
        html += `<div class="safety-notice">
            ${data.safety_assessment.required_disclaimers.join('<br>')}
        </div>`;
    }

    // Add safety notices from response if available
    if (response.safety_notices && response.safety_notices.length > 0) {
        html += `<div class="safety-notice">
            ${response.safety_notices.join('<br>')}
        </div>`;
    }

    // Add Sources (from response.sources)
    const sources = response.sources || [];
    if (sources.length > 0) {
        html += `<div class="sources-section">
            <div class="sources-title">Sources:</div>`;
        sources.forEach(src => {
            const sourceName = src.source || src.chunk_id || 'Unknown';
            const relevance = src.relevance_score ? (src.relevance_score * 100).toFixed(0) : 'N/A';
            html += `<span class="source-item" title="Relevance: ${relevance}%">
                ${sourceName}
            </span>`;
        });
        html += `</div>`;
    }

    html += `</div>`;
    messageDiv.innerHTML = html;

    chatHistory.appendChild(messageDiv);
    scrollToBottom();
}

function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}
