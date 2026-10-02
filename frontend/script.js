const ticketInput = document.getElementById("ticket");
const characterCount = document.getElementById("characterCount");
const predictBtn = document.getElementById("predictBtn");
const resultCard = document.getElementById("resultCard");

const PAGES = {
    predict: {
        title: "Adaptive Ticket Routing",
        subtitle: "Automatically classify support tickets and detect unfamiliar issues.",
    },
    categories: {
        title: "Categories",
        subtitle: "Support categories the model is currently trained on.",
    },
    status: {
        title: "Model Status",
        subtitle: "Model, feature pipeline and file information.",
    },
};

// The ticket that produced the current OUTSIDER result
let lastTicket = "";


// =====================================================
// HELPERS
// =====================================================

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function fmt(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(4) : "N/A";
}

async function api(url, options) {
    const response = await fetch(url, options);
    let data = null;
    try {
        data = await response.json();
    } catch (e) {
        data = null;
    }
    if (!response.ok) {
        throw new Error((data && data.error) || `Request failed (${response.status}).`);
    }
    return data;
}

function loadingBlock(title, text) {
    return `
        <div class="empty">
            <div class="spinner"></div>
            <h3>${escapeHtml(title)}</h3>
            <p>${escapeHtml(text)}</p>
        </div>`;
}

function errorBlock(message) {
    return `
        <div class="empty">
            <div class="empty-icon err">!</div>
            <h3>Something went wrong</h3>
            <p>${escapeHtml(message)}</p>
        </div>`;
}

const pipelineHtml = `
    <div class="pipeline">
        <div><span>Classifier</span><strong>Linear SVM</strong></div>
        <div><span>Features</span><strong>TF-IDF</strong></div>
        <div><span>Experiment Tracking</span><strong>MLflow</strong></div>
        <div><span>Data Versioning</span><strong>DVC</strong></div>
    </div>`;

const metricsHtml = (r) => `
    <div class="metrics">
        <div class="metric"><span>Confidence</span><strong>${fmt(r.confidence)}</strong></div>
        <div class="metric"><span>Similarity</span><strong>${fmt(r.similarity)}</strong></div>
    </div>`;


// =====================================================
// NAVIGATION
// =====================================================

function showView(name) {
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    document.getElementById(`view-${name}`).classList.add("active");

    document.querySelectorAll(".nav-item").forEach((n) => {
        n.classList.toggle("active", n.dataset.view === name);
    });

    document.getElementById("pageTitle").textContent = PAGES[name].title;
    document.getElementById("pageSubtitle").textContent = PAGES[name].subtitle;

    if (name === "categories") loadCategories();
    if (name === "status") loadStatus();
}

document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => showView(item.dataset.view));
});


// =====================================================
// CHARACTER COUNT
// =====================================================

ticketInput.addEventListener("input", () => {
    const length = ticketInput.value.length;
    characterCount.textContent = `${length} character${length === 1 ? "" : "s"}`;
});


// =====================================================
// PREDICT TICKET  ->  POST /api/predict
// =====================================================

async function predictTicket() {
    const ticket = ticketInput.value.trim();

    if (!ticket) {
        resultCard.innerHTML = errorBlock("Please enter a customer ticket before predicting.");
        ticketInput.focus();
        return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = "Analyzing...";
    resultCard.innerHTML = loadingBlock("Analyzing ticket", "The model is classifying your support request.");

    try {
        const result = await api("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticket }),
        });

        if (result.status === "KNOWN") {
            renderKnown(result);
        } else if (result.status === "OUTSIDER") {
            lastTicket = ticket;
            renderOutsider(result);
        } else {
            throw new Error("Unexpected response from the prediction API.");
        }
    } catch (error) {
        console.error("Prediction error:", error);
        resultCard.innerHTML = errorBlock(error.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Predict Ticket";
    }
}

function renderKnown(result) {
    resultCard.innerHTML = `
        <div class="result-header">
            <div>
                <h3>Prediction Result</h3>
                <p>This ticket matches an existing support category.</p>
            </div>
            <span class="badge known">✓ KNOWN CATEGORY</span>
        </div>
        <div class="category-box">
            <span class="label">Predicted Category</span>
            <span class="value">${escapeHtml(result.predicted_queue)}</span>
        </div>
        ${metricsHtml(result)}
        ${pipelineHtml}`;
}

function renderOutsider(result) {
    resultCard.innerHTML = `
        <div class="result-header">
            <div>
                <h3>Prediction Result</h3>
                <p>Review this ticket and decide how to route it.</p>
            </div>
            <span class="badge outsider">⚠ OUTSIDER TICKET</span>
        </div>

        <div class="outsider-panel">
            <p>This ticket does not sufficiently match an existing support category.</p>
        </div>

        ${metricsHtml(result)}

        <div class="create-box">

            <h4>How would you like to route this ticket?</h4>

            <div class="routing-option">
                <label for="existingCategory">
                    Add to Existing Category
                </label>

                <select id="existingCategory" class="category-select">
                    <option value="">Select an existing category</option>
                </select>
            </div>

            <div class="routing-or">
                <span>OR</span>
            </div>

            <div class="routing-option">
                <label for="newCategory">
                    Create New Category
                </label>

                <input
                    id="newCategory"
                    type="text"
                    placeholder="Enter new category name"
                    maxlength="80"
                >
            </div>

            <div class="create-row">
                <button id="createBtn" class="btn warn">
                    Add &amp; Retrain
                </button>
            </div>

            <div id="categoryLoadError"></div>
            <div id="createError"></div>

        </div>
    `;

    const existingSelect = document.getElementById("existingCategory");
    const newInput = document.getElementById("newCategory");
    const createBtn = document.getElementById("createBtn");

    // Load existing categories into dropdown
    fetchCategories()
        .then((categories) => {
            categories.forEach((category) => {
                const option = document.createElement("option");

                option.value = category.name;
                option.textContent = category.name;

                existingSelect.appendChild(option);
            });
        })
        .catch((error) => {
            console.error("Category loading error:", error);

            document.getElementById("categoryLoadError").innerHTML = `
                <div class="error-inline">
                    Unable to load existing categories. You can still create a new category.
                </div>
            `;
        });

    // Existing category selected -> disable new category input
    existingSelect.addEventListener("change", () => {
        if (existingSelect.value) {
            newInput.value = "";
            newInput.disabled = true;
        } else {
            newInput.disabled = false;
        }
    });

    // New category typed -> disable existing category dropdown
    newInput.addEventListener("input", () => {
        if (newInput.value.trim()) {
            existingSelect.value = "";
            existingSelect.disabled = true;
        } else {
            existingSelect.disabled = false;
        }
    });

    createBtn.addEventListener("click", createCategory);

    newInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            createCategory();
        }
    });
}

// =====================================================
// CREATE CATEGORY  ->  POST /api/create-category
// =====================================================

async function createCategory() {
    const input = document.getElementById("newCategory");
    const select = document.getElementById("existingCategory");
    const errorBox = document.getElementById("createError");

    const newCategory = input.value.trim();
    const existingCategory = select.value.trim();

    errorBox.innerHTML = "";

    // -------------------------------------------------
    // Validate routing choice
    // -------------------------------------------------

    if (!existingCategory && !newCategory) {
        errorBox.innerHTML = `
            <div class="error-inline">
                Select an existing category or enter a new category name.
            </div>
        `;
        return;
    }

    // Prevent both options
    if (existingCategory && newCategory) {
        errorBox.innerHTML = `
            <div class="error-inline">
                Please choose either an existing category or create a new category.
            </div>
        `;
        return;
    }

    const category = existingCategory || newCategory;
    const isExistingCategory = Boolean(existingCategory);

    const createBtn = document.getElementById("createBtn");

    createBtn.disabled = true;
    createBtn.textContent = "Retraining...";

    resultCard.innerHTML = loadingBlock(
        "Retraining model...",
        isExistingCategory
            ? "Adding the ticket to the selected category and retraining the SVM."
            : "Creating the new category and retraining the SVM."
    );

    try {
        const result = await api("/api/create-category", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                ticket: lastTicket,
                category: category
            }),
        });

        const name = result.category || category;
        const count =
            result.training_examples ??
            result.training_count ??
            result.count;

        const addedToExisting =
            result.status === "ADDED_TO_EXISTING";

        if (addedToExisting) {

            resultCard.innerHTML = `
                <div class="result-header">
                    <div>
                        <h3>Ticket Added to Existing Category</h3>
                        <p>Ticket added to existing category and model retrained.</p>
                    </div>
                    <span class="badge known">✓ SUCCESS</span>
                </div>

                <div class="category-box">
                    <span class="label">Category</span>
                    <span class="value">${escapeHtml(name)}</span>
                </div>

                <ul class="success-list">
                    <li>✓ Ticket added to existing category</li>
                    <li>✓ Model retrained successfully</li>
                </ul>

                ${count !== undefined
                    ? `
                    <div class="metrics">
                        <div class="metric">
                            <span>Training Examples</span>
                            <strong>${escapeHtml(count)}</strong>
                        </div>
                    </div>
                    `
                    : ""
                }
            `;

        } else {

            resultCard.innerHTML = `
                <div class="result-header">
                    <div>
                        <h3>New Category Created</h3>
                        <p>New category created and model retrained.</p>
                    </div>
                    <span class="badge known">✓ SUCCESS</span>
                </div>

                <div class="category-box">
                    <span class="label">Category</span>
                    <span class="value">${escapeHtml(name)}</span>
                </div>

                <ul class="success-list">
                    <li>✓ New category created</li>
                    <li>✓ Model retrained successfully</li>
                </ul>

                ${count !== undefined
                    ? `
                    <div class="metrics">
                        <div class="metric">
                            <span>Training Examples</span>
                            <strong>${escapeHtml(count)}</strong>
                        </div>
                    </div>
                    `
                    : ""
                }
            `;
        }

    } catch (error) {
        console.error("Category routing error:", error);

        resultCard.innerHTML = errorBlock(error.message);
    }
}


// =====================================================
// CATEGORIES  ->  GET /api/categories
// =====================================================
async function fetchCategories() {
    const data = await api("/api/categories");
    return data.categories || [];
}

async function loadCategories() {
    const box = document.getElementById("categoriesContainer");
    box.innerHTML = `<div class="card pad">${loadingBlock("Loading categories", "Reading training data.")}</div>`;

    try {
        const items = await fetchCategories();

        if (!items.length) {
            box.innerHTML = `
                <div class="card pad">
                    <div class="empty">
                        <h3>No categories found</h3>
                        <p>The training data has no queue values.</p>
                    </div>
                </div>`;
            return;
        }

        const max = Math.max(...items.map((c) => c.count));
        const total = items.reduce((sum, c) => sum + c.count, 0);

        box.innerHTML = `
            <p class="cat-summary">
                ${items.length} categories · ${total.toLocaleString()} training examples
            </p>

            <div class="cat-grid">
                ${items.map((c) => `
                    <div class="cat-card">

                        <div class="cat-info">
                            <strong>${escapeHtml(c.name)}</strong>
                            <span>${Number(c.count).toLocaleString()} training examples</span>

                            <div class="cat-bar">
                                <div style="width:${(c.count / max) * 100}%"></div>
                            </div>
                        </div>

                        <button 
                            class="cat-delete"
                            onclick="deleteCategory('${escapeHtml(c.name).replace(/'/g, "\\'")}', this)">
                            Delete
                        </button>

                    </div>
                `).join("")}
            </div>`;
    } catch (error) {
        console.error("Categories error:", error);
        box.innerHTML = `<div class="card pad">${errorBlock(error.message)}</div>`;
    }
}

async function deleteCategory(category, button) {
    const confirmed = confirm(
        `Are you sure you want to delete the "${category}" category?`
    );

    if (!confirmed) return;

    const card = button.closest(".cat-card");

    // Remove from UI for now
    card.remove();

    console.log(`Category deleted: ${category}`);
}

// =====================================================
// MODEL STATUS  ->  GET /api/model-status
// =====================================================

function fileRow(label, f) {
    if (!f.exists) {
        return `<tr><td>${label}</td><td><code>${escapeHtml(f.path)}</code></td><td colspan="2">File not found</td></tr>`;
    }
    return `
        <tr>
            <td>${label}</td>
            <td><code>${escapeHtml(f.path)}</code></td>
            <td>${escapeHtml(f.size_kb)} KB</td>
            <td>${escapeHtml(f.last_modified)}</td>
        </tr>`;
}

async function loadStatus() {
    const box = document.getElementById("statusContainer");
    box.innerHTML = `<div class="card pad">${loadingBlock("Loading model status", "Contacting the backend.")}</div>`;

    try {
        const data = await api("/api/model-status");

        box.innerHTML = `
            <div class="status-grid">
                <div class="status-item"><span>Model</span><strong>Linear SVM</strong></div>
                <div class="status-item"><span>Features</span><strong>TF-IDF</strong></div>
                <div class="status-item"><span>Experiment Tracking</span><strong>MLflow</strong></div>
                <div class="status-item"><span>Data Versioning</span><strong>DVC</strong></div>
                <div class="status-item"><span>Status</span><strong class="ok">${data.status === "online" ? "Online" : escapeHtml(data.status)}</strong></div>
            </div>
            <div class="card pad">
                <div class="card-header"><div><h3>Files</h3><p>Read from the server's file system.</p></div></div>
                <div class="table-wrap">
                    <table class="file-table">
                        <thead><tr><th>Item</th><th>Path</th><th>Size</th><th>Last modified</th></tr></thead>
                        <tbody>
                            ${fileRow("SVM model", data.model)}
                            ${fileRow("TF-IDF vectorizer", data.vectorizer)}
                            ${fileRow("Training data", data.training_data)}
                        </tbody>
                    </table>
                </div>
            </div>`;
    } catch (error) {
        console.error("Status error:", error);
        box.innerHTML = `<div class="card pad">${errorBlock(error.message)}</div>`;
    }
}


// =====================================================
// SYSTEM INDICATOR (real check against the backend)
// =====================================================

async function checkSystem() {
    const el = document.getElementById("systemStatus");
    const text = document.getElementById("systemStatusText");
    try {
        await api("/api/model-status");
        el.className = "system-status online";
        text.textContent = "System Online";
    } catch (e) {
        el.className = "system-status offline";
        text.textContent = "System Offline";
    }
}

predictBtn.addEventListener("click", predictTicket);
checkSystem();