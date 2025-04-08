// Initialize the knowledge graph visualization
let network = null;

function initializeNetwork(container) {
    const data = {
        nodes: new vis.DataSet([]),
        edges: new vis.DataSet([])
    };

    const options = {
        nodes: {
            shape: 'dot',
            size: 16,
            font: {
                size: 12
            }
        },
        edges: {
            arrows: 'to',
            smooth: {
                type: 'cubicBezier'
            }
        },
        physics: {
            stabilization: false,
            barnesHut: {
                gravitationalConstant: -80000,
                springConstant: 0.001,
                springLength: 200
            }
        }
    };

    network = new vis.Network(container, data, options);
}

// Initialize the network when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('knowledge-graph');
    initializeNetwork(container);
});

// Handle text processing form submission
document.getElementById('textForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    form.classList.add('loading');
    
    try {
        const text = document.getElementById('inputText').value;
        const formData = new FormData();
        formData.append('text', text);
        
        const response = await fetch('/process-text', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        updateKnowledgeGraph(data.triplets);
        displayProcessingResults('Knowledge Graph Built', data);
    } catch (error) {
        console.error('Error:', error);
        displayProcessingResults('Error processing text', { error: error.message });
    } finally {
        form.classList.remove('loading');
    }
});

// Handle query form submission
document.getElementById('queryForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    form.classList.add('loading');
    
    try {
        const query = document.getElementById('queryInput').value;
        const formData = new FormData();
        formData.append('query', query);
        
        const response = await fetch('/query', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        displayQueryResults(data);
    } catch (error) {
        console.error('Error:', error);
        displayQueryResults({ 
            query: document.getElementById('queryInput').value,
            answer: 'Error processing query: ' + error.message,
            sources: []
        });
    } finally {
        form.classList.remove('loading');
    }
});

// Update the knowledge graph visualization
function updateKnowledgeGraph(triplets) {
    const nodes = new Set();
    const edges = [];
    
    triplets.forEach(([subject, predicate, object]) => {
        nodes.add(subject);
        nodes.add(object);
        edges.push({
            from: subject,
            to: object,
            label: predicate
        });
    });
    
    const nodesArray = Array.from(nodes).map(node => ({
        id: node,
        label: node
    }));
    
    network.setData({
        nodes: new vis.DataSet(nodesArray),
        edges: new vis.DataSet(edges)
    });
}

// Display text processing results
function displayProcessingResults(title, data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="space-y-4">
            <h3 class="text-xl font-semibold mb-2">${title}</h3>
            <div class="bg-gray-100 p-4 rounded-lg">
                <h4 class="font-medium mb-2">Extracted Relationships:</h4>
                <ul class="list-disc pl-5 space-y-1">
                    ${data.triplets.map(([s, p, o]) => `
                        <li>${s} → ${p} → ${o}</li>
                    `).join('')}
                </ul>
            </div>
        </div>
    `;
}

// Display query results
function displayQueryResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="space-y-4">
            <h3 class="text-xl font-semibold mb-2">Query Results</h3>
            <div class="bg-blue-50 p-4 rounded-lg">
                <h4 class="font-medium text-blue-800">Question:</h4>
                <p class="mb-4">${data.query}</p>
                <h4 class="font-medium text-blue-800">Answer:</h4>
                <p class="mb-4">${data.answer}</p>
                ${data.sources.length > 0 ? `
                    <h4 class="font-medium text-blue-800">Sources:</h4>
                    <ul class="list-disc pl-5 space-y-1">
                        ${data.sources.map(source => `
                            <li class="text-sm text-gray-600">${source}</li>
                        `).join('')}
                    </ul>
                ` : ''}
            </div>
        </div>
    `;
}
