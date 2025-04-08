# Knowledge Nexus

An intelligent system for building and querying knowledge graphs from text. Knowledge Nexus uses natural language processing to extract entities and relationships from text, builds a knowledge graph, and allows users to query the graph using natural language.

## Features

- Extract entities and relationships from text
- Build and visualize knowledge graphs
- Query the knowledge graph using natural language
- Interactive web interface
- Real-time graph visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/knowledge-nexus.git
cd knowledge-nexus
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
cd src
python app.py
```

2. Open your browser and navigate to `http://localhost:8080`

3. Enter text in the "Input Document" section and click "Process Text" to build the knowledge graph

4. Use the "Query" section to ask questions about the processed text

## Example

Input text:
```
Microsoft was founded by Bill Gates and Paul Allen in 1975. They created Windows which became the world's most popular operating system. In 2014, Satya Nadella became CEO and led Microsoft into cloud computing with Azure.
```

Example queries:
- "Who is Bill Gates?"
- "What did Microsoft create?"
- "Who leads Microsoft?"

## Project Structure

```
knowledge-nexus/
├── src/
│   ├── app.py          # FastAPI web application
│   └── nexus.py        # Core Knowledge Nexus implementation
├── static/
│   ├── main.js         # Frontend JavaScript
│   └── style.css       # Custom styles
├── templates/
│   └── index.html      # Web interface template
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technologies Used

- Backend:
  - Python
  - FastAPI
  - spaCy
  - NetworkX
  - scikit-learn

- Frontend:
  - HTML/CSS
  - JavaScript
  - Tailwind CSS
  - vis.js (for graph visualization)

## License


