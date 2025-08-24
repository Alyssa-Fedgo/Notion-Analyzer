
# 🧠 Notion Analyzer: Mood Journal Insights

A Python tool that pulls your daily journal entries from Notion and structures them for reflection and analysis. Now orchestrated with Apache Airflow, it automatically updates your Notion page with insights from your entries, helping you identify patterns, enhance self-awareness, and support mental clarity using your own writing.

---

## ✨ Features

- Connects securely to your Notion workspace using the Notion API
- Recursively collects journal pages from your workspace
- Extracts and organizes entries into key reflective sections:
  - **Summary**
  - **Grateful**
  - **Intentions**
  - **Focus**
- Adds sentence framing for each section for better NLP and readability
- Automatically updates a Notion page with structured insights

---

## 📌 Use Cases

- Analyze themes in your daily reflections
- Track what influences your mood or productivity
- Use with AI tools or visualization libraries for deeper insight

---

## 🛠️ Requirements

- Python 3.8 or higher
- Apache Airflow (scheduler must be running to execute DAGs)
- Notion Integration Token
- Notion journal pages formatted with headers (e.g., `Grateful`, `Intentions`, etc.)

---

## 📦 Installation

```bash
git clone https://github.com/Alyssa-Fedgo/Notion-Analyzer.git
cd Notion-Analyzer
pip install -r requirements.txt
```

## 🔐 Notion Setup

1. Create an internal integration here: https://www.notion.so/my-integrations
2. Copy your Internal Integration Token
3. Share your journal pages or database with that integration
4. Set your token in the terminal session:

```
export NOTION_TOKEN="your-secret-token"

```
## 🚀 Usage via Airflow
1. Ensure Airflow scheduler is running and your DAG is deployed in the /dags folder
2. Trigger the DAG from the Airflow UI or CLI
3. After running, your Notion page will be updated with:
   - frequency of negative days
   - Common words
   - Correlation of words with polarity score

## 📂 Project Structure
```
Notion-Analyzer/
│
├── notionapi.py          # Main logic
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
└── journal_output.csv    # Output (generated after running)
```
## 🧠 Future Enhancements

- Improve NLP sentiment and topic modeling
- Integrate with visualization libraries


## 👩‍💻 Author

Alyssa Fedgo, 
Lead Data Engineer
[LinkedIn Profile](https://www.linkedin.com/in/alyssa-fedgo-mph/)


