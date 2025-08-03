
# 🧠 Notion Analyzer: Mood Journal Insights

A Python tool that pulls your daily journal entries from Notion and structures them into a clean dataset for reflection and analysis. Designed to help identify personal patterns, enhance self-awareness, and support mental clarity using your own writing.

---

## ✨ Features

- Connects securely to your Notion workspace using the Notion API
- Recursively collects journal pages from your workspace
- Extracts and organizes entries into key reflective sections:
  - **Summary**
  - **Grateful**
  - **Intentions**
  - **Focus**
- Outputs a formatted CSV that's easy to read, analyze, or visualize
- Adds sentence framing for each section for better NLP and readability

---

## 📌 Use Cases

- Analyze themes in your daily reflections
- Track what influences your mood or productivity
- Use with AI tools, Power BI, or Jupyter Notebooks for deeper insight

---

## 🛠️ Requirements

- Python 3.8 or higher
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
## 🚀 Usage

```
python notionapi.py
```
After running, you'll get:

- journal_output.csv — a CSV file with one row per day and structured columns

## 📊 Sample Output

| page\_id | Summary                   | Grateful                          | Intentions                        | Focus                               | Total                                                                                   |
| -------- | ------------------------- | --------------------------------- | --------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------- |
| abc123   | I learned to stay calm... | Today I'm grateful for my health. | Today I intend to be more patient | To make today a good day I would... | Today I'm grateful for... Today I intend to... To make today a good day... I learned... |

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

- Add NLP sentiment or topic modeling
- Integrate with visualization libraries
- Sync with calendar

## 👩‍💻 Author

Alyssa Fedgo
Lead Data Engineer
📫 LinkedIn Profile
🔗 GitHub Repo

