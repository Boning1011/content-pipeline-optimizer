# Content Pipeline Optimizer

## 🎯 Why This Matters

In today's content-driven world, media production pipelines are becoming increasingly complex and costly. Content teams face significant challenges:
- High costs of rework and revisions
- Unpredictable production timelines
- Resource allocation inefficiencies
- Difficulty in estimating true project costs

This simulator was developed to help content production managers, studio leads, and workflow architects optimize their production pipelines by understanding the real impact of revisions and mistakes on project timelines and costs.

## 💡 Business Impact

- **Cost Reduction**: Identify bottlenecks and inefficiencies that drive up production costs
- **Better Resource Planning**: Make data-driven decisions about team size and structure
- **Accurate Estimations**: Predict realistic project timelines that account for revisions
- **Process Optimization**: Simulate different workflow configurations to find the most efficient setup
- **ROI Analysis**: Quantify the potential impact of process improvements and automation

## 🔍 How It Works

The tool uses Markov Chain mathematics to model real-world content production scenarios, accounting for three types of common workflow disruptions:
- Complete restarts (e.g., major creative direction changes)
- Task revisions (e.g., client feedback cycles)
- Step-back revisions (e.g., dependencies with previous tasks)

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/content-pipeline-optimizer.git
cd content-pipeline-optimizer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run scripts/pipeline_simulator.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

### First-Time Setup

1. Adjust the number of tasks in your pipeline (3-15 tasks)
2. Set your mistake rates:
   - Type 0: Complete restart probability
   - Type 1: Current task revision probability
   - Type 2: Step-back revision probability
3. Optionally set custom time costs for each task
4. Explore the visualizations and metrics to understand your pipeline's efficiency

