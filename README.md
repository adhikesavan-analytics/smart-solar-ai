# ☀️ Smart Solar AI

Smart Solar AI is a multi-user business intelligence and analytics platform designed to help organizations analyze data, detect anomalies, and receive automated alerts in real time.

---

## 🚀 Key Features

### 🔐 Role-Based Access
- Admin: Manage users, companies, departments
- User: Access assigned data and analytics

### 📊 Data Analytics Dashboard
- Revenue, cost, efficiency tracking
- Interactive charts and KPIs

### 🧹 Data Processing Pipeline
- Upload CSV/Excel files
- Automatic cleaning and transformation

### 🚨 Alert System
- Detects:
  - Low efficiency
  - Revenue drops
  - Expense spikes
  - Low stock
  - High debt
  - Anomalies
- Sends email notifications automatically

### 📬 Email Integration
- SMTP-based alerts using Gmail

### 🗄️ Persistent Database
- PostgreSQL (Neon) for cloud storage
- Ensures data is not lost on restart

---

## 🏗️ System Architecture

User → Authentication → Data Upload → Processing → Analytics → Alerts → Email Notification

---

## 🛠️ Tech Stack

- **Frontend & Backend:** Streamlit (Python)
- **Database:** PostgreSQL (Neon)
- **Email Service:** Gmail SMTP
- **Version Control:** GitHub
- **Deployment:** Streamlit Community Cloud

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/adhikesavan-analytics/smart-solar-ai.git
cd smart-solar-ai
pip install -r requirements.txt
streamlit run app.py

🌐 Live App

👉 https://smart-solar-ai-dlkriwds8sci2xdtxb57a3.streamlit.app/

🎯 Project Highlights
Multi-tenant system (supports multiple companies)
Real-time alert engine
Modular architecture (modules/ folder)
Cloud deployment with persistent storage
Secure email integration

🔮 Future Improvements
Admin-configurable alert thresholds
Enhanced AI-driven insights
Custom domain integration
Advanced security features

👨‍💻 Author

Adhi Kesavan M


