---
title: Sentinex Mental Health AI
emoji: 🧠
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF0000,50:FF6B00,100:FFD700&height=200&section=header&text=SENTINEX&fontSize=80&fontColor=ffffff&fontAlignY=35&desc=Mental%20Health%20Intelligence%20AI&descAlignY=60&descSize=22&animation=fadeIn" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Orbitron&weight=900&size=28&duration=3000&pause=800&color=FF4500&center=true&vCenter=true&multiline=true&width=700&height=100&lines=🧠+Emotion+Detection+%7C+Risk+Scoring;🎭+Sarcasm+Intelligence+%7C+NLP+AI;🔴+CRITICAL+to+🟢+LOW+—+Real+Time)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-FFD21E?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

> **🚨 The AI that reads between the lines — detecting hidden pain, masked depression, and silent cries for help.**

</div>

---

<div align="center">

## 🔥 WHAT IS SENTINEX?

</div>

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     SENTINEX — Mental Health Intelligence System v2.0               ║
║                                                                      ║
║     "Not just what you say — but what you MEAN."                   ║
║                                                                      ║
║     Built with 3 Transformer Models + Rule-Based Psychology         ║
║     Detects: Emotion • Sarcasm • Risk • Hidden Distress             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

SENTINEX is an advanced **NLP-powered Mental Health Analysis AI** that goes beyond surface-level sentiment. It understands **sarcasm**, detects **psychological markers**, tracks **session mood trends**, and delivers a precise **4-tier risk assessment** — from 🟢 LOW to 🔴 CRITICAL.

---

<div align="center">

## ⚡ CORE CAPABILITIES

</div>

<table align="center">
<tr>
<td align="center" width="200">

### 🎭 Sarcasm Engine
Detects masked negativity hiding behind positive words.<br/>
*"Best week ever"* → SARCASM ✅

</td>
<td align="center" width="200">

### 😢 Emotion Detection
7-class emotion analysis using DistilRoBERTa.<br/>
Sadness • Joy • Fear • Anger • Disgust • Surprise • Neutral

</td>
<td align="center" width="200">

### 🧬 Psych Markers
Rule-based psychology keyword system.<br/>
Hopelessness • Isolation • Self-Blame • Sleep • Pessimism

</td>
<td align="center" width="200">

### 📈 Context Memory
Tracks last 10 messages for trend analysis.<br/>
Escalating sadness → Risk boost applied automatically

</td>
</tr>
</table>

---

<div align="center">

## 🚨 RISK LEVEL SYSTEM

</div>

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   🟢  LOW      Score: 0.00 – 0.30   "You're doing well"│
│   🟡  MODERATE Score: 0.30 – 0.55   "Some stress noted"│
│   🟠  HIGH     Score: 0.55 – 0.75   "Seek support"     │
│   🔴  CRITICAL Score: 0.75 – 1.00   "Immediate help"   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

<div align="center">

## 🧠 ARCHITECTURE

</div>

```
INPUT TEXT
    │
    ▼
┌─────────────────────────────────────────────────┐
│              SENTINEX PIPELINE                  │
│                                                 │
│  ┌─────────────┐  ┌──────────────┐             │
│  │  Emotion    │  │  Sentiment   │             │
│  │DistilRoBERTa│  │  RoBERTa     │             │
│  │  7 classes  │  │  (Twitter)   │             │
│  └──────┬──────┘  └──────┬───────┘             │
│         │                │                     │
│  ┌──────▼──────┐  ┌──────▼───────┐             │
│  │  Sarcasm    │  │   Psych      │             │
│  │  RoBERTa +  │  │   Marker     │             │
│  │  Rule Engine│  │   Scanner    │             │
│  └──────┬──────┘  └──────┬───────┘             │
│         │                │                     │
│         └────────┬────────┘                    │
│                  ▼                             │
│         ┌────────────────┐                    │
│         │  Risk Scorer   │                    │
│         │ Context Memory │                    │
│         └────────┬───────┘                    │
│                  ▼                             │
│    🟢 LOW / 🟡 MODERATE / 🟠 HIGH / 🔴 CRITICAL│
└─────────────────────────────────────────────────┘
```

---

<div align="center">

## 🤖 MODELS USED

</div>

| Model | Purpose | Source |
|-------|---------|--------|
| `j-hartmann/emotion-english-distilroberta-base` | 7-class Emotion Detection | 🤗 HuggingFace |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment Analysis | 🤗 HuggingFace |
| `jkhan447/sarcasm-detection-RoBerta-base-POS` | Sarcasm Detection | 🤗 HuggingFace |

---

<div align="center">

## 🧪 TEST RESULTS

</div>

| Input | Expected | SENTINEX Output | Status |
|-------|----------|-----------------|--------|
| *"Best week ever! Lost job, cat died..."* | HIGH + SARCASM | 🟠 HIGH \| SARCASM ✅ | ✅ PASS |
| *"Got everything... why do I still cry myself to sleep?"* | HIGH | 🟠 HIGH \| SLEEP + PESSIMISM markers | ✅ PASS |
| *"Got my dream job! So grateful!"* | LOW | 🟢 LOW \| JOY 90% | ✅ PASS |
| *"Nobody understands. It's all my fault. Can't sleep."* | CRITICAL | 🔴 CRITICAL \| 3 psych markers | ✅ PASS |
| *"Went to market, cooked dinner, watched TV."* | LOW | 🟢 LOW \| NEUTRAL | ✅ PASS |

---

<div align="center">

## 🛠️ TECH STACK

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-RoBERTa-blueviolet?style=for-the-badge)

</div>

---

<div align="center">

## 👨‍💻 BUILT BY

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:FF0000,100:FFD700&height=60&text=Siddhant%20Chandorkar&fontSize=28&fontColor=ffffff&fontAlign=50&fontAlignY=50" width="500"/>

<br/><br/>

[![GitHub](https://img.shields.io/badge/GitHub-siddhantchandorkar752--ai-181717?style=for-the-badge&logo=github)](https://github.com/siddhantchandorkar752-ai)
[![HuggingFace](https://img.shields.io/badge/🤗-siddhantchandorkar-FFD21E?style=for-the-badge)](https://huggingface.co/siddhantchandorkar)

<br/>

*"I don't just build AI. I build AI that understands humans."*

</div>

---

<div align="center">

> ⚠️ **Disclaimer:** SENTINEX is an AI research tool and is NOT a substitute for professional mental health care. If you or someone you know is in crisis, please contact a mental health professional immediately.
>
> 🇮🇳 **iCall India:** 9152987821 | **Vandrevala Foundation:** 1860-2662-345

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FFD700,50:FF6B00,100:FF0000&height=120&section=footer&text=SENTINEX%20v2.0&fontSize=30&fontColor=ffffff&fontAlignY=65&animation=fadeIn" width="100%"/>

</div>
