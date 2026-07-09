# Fake News Detection using Fine-tuned DistilBERT

> Binary classification of news articles as real or fake using a 
> fine-tuned DistilBERT transformer model, with a live Gradio demo.

## 🚀 Live Demo
https://3ccb38297a368fbe42.gradio.live/ 

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | XX.X% |
| Weighted F1 | 0.XXX |
| Fake News Recall | 0.XXX |
| Real News Precision | 0.XXX |

![Training History](figures/training_history.png)
![Confusion Matrix](figures/confusion_matrix.png)

---

## Model

**Architecture:** DistilBERT (distilbert-base-uncased) + classification head  
**Pre-training:** 66M parameter model pre-trained on Wikipedia + BookCorpus  
**Fine-tuning:** 3 epochs on 20,000 balanced news articles  
**Max sequence length:** 256 tokens  
**Hardware:** Google Colab T4 GPU (~20 minutes training)

## Dataset

- **WELFake Dataset** — 72,134 news articles (Kaggle)
- Balanced to 10,000 fake + 10,000 real for training
- 80/10/10 train/val/test split with stratification

## Key Design Decisions

**Why DistilBERT over full BERT?**  
DistilBERT retains 97% of BERT's performance at 40% smaller size — 
practical for Colab training while maintaining state-of-the-art NLP capability.

**Why title + text combined?**  
Fake news headlines often use sensationalist language distinct from the 
article body. Combining both with a [SEP] separator gives the model 
access to both signals.

**Why F1 over accuracy?**  
With balanced classes, accuracy and F1 converge — but F1 separately 
tracks false positives (real news falsely flagged) and false negatives 
(fake news missed), making it more informative for real-world deployment.

## How to Run

```bash
pip install -r requirements.txt
```

Open `fake_news_detection_bert.ipynb` in Google Colab (GPU runtime).  
Run all cells in order. Gradio demo launches at the end.

## Model Weights

The fine-tuned model weights are not stored in this repository due to 
size constraints. To use the trained model:

**Option 1 — Retrain yourself:**  
Run all cells in the notebook (~20 mins on Colab T4 GPU)

**Option 2 — HuggingFace Hub (coming soon):**  
Model will be uploaded to huggingface.co/divyamthakur/fake-news-bert

## Author

**Divyam Thakur** | B.Tech AI/ML, Manipal University Jaipur
