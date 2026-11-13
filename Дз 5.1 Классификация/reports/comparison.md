| Метод | Macro F1 | Accuracy | Улучшение |
|-------|----------|----------|-----------|
| 0. baseline: tfidf 5000 1-2gram + logreg | 0.6809 | 0.8012 | - |
| 1. лемматизация + baseline | 0.6785 | 0.7971 | -0.4% |
| 2. tfidf 1-3gram + logreg | 0.6847 | 0.8065 | +0.6% |
| 3. tfidf + random forest | 0.6319 | 0.7638 | -7.2% |
| 4. tfidf + linear svc | 0.7582 | 0.8117 | +11.4% |
| 5. tfidf + logreg balanced | 0.7136 | 0.7846 | +4.8% |
| 5b. tfidf + linear svc balanced | 0.7385 | 0.8012 | +8.5% |
| 6. word + char tfidf + logreg balanced | 0.7487 | 0.8179 | +10.0% |
| 7. word + char tfidf + logreg balanced, C=3 | 0.7631 | 0.8262 | +12.1% |
