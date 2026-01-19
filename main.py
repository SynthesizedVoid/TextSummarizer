import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Literal, Protocol, cast
from numpy.typing import NDArray
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

ISentimentLabel = Literal['Positive', 'Negative', 'Neutral']
ISummaryMap = Dict[str, str]
ISentimentScoreMap = Dict[str, float]


class _Sentiment(Protocol):
    polarity: float
    subjectivity: float


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def fetch_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError("'text.txt' file couldn't be found.")
    if not path.is_file():
        raise ValueError('The path should be a file.')
    text: str = path.read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError('Empty file.')
    return text


def summarize(sentences: List[str], n: int) -> str:
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(sentences)
    dense: NDArray[np.float64] = matrix.toarray()  # pyright: ignore[reportAttributeAccessIssue]
    scores: NDArray[np.float64] = dense.sum(axis=1)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    ranked.sort()
    return ' '.join(sentences[i] for i in ranked)


def get_polarity(text: str) -> float:
    sentiment = cast(_Sentiment, TextBlob(text).sentiment)
    return sentiment.polarity


def get_sentiment(score: float) -> ISentimentLabel:
    if score > 0:
        return 'Positive'
    if score < 0:
        return 'Negative'
    return 'Neutral'


text_path: Path = Path('text.txt')
text: str = fetch_text(text_path)
sentences = split_sentences(text)
summaries: ISummaryMap = {
    'Short': summarize(sentences, 3),
    'Medium': summarize(sentences, 5),
    'Long': summarize(sentences, 7),
}
sentiment_scores: ISentimentScoreMap = {}
for type, summary in summaries.items():
    polarity = get_polarity(summary)
    sentiment_scores[type] = polarity
    print(f'{type} Summary Result:')
    print(f'Summarized Text:\n{summary}')
    print('Sentiment:', get_sentiment(polarity))
    print('Score:', round(polarity, 3), end='\n\n')

avg_score: float = sum(sentiment_scores.values()) / len(sentiment_scores)
print('Final Result:')
print('Average Score:', round(avg_score, 3))
print('Overall Sentiment:', get_sentiment(avg_score))
labels: List[str] = list(sentiment_scores.keys())
values: List[float] = list(sentiment_scores.values())
plt.figure()
plt.bar(labels, values)
plt.axhline(0)
plt.xlabel('Summary Length')
plt.ylabel('Polarity Score')
plt.title('Sentiment Comparison Across Summaries')
plt.show()
