"""The fixed reading list used to build the paper collection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    key: str
    arxiv_id: str
    title: str
    year: int
    topic: str

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}"

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


CATALOG: tuple[Paper, ...] = (
    Paper(
        key="attention",
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        year=2017,
        topic="Transformer foundations",
    ),
    Paper(
        key="bert",
        arxiv_id="1810.04805",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        year=2018,
        topic="Bidirectional pretraining",
    ),
    Paper(
        key="roberta",
        arxiv_id="1907.11692",
        title="RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        year=2019,
        topic="Pretraining improvements",
    ),
    Paper(
        key="electra",
        arxiv_id="2003.10555",
        title="ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
        year=2020,
        topic="Sample-efficient pretraining",
    ),
    Paper(
        key="t5",
        arxiv_id="1910.10683",
        title="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        year=2019,
        topic="Text-to-text transfer learning",
    ),
    Paper(
        key="gpt3",
        arxiv_id="2005.14165",
        title="Language Models are Few-Shot Learners",
        year=2020,
        topic="Scaling and in-context learning",
    ),
    Paper(
        key="instructgpt",
        arxiv_id="2203.02155",
        title="Training Language Models to Follow Instructions with Human Feedback",
        year=2022,
        topic="Alignment with human feedback",
    ),
    Paper(
        key="chinchilla",
        arxiv_id="2203.15556",
        title="Training Compute-Optimal Large Language Models",
        year=2022,
        topic="Scaling laws and data budgets",
    ),
    Paper(
        key="llama",
        arxiv_id="2302.13971",
        title="LLaMA: Open and Efficient Foundation Language Models",
        year=2023,
        topic="Efficient open language models",
    ),
    Paper(
        key="llama2",
        arxiv_id="2307.09288",
        title="Llama 2: Open Foundation and Fine-Tuned Chat Models",
        year=2023,
        topic="Open chat models and safety",
    ),
    Paper(
        key="mistral",
        arxiv_id="2310.06825",
        title="Mistral 7B",
        year=2023,
        topic="Efficient small language models",
    ),
    Paper(
        key="gpt4",
        arxiv_id="2303.08774",
        title="GPT-4 Technical Report",
        year=2023,
        topic="Frontier multimodal models",
    ),
)

CATALOG_BY_KEY = {paper.key: paper for paper in CATALOG}
