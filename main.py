import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict
from tqdm import tqdm
import warnings
from fastapi import FastAPI
from pydantic import BaseModel

warnings.filterwarnings('ignore')

# -------------------- CONFIG --------------------
CONFIG = {
    "retrieval_top_k": 5,
    "rerank_top_k": 3,
    "similarity_threshold": 0.65,
    "cross_encoder_threshold": 3.0,
    "entailment_threshold": 0.70,
    "contradiction_threshold": 0.70,
    "min_support_ratio": 0.70,
    "allow_any_contradiction": True,
}

# -------------------- UTILS --------------------
def split_into_claims(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        subclaims = re.split(
            r'\s+(?:and|but|however|while|whereas|though|although)\s+(?=[A-Z])',
            sent,
            flags=re.IGNORECASE
        )
        for claim in subclaims:
            claim = claim.strip().rstrip(',;')
            if len(claim) > 10:
                claims.append(claim)
    return claims if claims else [text.strip()]

# -------------------- HALLUCINATION DETECTOR --------------------
class ProductionHallucinationDetector:
    def __init__(self, device: str = "cpu", use_fast_mode: bool = False):
        self.device = device
        self.use_fast_mode = use_fast_mode

        print("🔧 Loading models...")
        self.bi_encoder = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            device=device
        )
        if not use_fast_mode:
            self.cross_encoder = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2',
                device=device
            )
        else:
            self.cross_encoder = None
        self.nli_pipeline = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=0 if device == "cuda" else -1,
            top_k=None
        )

        self.kb_texts = []
        self.kb_embeddings = None

    def build_index(self, kb_texts: List[str]):
        self.kb_texts = kb_texts
        self.kb_embeddings = self.bi_encoder.encode(
            kb_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=True
        )

    def retrieve_candidates(self, claim: str, top_k: int) -> List[Dict]:
        claim_emb = self.bi_encoder.encode(claim, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
        similarities = cosine_similarity(claim_emb, self.kb_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.kb_texts[idx], float(similarities[idx]), int(idx)) for idx in top_indices]

    def rerank_candidates(self, claim: str, candidates: List[tuple]) -> List[Dict]:
        if not candidates:
            return []

        if self.use_fast_mode or self.cross_encoder is None:
            return [
                {"passage": cand[0], "similarity_score": cand[1], "rerank_score": cand[1], "kb_idx": cand[2]}
                for cand in candidates
            ]

        pairs = [[claim, passage] for passage, _, _ in candidates]
        rerank_scores = self.cross_encoder.predict(pairs, show_progress_bar=False, batch_size=32)
        reranked = [
            {"passage": cand[0], "similarity_score": cand[1], "rerank_score": float(rerank_scores[i]), "kb_idx": cand[2]}
            for i, cand in enumerate(candidates)
        ]
        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked

    def verify_with_nli(self, claim: str, passage: str) -> Dict:
        nli_input = f"{passage} {claim}"
        try:
            results = self.nli_pipeline(nli_input, truncation=True, max_length=512)
            label_scores = {item['label'].lower(): item['score'] for item in results[0]}
            entailment = label_scores.get('entailment', 0.0)
            contradiction = label_scores.get('contradiction', 0.0)
            neutral = label_scores.get('neutral', 0.0)

            if entailment >= CONFIG['entailment_threshold']:
                verdict = "supported"
            elif contradiction >= CONFIG['contradiction_threshold']:
                verdict = "contradicted"
            else:
                verdict = "neutral"

            return {"verdict": verdict, "entailment_score": entailment, "contradiction_score": contradiction, "neutral_score": neutral}
        except Exception as e:
            return {"verdict": "error", "entailment_score": 0.0, "contradiction_score": 0.0, "neutral_score": 1.0}

    def verify_claim(self, claim: str) -> Dict:
        candidates = self.retrieve_candidates(claim, CONFIG['retrieval_top_k'])
        if not candidates or candidates[0][1] < 0.1:
            return {"claim": claim, "verdict": "unsupported", "confidence": 0.0, "evidence": [], "reasoning": "No relevant passages found in KB"}

        reranked = self.rerank_candidates(claim, candidates)[:CONFIG['rerank_top_k']]
        best = reranked[0]
        best_passage = best['passage']
        best_similarity = best['similarity_score']
        best_rerank = best['rerank_score']

        nli_result = self.verify_with_nli(claim, best_passage)
        verdict = nli_result['verdict']

        if verdict == "neutral":
            if best_similarity >= CONFIG['similarity_threshold']:
                if not self.use_fast_mode and best_rerank >= CONFIG['cross_encoder_threshold']:
                    verdict = "supported"
                elif self.use_fast_mode:
                    verdict = "supported"

        confidence = (
            min(best_similarity, nli_result['entailment_score']) if verdict == "supported"
            else nli_result['contradiction_score'] if verdict == "contradicted"
            else 0.5
        )

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "evidence": reranked,
            "nli_scores": nli_result,
            "best_similarity": best_similarity,
            "reasoning": self._get_reasoning(verdict, best_similarity, nli_result)
        }

    def _get_reasoning(self, verdict: str, similarity: float, nli: Dict) -> str:
        if verdict == "supported":
            return f"High similarity ({similarity:.2f}) and entailment ({nli['entailment_score']:.2f})"
        elif verdict == "contradicted":
            return f"Contradiction detected (conf: {nli['contradiction_score']:.2f})"
        elif verdict == "unsupported":
            return f"Low similarity ({similarity:.2f}), no KB support"
        else:
            return f"Neutral - similarity {similarity:.2f}, needs verification"

    def verify_response(self, response_text: str) -> Dict:
        claims = split_into_claims(response_text)
        claim_results = [self.verify_claim(c) for c in claims]

        supported = sum(1 for r in claim_results if r['verdict'] == 'supported')
        contradicted = sum(1 for r in claim_results if r['verdict'] == 'contradicted')
        unsupported = sum(1 for r in claim_results if r['verdict'] == 'unsupported')
        total = len(claim_results)
        support_ratio = supported / total if total > 0 else 0
        avg_confidence = np.mean([r['confidence'] for r in claim_results])

        if contradicted > 0 and CONFIG['allow_any_contradiction']:
            overall_verdict = "hallucinated"
            reasoning = f"{contradicted}/{total} claims contradicted by KB"
        elif support_ratio >= CONFIG['min_support_ratio']:
            overall_verdict = "verified"
            reasoning = f"{supported}/{total} claims supported ({support_ratio:.0%})"
        elif unsupported > total * 0.5:
            overall_verdict = "insufficient_evidence"
            reasoning = f"{unsupported}/{total} claims lack KB support"
        else:
            overall_verdict = "partially_verified"
            reasoning = f"Mixed results: {supported} supported, {contradicted} contradicted"

        return {
            "overall_verdict": overall_verdict,
            "confidence": float(avg_confidence),
            "claims": claim_results,
            "reasoning": reasoning
        }

# -------------------- FASTAPI --------------------
class RequestBody(BaseModel):
    llm_response: str
    knowledge_base: List[str]

app = FastAPI(title="Semantic Similarity & Hallucination Detector")

@app.post("/verify")
def verify_llm_response(req: RequestBody):
    detector = ProductionHallucinationDetector(device="cpu", use_fast_mode=False)
    detector.build_index(req.knowledge_base)
    result = detector.verify_response(req.llm_response)
    return result

# -------------------- MAIN (OPTIONAL LOCAL TEST) --------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
