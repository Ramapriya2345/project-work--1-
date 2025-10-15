import numpy as np
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict
from tqdm import tqdm
import warnings
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------- LOGGING SETUP --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# -------------------- CONFIG --------------------
CONFIG = {
    "retrieval_top_k": 5,
    "rerank_top_k": 3,
    "similarity_threshold": 0.7,
    "cross_encoder_threshold": 3.0,
    "entailment_threshold": 0.70,
    "contradiction_threshold": 0.70,
    "min_support_ratio": 0.70,
    "allow_any_contradiction": True,
}

# -------------------- UTILS --------------------
def split_into_claims(text: str) -> List[str]:
    logger.info("Splitting response into claims...")
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
    logger.info(f"Total claims extracted: {len(claims)}")
    return claims if claims else [text.strip()]

# -------------------- HALLUCINATION DETECTOR --------------------
class ProductionHallucinationDetector:
    def __init__(self, device: str = "cpu", use_fast_mode: bool = False):
        self.device = device
        self.use_fast_mode = use_fast_mode

        logger.info("🔧 Loading models...")
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
        logger.info("✅ Model loading complete.")

    def build_index(self, kb_texts: List[str]):
        logger.info("Building KB embeddings...")
        self.kb_texts = kb_texts
        self.kb_embeddings = self.bi_encoder.encode(
            kb_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=True
        )
        logger.info(f"✅ KB index built with {len(kb_texts)} entries.")

    def retrieve_candidates(self, claim: str, top_k: int) -> List[Dict]:
        logger.info(f"Retrieving top-{top_k} candidates for claim: {claim[:50]}...")
        claim_emb = self.bi_encoder.encode(claim, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
        similarities = cosine_similarity(claim_emb, self.kb_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.kb_texts[idx], float(similarities[idx]), int(idx)) for idx in top_indices]

    def rerank_candidates(self, claim: str, candidates: List[tuple]) -> List[Dict]:
        if not candidates:
            logger.warning("No candidates to rerank.")
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
        logger.info(f"Top reranked passage score: {reranked[0]['rerank_score']:.3f}")
        return reranked

    def verify_with_nli(self, claim: str, passage: str) -> Dict:
        logger.info("Running NLI verification...")
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

            logger.info(f"NLI verdict: {verdict} | entailment={entailment:.2f}, contradiction={contradiction:.2f}")
            return {"verdict": verdict, "entailment_score": entailment, "contradiction_score": contradiction, "neutral_score": neutral}
        except Exception as e:
            logger.error(f"NLI error: {e}")
            return {"verdict": "error", "entailment_score": 0.0, "contradiction_score": 0.0, "neutral_score": 1.0}

    def verify_claim(self, claim: str) -> Dict:
        logger.info(f"Verifying claim: {claim}")
        candidates = self.retrieve_candidates(claim, CONFIG['retrieval_top_k'])
        if not candidates or candidates[0][1] < 0.1:
            logger.warning("No relevant passages found in KB.")
            return {"claim": claim, "verdict": "unsupported", "confidence": 0.0, "evidence": [], "reasoning": "No relevant passages found in KB"}

        reranked = self.rerank_candidates(claim, candidates)[:CONFIG['rerank_top_k']]
        best = reranked[0]
        nli_result = self.verify_with_nli(claim, best['passage'])
        verdict = nli_result['verdict']

        if verdict == "neutral" and best['similarity_score'] >= CONFIG['similarity_threshold']:
            if not self.use_fast_mode and best['rerank_score'] >= CONFIG['cross_encoder_threshold']:
                verdict = "supported"
            elif self.use_fast_mode:
                verdict = "supported"

        confidence = (
            min(best['similarity_score'], nli_result['entailment_score']) if verdict == "supported"
            else nli_result['contradiction_score'] if verdict == "contradicted"
            else 0.5
        )
        logger.info(f"Claim verdict: {verdict} (conf={confidence:.2f})")

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "evidence": reranked,
            "nli_scores": nli_result,
        }

    def verify_response(self, response_text: str) -> Dict:
        logger.info("Starting full response verification...")
        claims = split_into_claims(response_text)
        claim_results = [self.verify_claim(c) for c in claims]

        supported = sum(1 for r in claim_results if r['verdict'] == 'supported')
        contradicted = sum(1 for r in claim_results if r['verdict'] == 'contradicted')
        total = len(claim_results)
        support_ratio = supported / total if total > 0 else 0
        avg_confidence = np.mean([r['confidence'] for r in claim_results])

        if contradicted > 0 and CONFIG['allow_any_contradiction']:
            overall_verdict = "hallucinated"
        elif support_ratio >= CONFIG['min_support_ratio']:
            overall_verdict = "verified"
        else:
            overall_verdict = "partially_verified"

        logger.info(f"✅ Overall Verdict: {overall_verdict.upper()} | Confidence: {avg_confidence:.2f}")
        return {
            "overall_verdict": overall_verdict,
            "confidence": float(avg_confidence),
            "claims": claim_results,
        }

# -------------------- FASTAPI --------------------
class RequestBody(BaseModel):
    llm_response: str
    knowledge_base: List[str]

app = FastAPI(title="Semantic Similarity & Hallucination Detector")

@app.post("/verify")
def verify_llm_response(req: RequestBody):
    logger.info("Received new verification request.")
    detector = ProductionHallucinationDetector(device="cpu", use_fast_mode=False)
    detector.build_index(req.knowledge_base)
    result = detector.verify_response(req.llm_response)
    logger.info("Verification completed successfully.")
    return result

# -------------------- MAIN --------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting FastAPI app on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
