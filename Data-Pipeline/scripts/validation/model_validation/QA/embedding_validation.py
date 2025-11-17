import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingValidator:
    """Validate embedding quality"""

    def __init__(
        self,
        expected_dimension: int = 384,  # all-MiniLM-L6-v2
        min_avg_similarity: float = 0.1,
        max_avg_similarity: float = 0.7,
        min_diversity_score: float = 0.3
    ):
        self.expected_dimension = expected_dimension
        self.min_avg_similarity = min_avg_similarity
        self.max_avg_similarity = max_avg_similarity
        self.min_diversity_score = min_diversity_score

    def validate_embeddings(
        self,
        embeddings: List[List[float]],
        chunks: List[Dict] = None,
        book_id: str = None
    ) -> Dict:
        """
        Validate embedding quality

        Args:
            embeddings: List of embedding vectors
            chunks: Optional chunk metadata for context-aware validation
            book_id: Book identifier

        Returns:
            Validation results with quality metrics
        """
        logger.info("=" * 70)
        logger.info(
            f"EMBEDDING VALIDATION{f' for {book_id}' if book_id else ''}")
        logger.info("=" * 70)

        if not embeddings:
            return {
                'validation_passed': False,
                'failure_reason': 'No embeddings provided'
            }

        embeddings_array = np.array(embeddings)

        logger.info(f"Validating {len(embeddings)} embeddings...")

        # Initialize validation results
        validation_checks = []
        metrics = {
            'total_embeddings': len(embeddings),
            'embedding_dimension': embeddings_array.shape[1]
        }

        # ============================================
        # CHECK 1: Correct Dimension
        # ============================================
        dimension_correct = embeddings_array.shape[1] == self.expected_dimension
        validation_checks.append({
            'check': 'dimension',
            'passed': dimension_correct,
            'message': f"Dimension: {embeddings_array.shape[1]} (expected: {self.expected_dimension})",
            'critical': True
        })

        if not dimension_correct:
            logger.error(
                f"✗ Wrong embedding dimension: {embeddings_array.shape[1]} != {self.expected_dimension}")
        else:
            logger.info(
                f"✓ Embedding dimension correct: {self.expected_dimension}")

        # ============================================
        # CHECK 2: No Invalid Values
        # ============================================
        has_nan = np.isnan(embeddings_array).any()
        has_inf = np.isinf(embeddings_array).any()
        no_invalid = not (has_nan or has_inf)

        validation_checks.append({
            'check': 'numeric_validity',
            'passed': no_invalid,
            'message': f"NaN: {has_nan}, Inf: {has_inf}",
            'critical': True
        })

        if not no_invalid:
            logger.error(
                f"✗ Invalid values detected - NaN: {has_nan}, Inf: {has_inf}")
        else:
            logger.info("✓ No NaN or Inf values")

        # ============================================
        # CHECK 3: No Zero Vectors
        # ============================================
        zero_vectors = np.all(embeddings_array == 0, axis=1).sum()
        no_zeros = zero_vectors == 0

        metrics['zero_vectors_count'] = int(zero_vectors)

        validation_checks.append({
            'check': 'non_zero_vectors',
            'passed': no_zeros,
            'message': f"Zero vectors: {zero_vectors}",
            'critical': True
        })

        if not no_zeros:
            logger.error(f"✗ Found {zero_vectors} zero vectors")
        else:
            logger.info("✓ No zero vectors")

        # ============================================
        # CHECK 4: Pairwise Similarity Distribution
        # ============================================
        logger.info("Calculating pairwise similarities...")

        # Sample for large datasets to avoid memory issues
        sample_size = min(1000, len(embeddings))
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size,
                                       replace=False)
            sample_embeddings = embeddings_array[indices]
        else:
            sample_embeddings = embeddings_array

        similarities = cosine_similarity(sample_embeddings)
        np.fill_diagonal(similarities, 0)  # Remove self-similarity

        avg_similarity = np.mean(similarities[similarities > 0])
        std_similarity = np.std(similarities[similarities > 0])

        metrics['avg_pairwise_similarity'] = float(avg_similarity)
        metrics['std_pairwise_similarity'] = float(std_similarity)
        metrics['min_similarity'] = float(
            np.min(similarities[similarities > 0]))
        metrics['max_similarity'] = float(np.max(similarities))

        similarity_ok = (
            self.min_avg_similarity <= avg_similarity <= self.max_avg_similarity
        )

        validation_checks.append({
            'check': 'similarity_range',
            'passed': similarity_ok,
            'message': f"Avg similarity: {avg_similarity:.3f} (range: [{self.min_avg_similarity}, {self.max_avg_similarity}])",
            'critical': False
        })

        if not similarity_ok:
            logger.warning(
                f"⚠ Avg similarity {avg_similarity:.3f} outside acceptable range")
        else:
            logger.info(f"✓ Avg similarity in range: {avg_similarity:.3f}")

        # ============================================
        # CHECK 5: Embedding Diversity
        # ============================================
        dimension_variances = np.var(embeddings_array, axis=0)
        avg_variance = np.mean(dimension_variances)

        metrics['avg_dimension_variance'] = float(avg_variance)

        diversity_ok = avg_variance >= self.min_diversity_score

        validation_checks.append({
            'check': 'diversity',
            'passed': diversity_ok,
            'message': f"Diversity score: {avg_variance:.4f} (min: {self.min_diversity_score})",
            'critical': False
        })

        if not diversity_ok:
            logger.warning(f"⚠ Low diversity: {avg_variance:.4f}")
        else:
            logger.info(f"✓ Good diversity: {avg_variance:.4f}")

        # ============================================
        # CHECK 6: Chapter Separation (if chunks provided)
        # ============================================
        if chunks and len(chunks) == len(embeddings):
            separation_score = self._check_chapter_separation(embeddings_array,
                                                              chunks)
            metrics['chapter_separation_score'] = float(separation_score)

            separation_ok = separation_score >= 0.1

            validation_checks.append({
                'check': 'chapter_separation',
                'passed': separation_ok,
                'message': f"Chapter separation: {separation_score:.3f}",
                'critical': False
            })

            if not separation_ok:
                logger.warning(
                    f"⚠ Poor chapter separation: {separation_score:.3f}")
            else:
                logger.info(
                    f"✓ Good chapter separation: {separation_score:.3f}")

        # ============================================
        # Overall Validation Decision
        # ============================================
        critical_checks = [c for c in validation_checks if
                           c.get('critical', False)]
        all_critical_passed = all(c['passed'] for c in critical_checks)

        non_critical_checks = [c for c in validation_checks if
                               not c.get('critical', False)]
        non_critical_passed = sum(1 for c in non_critical_checks if c['passed'])
        non_critical_pass_rate = (
            non_critical_passed / len(non_critical_checks)
            if non_critical_checks else 1.0
        )

        overall_passed = all_critical_passed and (non_critical_pass_rate >= 0.5)

        result = {
            'validation_passed': overall_passed,
            'metrics': metrics,
            'validation_checks': validation_checks,
            'summary': {
                'total_checks': len(validation_checks),
                'critical_checks_passed': sum(
                    1 for c in critical_checks if c['passed']),
                'critical_checks_total': len(critical_checks),
                'non_critical_pass_rate': non_critical_pass_rate
            }
        }

        if not overall_passed:
            failed_critical = [c['check'] for c in critical_checks if
                               not c['passed']]
            result[
                'failure_reason'] = f"Failed critical checks: {', '.join(failed_critical)}" if failed_critical else "Too many non-critical failures"

        self._log_validation_summary(result)

        return result

    def _check_chapter_separation(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict]
    ) -> float:
        """
        Check if embeddings from different chapters are well-separated

        Returns:
            Separation score (higher = better)
        """
        # Group embeddings by chapter
        chapter_groups = {}
        for i, chunk in enumerate(chunks):
            chapter_id = chunk.get('chapter_id')
            if chapter_id:
                if chapter_id not in chapter_groups:
                    chapter_groups[chapter_id] = []
                chapter_groups[chapter_id].append(i)

        if len(chapter_groups) < 2:
            return 1.0  # Only one chapter, no separation needed

        # Calculate intra-chapter vs inter-chapter similarity
        intra_similarities = []
        inter_similarities = []

        for chapter_id, indices in chapter_groups.items():
            if len(indices) < 2:
                continue

            # Intra-chapter similarity (within same chapter)
            chapter_embeddings = embeddings[indices]
            chapter_sims = cosine_similarity(chapter_embeddings)
            np.fill_diagonal(chapter_sims, 0)
            intra_similarities.extend(chapter_sims[chapter_sims > 0])

            # Inter-chapter similarity (with other chapters)
            other_indices = [
                i for cid, idxs in chapter_groups.items()
                if cid != chapter_id
                for i in idxs
            ]

            if other_indices:
                other_embeddings = embeddings[other_indices]
                cross_sims = cosine_similarity(chapter_embeddings,
                                               other_embeddings)
                inter_similarities.extend(cross_sims.flatten())

        if not intra_similarities or not inter_similarities:
            return 0.5

        avg_intra = np.mean(intra_similarities)
        avg_inter = np.mean(inter_similarities)

        # Good separation: high intra-chapter, low inter-chapter
        separation_score = avg_intra - avg_inter

        logger.info(f"  Intra-chapter similarity: {avg_intra:.3f}")
        logger.info(f"  Inter-chapter similarity: {avg_inter:.3f}")
        logger.info(f"  Separation score: {separation_score:.3f}")

        return max(0, separation_score)

    def _log_validation_summary(self, result: Dict):
        """Log validation summary"""
        logger.info("\n" + "=" * 70)
        logger.info("EMBEDDING VALIDATION SUMMARY")
        logger.info("=" * 70)

        metrics = result['metrics']
        logger.info(f"Total embeddings: {metrics['total_embeddings']}")
        logger.info(f"Dimension: {metrics['embedding_dimension']}")
        logger.info(
            f"Avg similarity: {metrics.get('avg_pairwise_similarity', 0):.3f}")
        logger.info(
            f"Diversity: {metrics.get('avg_dimension_variance', 0):.4f}")

        if 'chapter_separation_score' in metrics:
            logger.info(
                f"Chapter separation: {metrics['chapter_separation_score']:.3f}")

        summary = result['summary']
        logger.info(
            f"\nCritical checks: {summary['critical_checks_passed']}/{summary['critical_checks_total']}")
        logger.info(
            f"Non-critical pass rate: {summary['non_critical_pass_rate']:.1%}")

        logger.info(
            f"\nOverall Validation: {'✓ PASSED' if result['validation_passed'] else '✗ FAILED'}")

        if not result['validation_passed']:
            logger.error(f"Failure reason: {result.get('failure_reason')}")

        logger.info("=" * 70)

    def save_validation_report(self, validation_result: Dict, output_path: str):
        """Save validation report to file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(validation_result, f, indent=2)

        logger.info(f"Embedding validation report saved to: {output_path}")