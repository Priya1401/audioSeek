import json
import logging
import time
from typing import List, Dict
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


class QAValidator:
    """Validate Q&A system performance"""

    def __init__(
        self,
        min_rouge_score: float = 0.4,
        min_citation_count: int = 1,
        max_response_time: float = 15.0
    ):
        self.min_rouge_score = min_rouge_score
        self.min_citation_count = min_citation_count
        self.max_response_time = max_response_time
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def create_test_cases(self, book_id: str) -> List[Dict]:
        """
        Create test cases for Q&A validation.
        In production, load these from a test dataset file.
        """
        # You should have these as ground truth for each book
        test_cases = {
            'romeo_and_juliet': [
                # {
                #     'query': 'What happens in the prologue?',
                #     'expected_answer': 'The prologue introduces the long-standing feud between the Montagues and Capulets and sets the stage for the tragic events involving Romeo and Juliet. It explains that the lovers are ‘star-crossed’ and that their deaths will end the families’ strife.',
                #     'chapter_id': None,
                #     'query_type': 'factual'
                # },
                {
                    'query': 'Who are Romeo and Juliet?',
                    'expected_answer': 'Romeo and Juliet are two lovers whose families are enemies. Juliet is from the Capulet family, not yet fourteen, and pressured to marry Paris. Romeo is from the Montague family. Their love is intense and ends tragically when Romeo buys poison and dies beside Juliet. Their deaths are called ‘poor sacrifices of our enmity.',
                    'chapter_id': None,
                    'query_type': 'factual'
                },
                {
                    'query': 'What happened in chapter 1?',
                    'expected_answer': 'The context shows a street fight between Montague and Capulet servants. Samson and Gregory provoke Abraham and Balthazar by biting a thumb. The argument becomes a fight until Benvolio tries to stop it, and then Tibbled arrives and confronts him.',
                    'chapter_id': 1,
                    'query_type': 'chapter_specific'
                },
                {
                    'query': 'How do Romeo and Juliet meet?',
                    'expected_answer': 'The passages do not describe how Romeo and Juliet meet. They only cover their secret marriage, Romeo’s banishment, and Juliet being asked to marry Paris.',
                    'chapter_id': 2,
                    'query_type': 'factual'
                },
                {
                    'query': 'What is the main theme of the story?',
                    'expected_answer': 'The main theme is the sudden love between Romeo and Juliet against the long family feud. This appears in Tybalt seeing Romeo as an enemy, the Prince’s warnings, their rushed marriage, and Juliet being pushed to marry Paris.',
                    'chapter_id': None,
                    'query_type': 'analytical'
                }
            ],
            'around_the_world': [
                {
                    'query': 'Who is Phileas Fogg?',
                    'expected_answer': 'Phileas Fogg is an English gentleman who makes a wager to travel around the world in 80 days.',
                    'chapter_id': None,
                    'query_type': 'factual'
                },
                {
                    'query': 'What is the bet about?',
                    'expected_answer': 'Phileas Fogg bets that he can travel around the world in eighty days.',
                    'chapter_id': 1,
                    'query_type': 'factual'
                },
                {
                    'query': 'Who is Passepartout?',
                    'expected_answer': 'Passepartout is Phileas Fogg\'s French valet and traveling companion.',
                    'chapter_id': None,
                    'query_type': 'factual'
                }
            ]
        }

        return test_cases.get(book_id, [])

    def validate_qa_system(
        self,
        book_id: str,
        qa_service,
        test_cases: List[Dict] = None
    ) -> Dict:
        """
        Validate Q&A system using test cases

        Args:
            book_id: Book identifier
            qa_service: QAService instance
            test_cases: Optional test cases (if None, uses default)

        Returns:
            Validation results with metrics
        """
        if test_cases is None:
            test_cases = self.create_test_cases(book_id)

        if not test_cases:
            logger.warning(f"No test cases found for book_id={book_id}")
            return {
                'validation_passed': False,
                'failure_reason': 'No test cases available'
            }

        logger.info("=" * 70)
        logger.info(f"Q&A VALIDATION for {book_id}")
        logger.info(f"Running {len(test_cases)} test queries...")
        logger.info("=" * 70)

        results = []

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\nTest {i}/{len(test_cases)}: {test_case['query']}")

            start_time = time.time()

            try:
                # Make Q&A request
                from models import QueryRequest

                request = QueryRequest(
                    query=test_case['query'],
                    book_id=book_id,
                    top_k=5
                )

                response = qa_service.ask_question(request)
                response_time = time.time() - start_time

                # Calculate ROUGE-L score
                rouge_score = self.scorer.score(
                    test_case['expected_answer'],
                    response.answer
                )
                rouge_l = rouge_score['rougeL'].fmeasure

                # Check individual criteria
                rouge_passed = rouge_l >= self.min_rouge_score
                citations_passed = len(
                    response.citations) >= self.min_citation_count
                time_passed = response_time <= self.max_response_time

                overall_passed = rouge_passed and citations_passed and time_passed

                result = {
                    'test_id': i,
                    'query': test_case['query'],
                    'query_type': test_case['query_type'],
                    'chapter_id': test_case.get('chapter_id'),
                    'expected_answer': test_case['expected_answer'],
                    'actual_answer': response.answer,
                    'rouge_l': round(rouge_l, 4),
                    'citations_count': len(response.citations),
                    'citations': response.citations,
                    'response_time': round(response_time, 3),
                    'rouge_passed': rouge_passed,
                    'citations_passed': citations_passed,
                    'time_passed': time_passed,
                    'test_passed': overall_passed
                }

                logger.info(
                    f"  ROUGE-L: {rouge_l:.3f} ({'✓' if rouge_passed else '✗'})")
                logger.info(
                    f"  Citations: {len(response.citations)} ({'✓' if citations_passed else '✗'})")
                logger.info(
                    f"  Time: {response_time:.2f}s ({'✓' if time_passed else '✗'})")
                logger.info(
                    f"  Overall: {'✓ PASSED' if overall_passed else '✗ FAILED'}")

            except Exception as e:
                logger.error(f"  ✗ ERROR: {e}")
                result = {
                    'test_id': i,
                    'query': test_case['query'],
                    'query_type': test_case['query_type'],
                    'test_passed': False,
                    'error': str(e)
                }

            results.append(result)

        # Calculate summary statistics
        summary = self._calculate_summary(results)

        validation_result = {
            'book_id': book_id,
            'validation_passed': summary['pass_rate'] >= 0.7,
            # 70% pass rate required
            'individual_results': results,
            'summary': summary,
            'thresholds': {
                'min_rouge_score': self.min_rouge_score,
                'min_citation_count': self.min_citation_count,
                'max_response_time': self.max_response_time,
                'min_pass_rate': 0.7
            }
        }

        self._log_validation_summary(validation_result)

        return validation_result

    def _calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics"""
        total = len(results)
        passed = sum(1 for r in results if r.get('test_passed', False))
        failed = total - passed

        # Filter out error cases for metric calculations
        valid_results = [r for r in results if 'rouge_l' in r]

        if not valid_results:
            return {
                'total_tests': total,
                'passed': 0,
                'failed': total,
                'pass_rate': 0,
                'errors': total
            }

        summary = {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'avg_rouge_l': np.mean([r['rouge_l'] for r in valid_results]),
            'std_rouge_l': np.std([r['rouge_l'] for r in valid_results]),
            'avg_citations': np.mean(
                [r['citations_count'] for r in valid_results]),
            'avg_response_time': np.mean(
                [r['response_time'] for r in valid_results]),
            'by_query_type': {}
        }

        # Break down by query type
        for result in valid_results:
            q_type = result.get('query_type', 'unknown')
            if q_type not in summary['by_query_type']:
                summary['by_query_type'][q_type] = []
            summary['by_query_type'][q_type].append(result['test_passed'])

        # Calculate pass rate by type
        for q_type, passed_list in summary['by_query_type'].items():
            summary['by_query_type'][q_type] = {
                'total': len(passed_list),
                'passed': sum(passed_list),
                'pass_rate': sum(passed_list) / len(passed_list)
            }

        return summary

    def _log_validation_summary(self, result: Dict):
        """Log validation summary"""
        logger.info("\n" + "=" * 70)
        logger.info("Q&A VALIDATION SUMMARY")
        logger.info("=" * 70)

        summary = result['summary']
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Avg ROUGE-L: {summary.get('avg_rouge_l', 0):.3f}")
        logger.info(f"Avg Citations: {summary.get('avg_citations', 0):.1f}")
        logger.info(
            f"Avg Response Time: {summary.get('avg_response_time', 0):.2f}s")

        if 'by_query_type' in summary:
            logger.info("\nPerformance by Query Type:")
            for q_type, stats in summary['by_query_type'].items():
                logger.info(
                    f"  {q_type}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")

        logger.info(
            f"\nOverall Validation: {'✓ PASSED' if result['validation_passed'] else '✗ FAILED'}")
        logger.info("=" * 70)

    def save_validation_report(self, validation_result: Dict, output_path: str):
        """Save validation report to file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(validation_result, f, indent=2)

        logger.info(f"Q&A validation report saved to: {output_path}")