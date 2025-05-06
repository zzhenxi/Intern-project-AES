import logging
import json
import numpy as np
import os
from datetime import datetime


logger = logging.getLogger(__name__)


import numpy as np
from sklearn.metrics import confusion_matrix

def quadratic_weighted_kappa(holistic_scores, true_scores, bins=None):
    holistic_scores = np.array(holistic_scores, dtype=float)
    true_scores = np.array(true_scores, dtype=float)
    
    if bins is None:
        unique_ratings = np.sort(np.unique(np.concatenate([holistic_scores, true_scores])))
        num_ratings = len(unique_ratings)
        rating_to_idx = {rating: idx for idx, rating in enumerate(unique_ratings)}
        holistic_indices = np.array([rating_to_idx[score] for score in holistic_scores])
        true_indices = np.array([rating_to_idx[score] for score in true_scores])
    else:
        bins = np.array(bins)
        holistic_indices = np.digitize(holistic_scores, bins) 
        true_indices = np.digitize(true_scores, bins)
        num_ratings = len(bins) + 1
    
    observed = confusion_matrix(true_indices, holistic_indices, 
                               labels=list(range(num_ratings)))
    
    hist_true = np.bincount(true_indices, minlength=num_ratings)
    hist_holistic = np.bincount(holistic_indices, minlength=num_ratings)
    
    expected = np.outer(hist_true, hist_holistic) / float(len(true_scores))
    
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
    
    numerator = np.sum(weights * observed)
    denominator = np.sum(weights * expected)
    
    if denominator == 0:
        return 1.0
    
    qwk = 1 - (numerator / denominator)
    return qwk

def analyze_results(evaluation_results, multi_agent):
    logger.info("Analyzing evaluation results")
    
    if not evaluation_results:
        logger.warning("No evaluation results to analyze")
        return {}
    
    # Extract holistic scores
    holistic_scores = [result['final_assessment']['holistic_score'] for result in evaluation_results]
    
    # Extract true scores if available
    true_scores = [result['metadata']['input_data']['score']['holistic_score'] for result in evaluation_results]
    
    # Calculate basic statistics
    stats = {
        'count': len(holistic_scores),
        'mean': np.mean(holistic_scores) if holistic_scores else None,
        'median': np.median(holistic_scores) if holistic_scores else None,
        'min': np.min(holistic_scores) if holistic_scores else None,
        'max': np.max(holistic_scores) if holistic_scores else None,
        'quartiles': np.percentile(holistic_scores, [25, 50, 75]) if holistic_scores else None
    }
    
    if true_scores and len(true_scores) == len(holistic_scores):
        stats['qwk'] = quadratic_weighted_kappa(holistic_scores, true_scores)
    
    # Calculate trait statistics
    if multi_agent:
        trait_stats = _calculate_trait_statistics(evaluation_results)
        stats['trait_statistics'] = trait_stats
    
    logger.info("Completed analysis of evaluation results")
    return stats

def _calculate_trait_statistics(evaluation_results):
    """Calculate statistics for individual traits across evaluations"""
    # Initialize empty trait statistics
    trait_stats = {}
    
    # Collect all trait scores
    for result in evaluation_results:
        if 'final_assessment' not in result or 'trait_summary' not in result['final_assessment']:
            continue
            
        for trait_data in result['final_assessment']['trait_summary']:
            trait_name = trait_data['trait']
            if trait_name not in trait_stats:
                trait_stats[trait_name] = []
            
            trait_stats[trait_name].append(trait_data['score'])
    
    # Calculate statistics for each trait
    for trait_name, scores in trait_stats.items():
        trait_stats[trait_name] = {
            'count': len(scores),
            'mean': np.mean(scores),
            'median': np.median(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
    
    return trait_stats

def save_results(evaluation_results, stats, dataset_name, result_dir, multi_agent) :
    """Save evaluation results and statistics to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(result_dir, f"{dataset_name}_{timestamp}")
    
    # Create directory for this evaluation run
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # Save individual evaluation results
    results_path = os.path.join(result_path, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Save statistics
    stats_path = os.path.join(result_path, "statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        stats_clean = convert_ndarray(stats)
        json.dump(stats_clean, f, indent=2)
    
    # Generate summary report
    report_path = os.path.join(result_path, "summary_report.txt")
    _generate_summary_report(stats, report_path, multi_agent)
    
    logger.info(f"Saved evaluation results to {result_path}")
    return result_path

def _generate_summary_report(stats, report_path, multi_agent):
    """Generate a human-readable summary report from statistics"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("===== ESSAY EVALUATION SUMMARY REPORT =====\n\n")
        
        # Write basic statistics
        f.write(f"Total essays evaluated: {stats['count']}\n")
        f.write(f"Average holistic score: {stats['mean']:.2f}\n")
        f.write(f"Median holistic score: {stats['median']:.2f}\n")
        f.write(f"Minimum score: {stats['min']:.2f}\n")
        f.write(f"Maximum score: {stats['max']:.2f}\n")
        f.write(f"QWK score: {stats['qwk']:.2f}\n")
        
        # Write quartiles
        if stats['quartiles'] is not None:
            f.write(f"25th percentile: {stats['quartiles'][0]:.2f}\n")
            f.write(f"50th percentile: {stats['quartiles'][1]:.2f}\n")
            f.write(f"75th percentile: {stats['quartiles'][2]:.2f}\n")
        
        if multi_agent:
            # Write trait statistics
            f.write("\n===== TRAIT STATISTICS =====\n\n")
            for trait_name, trait_stats in stats['trait_statistics'].items():
                f.write(f"Trait: {trait_name}\n")
                f.write(f"  Average score: {trait_stats['mean']:.2f}\n")
                f.write(f"  Median score: {trait_stats['median']:.2f}\n")
                f.write(f"  Range: {trait_stats['min']:.2f} - {trait_stats['max']:.2f}\n\n")


def convert_ndarray(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):  # np.float64, np.int64 등
        return obj.item()
    else:
        return obj