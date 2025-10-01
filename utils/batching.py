#!/usr/bin/env python3
"""
Batching Utilities for LLM Call Optimization

Provides generic batching logic for reducing LLM API calls through
intelligent grouping and parallel processing.
"""

from typing import List, TypeVar, Callable, Optional, Any
from dataclasses import dataclass
import asyncio
import logging
import time
import math

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BatchResult:
    """Result of processing a single batch."""
    success: bool
    data: Any
    error: Optional[str] = None
    batch_index: int = 0
    item_indices: List[int] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.item_indices is None:
            self.item_indices = []


@dataclass
class BatchMetrics:
    """Metrics for batch processing operation."""
    total_items: int = 0
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    total_llm_calls: int = 0
    saved_llm_calls: int = 0
    total_time: float = 0.0
    average_batch_time: float = 0.0
    fallback_count: int = 0
    partial_success_count: int = 0
    
    def __str__(self):
        return f"""BatchMetrics:
  Total Items: {self.total_items}
  Total Batches: {self.total_batches}
  Successful: {self.successful_batches}
  Failed: {self.failed_batches}
  LLM Calls: {self.total_llm_calls}
  Saved Calls: {self.saved_llm_calls} ({self.savings_percent:.1f}%)
  Total Time: {self.total_time:.2f}s
  Avg Batch Time: {self.average_batch_time:.2f}s
  Fallbacks: {self.fallback_count}"""
    
    @property
    def savings_percent(self) -> float:
        if self.total_items == 0:
            return 0.0
        return (self.saved_llm_calls / self.total_items) * 100


# ============================================================================
# CORE BATCHING FUNCTIONS
# ============================================================================

def chunk_items(items: List[T], batch_size: int) -> List[List[T]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        batch_size: Maximum items per chunk
        
    Returns:
        List of chunks
        
    Example:
        >>> chunk_items([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    chunks = []
    for i in range(0, len(items), batch_size):
        chunks.append(items[i:i + batch_size])
    
    return chunks


async def batch_process(
    items: List[T],
    process_fn: Callable[[List[T]], Any],
    batch_size: int,
    max_parallel: int = 3,
    enable_fallback: bool = True
) -> tuple[List[R], BatchMetrics]:
    """
    Process items in batches with parallel execution.
    
    This is the main batching function that:
    1. Splits items into batches
    2. Processes batches in parallel (up to max_parallel)
    3. Falls back to individual processing on failure
    4. Tracks metrics
    
    Args:
        items: Items to process
        process_fn: Async function that processes a batch
        batch_size: Items per batch
        max_parallel: Max batches to process simultaneously
        enable_fallback: If True, retry failed batches individually
        
    Returns:
        Tuple of (results, metrics)
        
    Example:
        >>> results, metrics = await batch_process(
        ...     items=[pc1, pc2, pc3],
        ...     process_fn=translate_batch,
        ...     batch_size=2
        ... )
    """
    if not items:
        return [], BatchMetrics()
    
    start_time = time.time()
    metrics = BatchMetrics(
        total_items=len(items),
        total_batches=math.ceil(len(items) / batch_size)
    )
    
    logger.info("=" * 70)
    logger.info("BATCH PROCESSING START")
    logger.info("=" * 70)
    logger.info(f"Total items: {len(items)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Expected batches: {metrics.total_batches}")
    logger.info(f"Max parallel: {max_parallel}")
    
    # Split into batches
    batches = chunk_items(items, batch_size)
    
    # Process batches with limited parallelism
    all_results = []
    
    for batch_group_start in range(0, len(batches), max_parallel):
        batch_group = batches[batch_group_start:batch_group_start + max_parallel]
        
        logger.info(f"\n📦 Processing batch group {batch_group_start // max_parallel + 1}")
        logger.info(f"   Batches in group: {len(batch_group)}")
        
        # Process this group of batches in parallel
        tasks = [
            _process_single_batch(
                batch=batch,
                batch_index=batch_group_start + i,
                process_fn=process_fn,
                enable_fallback=enable_fallback
            )
            for i, batch in enumerate(batch_group)
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract results and update metrics
        for batch_idx, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                logger.error(f"❌ Batch {batch_group_start + batch_idx} raised exception: {batch_result}")
                metrics.failed_batches += 1
            elif isinstance(batch_result, BatchResult):
                if batch_result.success:
                    all_results.extend(batch_result.data)
                    metrics.successful_batches += 1
                    metrics.total_llm_calls += 1
                else:
                    metrics.failed_batches += 1
                    if enable_fallback:
                        metrics.fallback_count += len(batch_result.item_indices)
            else:
                logger.warning(f"⚠️ Unexpected batch result type: {type(batch_result)}")
    
    # Calculate final metrics
    metrics.total_time = time.time() - start_time
    metrics.average_batch_time = metrics.total_time / max(metrics.total_batches, 1)
    metrics.saved_llm_calls = metrics.total_items - metrics.total_llm_calls - metrics.fallback_count
    
    logger.info("\n" + "=" * 70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(str(metrics))
    logger.info("=" * 70 + "\n")
    
    return all_results, metrics


async def _process_single_batch(
    batch: List[T],
    batch_index: int,
    process_fn: Callable,
    enable_fallback: bool
) -> BatchResult:
    """
    Process a single batch with error handling.
    
    Args:
        batch: Items in this batch
        batch_index: Index of this batch
        process_fn: Function to process the batch
        enable_fallback: Whether to retry individually on failure
        
    Returns:
        BatchResult with success status and data
    """
    start_time = time.time()
    item_indices = list(range(len(batch)))
    
    logger.info(f"📤 Sending batch {batch_index} with {len(batch)} items")
    
    try:
        # Try batch processing
        result_data = await process_fn(batch)
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Batch {batch_index} completed in {processing_time:.2f}s")
        
        return BatchResult(
            success=True,
            data=result_data,
            batch_index=batch_index,
            item_indices=item_indices,
            processing_time=processing_time
        )
    
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Batch {batch_index} failed after {processing_time:.2f}s: {e}")
        
        # Try fallback if enabled
        if enable_fallback:
            logger.warning(f"⚠️ Falling back to individual processing for batch {batch_index}")
            fallback_results = await _fallback_individual_processing(batch, process_fn)
            
            return BatchResult(
                success=True,
                data=fallback_results,
                error=f"Batch failed, used fallback: {str(e)}",
                batch_index=batch_index,
                item_indices=item_indices,
                processing_time=processing_time
            )
        else:
            return BatchResult(
                success=False,
                data=[],
                error=str(e),
                batch_index=batch_index,
                item_indices=item_indices,
                processing_time=processing_time
            )


async def _fallback_individual_processing(
    items: List[T],
    process_fn: Callable
) -> List[R]:
    """
    Fall back to processing items individually.
    
    This is called when batch processing fails.
    
    Args:
        items: Items that failed in batch
        process_fn: Function to process individual items
        
    Returns:
        List of results (one per item)
    """
    logger.info(f"   🔄 Processing {len(items)} items individually")
    
    results = []
    for i, item in enumerate(items):
        try:
            # Process single item (wrap in list for process_fn)
            result = await process_fn([item])
            if isinstance(result, list) and len(result) > 0:
                results.append(result[0])
            else:
                results.append(result)
        except Exception as e:
            logger.error(f"   ❌ Individual item {i} failed: {e}")
            results.append(None)  # Placeholder for failed item
    
    logger.info(f"   ✅ Individual processing complete: {len([r for r in results if r])} succeeded")
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Batch processing numbers
    async def example_batch_processor(numbers: List[int]) -> List[int]:
        """Example: Square all numbers in batch."""
        await asyncio.sleep(0.1)  # Simulate API call
        return [n ** 2 for n in numbers]
    
    async def demo():
        numbers = list(range(1, 26))  # 1-25
        
        results, metrics = await batch_process(
            items=numbers,
            process_fn=example_batch_processor,
            batch_size=10,
            max_parallel=3
        )
        
        print(f"\nResults: {results}")
        print(f"\nMetrics:\n{metrics}")
    
    asyncio.run(demo())