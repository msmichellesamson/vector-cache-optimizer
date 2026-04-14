from typing import Dict, List, Optional, Set
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from .cache_warmer import CacheWarmer
from ..monitoring.hit_rate_tracker import HitRateTracker


@dataclass
class WarmupJob:
    """Represents a cache warmup job."""
    key_pattern: str
    priority: int
    created_at: datetime
    last_run: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


class AdaptiveWarmupScheduler:
    """Intelligent scheduler for cache warmup operations."""
    
    def __init__(
        self, 
        cache_warmer: CacheWarmer,
        hit_rate_tracker: HitRateTracker,
        min_interval: int = 300,  # 5 minutes
        max_jobs: int = 50
    ):
        self.cache_warmer = cache_warmer
        self.hit_rate_tracker = hit_rate_tracker
        self.min_interval = min_interval
        self.max_jobs = max_jobs
        self.jobs: Dict[str, WarmupJob] = {}
        self.running_jobs: Set[str] = set()
        self.logger = logging.getLogger(__name__)
    
    async def schedule_warmup(self, key_pattern: str, priority: int = 5) -> bool:
        """Schedule a new warmup job."""
        if len(self.jobs) >= self.max_jobs:
            self.logger.warning(f"Max jobs limit reached: {self.max_jobs}")
            return False
        
        if key_pattern in self.jobs:
            self.jobs[key_pattern].priority = max(
                self.jobs[key_pattern].priority, priority
            )
            return True
        
        self.jobs[key_pattern] = WarmupJob(
            key_pattern=key_pattern,
            priority=priority,
            created_at=datetime.utcnow()
        )
        self.logger.info(f"Scheduled warmup job: {key_pattern}")
        return True
    
    async def run_scheduler(self) -> None:
        """Run the adaptive scheduler loop."""
        while True:
            try:
                await self._process_jobs()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(30)
    
    async def _process_jobs(self) -> None:
        """Process pending warmup jobs based on priority and hit rates."""
        eligible_jobs = self._get_eligible_jobs()
        
        for job in eligible_jobs[:3]:  # Process max 3 concurrent jobs
            if job.key_pattern in self.running_jobs:
                continue
            
            self.running_jobs.add(job.key_pattern)
            asyncio.create_task(self._execute_job(job))
    
    def _get_eligible_jobs(self) -> List[WarmupJob]:
        """Get jobs eligible for execution based on priority and timing."""
        now = datetime.utcnow()
        eligible = []
        
        for job in self.jobs.values():
            if job.last_run and (now - job.last_run).seconds < self.min_interval:
                continue
            
            # Higher priority for low hit rate patterns
            hit_rate = self.hit_rate_tracker.get_pattern_hit_rate(job.key_pattern)
            if hit_rate < 0.3 or job.priority >= 8:  # High priority or low hit rate
                eligible.append(job)
        
        return sorted(eligible, key=lambda x: (-x.priority, x.created_at))
    
    async def _execute_job(self, job: WarmupJob) -> None:
        """Execute a warmup job."""
        try:
            await self.cache_warmer.warm_pattern(job.key_pattern)
            job.success_count += 1
            job.last_run = datetime.utcnow()
            self.logger.info(f"Warmup job completed: {job.key_pattern}")
        except Exception as e:
            job.failure_count += 1
            self.logger.error(f"Warmup job failed: {job.key_pattern} - {e}")
        finally:
            self.running_jobs.discard(job.key_pattern)
    
    def get_job_stats(self) -> Dict[str, Dict]:
        """Get statistics for all jobs."""
        return {
            pattern: {
                "priority": job.priority,
                "success_count": job.success_count,
                "failure_count": job.failure_count,
                "last_run": job.last_run.isoformat() if job.last_run else None
            }
            for pattern, job in self.jobs.items()
        }
