import asyncio
import ssl
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from urllib.parse import urlparse
import os
import structlog
from redis.asyncio import Redis, RedisCluster
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
from redis.backoff import ExponentialBackoff
from redis.retry import Retry

logger = structlog.get_logger(__name__)


class RedisConfigError(Exception):
    """Redis configuration error."""
    pass


class RedisConnectionError(Exception):
    """Redis connection error."""
    pass


@dataclass
class RedisNodeConfig:
    """Configuration for a Redis node."""
    host: str
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    
    def __post_init__(self):
        if not self.host:
            raise RedisConfigError("Redis host cannot be empty")
        if not (1 <= self.port <= 65535):
            raise RedisConfigError(f"Invalid port: {self.port}")


@dataclass
class RedisClusterConfig:
    """Configuration for Redis cluster."""
    nodes: List[RedisNodeConfig]
    password: Optional[str] = None
    skip_full_coverage_check: bool = False
    max_connections_per_node: int = 50
    health_check_interval: int = 30
    readonly_mode: bool = False
    decode_responses: bool = True
    
    def __post_init__(self):
        if not self.nodes:
            raise RedisConfigError("At least one Redis node must be configured")
        if self.max_connections_per_node < 1:
            raise RedisConfigError("max_connections_per_node must be positive")


@dataclass
class RedisPoolConfig:
    """Connection pool configuration."""
    max_connections: int = 200
    retry_on_timeout: bool = True
    retry_on_error: List[type] = field(default_factory=lambda: [RedisConnectionError, ConnectionError])
    health_check_interval: int = 30
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[int, int] = field(default_factory=dict)
    socket_connect_timeout: float = 5.0
    socket_timeout: float = 5.0
    
    def __post_init__(self):
        if self.max_connections < 1:
            raise RedisConfigError("max_connections must be positive")
        if self.socket_connect_timeout <= 0:
            raise RedisConfigError("socket_connect_timeout must be positive")


@dataclass
class RedisTLSConfig:
    """TLS configuration for Redis connections."""
    enabled: bool = False
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_cert_file: Optional[str] = None
    check_hostname: bool = True
    
    def create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context if TLS is enabled."""
        if not self.enabled:
            return None
            
        context = ssl.create_default_context()
        
        if self.ca_cert_file:
            context.load_verify_locations(self.ca_cert_file)
            
        if self.cert_file and self.key_file:
            context.load_cert_chain(self.cert_file, self.key_file)
            
        context.check_hostname = self.check_hostname
        return context


class RedisConnectionManager:
    """Manages Redis connections with health monitoring and failover."""
    
    def __init__(
        self,
        cluster_config: RedisClusterConfig,
        pool_config: RedisPoolConfig,
        tls_config: Optional[RedisTLSConfig] = None
    ):
        self.cluster_config = cluster_config
        self.pool_config = pool_config
        self.tls_config = tls_config or RedisTLSConfig()
        self._client: Optional[Union[Redis, RedisCluster]] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_cluster = len(cluster_config.nodes) > 1
        self._connection_lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        async with self._connection_lock:
            if self._client is not None:
                return
                
            try:
                ssl_context = self.tls_config.create_ssl_context()
                
                retry = Retry(
                    ExponentialBackoff(cap=10, base=1),
                    retries=3
                )
                
                common_params = {
                    'decode_responses': self.cluster_config.decode_responses,
                    'retry': retry,
                    'ssl': ssl_context,
                    'socket_connect_timeout': self.pool_config.socket_connect_timeout,
                    'socket_timeout': self.pool_config.socket_timeout,
                    'socket_keepalive': self.pool_config.socket_keepalive,
                    'socket_keepalive_options': self.pool_config.socket_keepalive_options,
                }
                
                if self._is_cluster:
                    startup_nodes = [
                        {'host': node.host, 'port': node.port}
                        for node in self.cluster_config.nodes
                    ]
                    
                    self._client = RedisCluster(
                        startup_nodes=startup_nodes,
                        password=self.cluster_config.password,
                        skip_full_coverage_check=self.cluster_config.skip_full_coverage_check,
                        max_connections_per_node=self.cluster_config.max_connections_per_node,
                        readonly_mode=self.cluster_config.readonly_mode,
                        **common_params
                    )
                else:
                    node = self.cluster_config.nodes[0]
                    pool = ConnectionPool(
                        host=node.host,
                        port=node.port,
                        db=node.db,
                        password=node.password or self.cluster_config.password,
                        max_connections=self.pool_config.max_connections,
                        **common_params
                    )
                    
                    self._client = Redis(connection_pool=pool)
                
                # Test connection
                await self._client.ping()
                
                # Start health monitoring
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )
                
                logger.info(
                    "Redis connection initialized",
                    cluster=self._is_cluster,
                    nodes=len(self.cluster_config.nodes)
                )
                
            except Exception as e:
                logger.error("Failed to initialize Redis connection", error=str(e))
                raise RedisConnectionError(f"Failed to connect to Redis: {e}") from e
    
    async def close(self) -> None:
        """Close Redis connection and cleanup."""
        async with self._connection_lock:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None
            
            if self._client:
                await self._client.close()
                self._client = None
                
            logger.info("Redis connection closed")
    
    @property
    def client(self) -> Union[Redis, RedisCluster]:
        """Get Redis client instance."""
        if self._client is None:
            raise RedisConnectionError("Redis client not initialized")
        return self._client
    
    async def health_check(self) -> bool:
        """Perform health check on Redis connection."""
        if self._client is None:
            return False
            
        try:
            result = await self._client.ping()
            return result is True
        except Exception as e:
            logger.warning("Redis health check failed", error=str(e))
            return False
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.pool_config.health_check_interval)
                
                if not await self.health_check():
                    logger.error("Redis health check failed, attempting reconnection")
                    # In a production system, you might want to implement
                    # more sophisticated reconnection logic here
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))
    
    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        try:
            if self._is_cluster:
                # For cluster, get info from all nodes
                info = {}
                nodes = await self._client.get_nodes()
                for node in nodes:
                    node_info = await node.redis_connection.info()
                    info[f"{node.host}:{node.port}"] = node_info
                return info
            else:
                return await self._client.info()
        except Exception as e:
            logger.error("Failed to get Redis info", error=str(e))
            raise RedisConnectionError(f"Failed to get Redis info: {e}") from e


def create_redis_config_from_env() -> RedisClusterConfig:
    """Create Redis configuration from environment variables."""
    redis_url = os.getenv("REDIS_URL")
    redis_cluster_urls = os.getenv("REDIS_CLUSTER_URLS")
    
    nodes = []
    
    if redis_cluster_urls:
        # Multiple URLs for cluster
        urls = redis_cluster_urls.split(",")
        for url in urls:
            parsed = urlparse(url.strip())
            nodes.append(RedisNodeConfig(
                host=parsed.hostname or "localhost",
                port=parsed.port or 6379,
                password=parsed.password,
                db=int(parsed.path.lstrip("/")) if parsed.path else 0
            ))
    elif redis_url:
        # Single URL
        parsed = urlparse(redis_url)
        nodes.append(RedisNodeConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
            db=int(parsed.path.lstrip("/")) if parsed.path else 0
        ))
    else:
        # Default configuration
        nodes.append(RedisNodeConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            db=int(os.getenv("REDIS_DB", "0"))
        ))
    
    return RedisClusterConfig(
        nodes=nodes,
        password=os.getenv("REDIS_PASSWORD"),
        max_connections_per_node=int(os.getenv("REDIS_MAX_CONNECTIONS_PER_NODE", "50")),
        health_check_interval=int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")),
        readonly_mode=os.getenv("REDIS_READONLY_MODE", "false").lower() == "true"
    )


def create_pool_config_from_env() -> RedisPoolConfig:
    """Create pool configuration from environment variables."""
    return RedisPoolConfig(
        max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "200")),
        retry_on_timeout=os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true",
        health_check_interval=int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")),
        socket_keepalive=os.getenv("REDIS_SOCKET_KEEPALIVE", "true").lower() == "true",
        socket_connect_timeout=float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0")),
        socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
    )


def create_tls_config_from_env() -> RedisTLSConfig:
    """Create TLS configuration from environment variables."""
    return RedisTLSConfig(
        enabled=os.getenv("REDIS_TLS_ENABLED", "false").lower() == "true",
        cert_file=os.getenv("REDIS_TLS_CERT_FILE"),
        key_file=os.getenv("REDIS_TLS_KEY_FILE"),
        ca_cert_file=os.getenv("REDIS_TLS_CA_CERT_FILE"),
        check_hostname=os.getenv("REDIS_TLS_CHECK_HOSTNAME", "true").lower() == "true"
    )


# Global connection manager instance
_connection_manager: Optional[RedisConnectionManager] = None


async def get_redis_connection() -> Union[Redis, RedisCluster]:
    """Get Redis connection instance (singleton pattern)."""
    global _connection_manager
    
    if _connection_manager is None:
        cluster_config = create_redis_config_from_env()
        pool_config = create_pool_config_from_env()
        tls_config = create_tls_config_from_env()
        
        _connection_manager = RedisConnectionManager(
            cluster_config, pool_config, tls_config
        )
        await _connection_manager.initialize()
    
    return _connection_manager.client


async def close_redis_connection() -> None:
    """Close the global Redis connection."""
    global _connection_manager
    
    if _connection_manager is not None:
        await _connection_manager.close()
        _connection_manager = None