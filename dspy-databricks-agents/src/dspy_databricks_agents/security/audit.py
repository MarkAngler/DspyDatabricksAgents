"""Audit logging for security events."""

import time
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import threading
from collections import deque

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFY = "data_modify"
    CONFIG_CHANGE = "config_change"
    SECURITY_ALERT = "security_alert"


@dataclass
class AuditEvent:
    """An audit event."""
    event_type: AuditEventType
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    username: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    ip_address: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["timestamp_iso"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return json.dumps(data, default=str)


class AuditLogger:
    """Thread-safe audit logger."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        max_events: int = 10000,
        retention_days: int = 90
    ):
        self.log_file = log_file
        self.max_events = max_events
        self.retention_days = retention_days
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.RLock()
    
    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        with self._lock:
            self._events.append(event)
            
            # Log to file if configured
            if self.log_file:
                self._write_to_file(event)
            
            # Log to standard logger
            logger.info(f"AUDIT: {event.event_type.value} - {event.to_json()}")
    
    def log_auth_success(
        self,
        user_id: str,
        username: str,
        ip_address: Optional[str] = None
    ) -> None:
        """Log successful authentication."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            result="success"
        )
        self.log_event(event)
    
    def log_auth_failure(
        self,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        reason: Optional[str] = None
    ) -> None:
        """Log failed authentication."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_FAILURE,
            username=username,
            ip_address=ip_address,
            result="failure",
            metadata={"reason": reason} if reason else {}
        )
        self.log_event(event)
    
    def log_permission_check(
        self,
        user_id: str,
        username: str,
        resource: str,
        action: str,
        granted: bool
    ) -> None:
        """Log permission check."""
        event = AuditEvent(
            event_type=AuditEventType.PERMISSION_GRANTED if granted else AuditEventType.PERMISSION_DENIED,
            user_id=user_id,
            username=username,
            resource=resource,
            action=action,
            result="granted" if granted else "denied"
        )
        self.log_event(event)
    
    def log_data_access(
        self,
        user_id: str,
        username: str,
        resource: str,
        action: str = "read",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log data access."""
        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=user_id,
            username=username,
            resource=resource,
            action=action,
            metadata=metadata or {}
        )
        self.log_event(event)
    
    def log_security_alert(
        self,
        alert_type: str,
        description: str,
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security alert."""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT,
            action=alert_type,
            metadata={
                "description": description,
                "severity": severity,
                **(metadata or {})
            }
        )
        self.log_event(event)
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events."""
        with self._lock:
            events = list(self._events)
        
        # Filter events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # Sort by timestamp descending and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def _write_to_file(self, event: AuditEvent) -> None:
        """Write event to audit log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(event.to_json() + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def cleanup_old_events(self) -> int:
        """Remove events older than retention period."""
        cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
        
        with self._lock:
            old_count = len(self._events)
            self._events = deque(
                (e for e in self._events if e.timestamp > cutoff_time),
                maxlen=self.max_events
            )
            removed = old_count - len(self._events)
        
        logger.info(f"Cleaned up {removed} old audit events")
        return removed