import os

def get_clean_env():
    """Return a copy of os.environ with large variables removed to avoid E2BIG errors.
    
    This is critical for Celery workers which may have huge configuration or state 
    in environment variables, causing subprocess.run() to fail with OSError: [Errno 7] Argument list too long.
    """
    env = os.environ.copy()
    # 10KB limit per variable should be safe enough for normal vars while catching huge config dumps
    limit = 10000 
    for k, v in list(env.items()):
        if len(v) > limit:
            # Keep critical paths even if huge (unlikely but safe)
            if k in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH"):
                 continue
            del env[k]
    return env
