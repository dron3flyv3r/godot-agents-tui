try:
    import stable_baselines3
    import ray
    STABLE_BASELINES3_AVAILABLE = True
    RAY_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    RAY_AVAILABLE = False
    
if __name__ == "__main__":
    if STABLE_BASELINES3_AVAILABLE and not RAY_AVAILABLE:
        exit(2)
    elif not STABLE_BASELINES3_AVAILABLE and RAY_AVAILABLE:
        exit(3)
    elif not STABLE_BASELINES3_AVAILABLE and not RAY_AVAILABLE:
        exit(4)
    else:
        exit(0)