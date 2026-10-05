class Config:
    # Reproducibility
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_JOBS: int = -1
    TARGET: str = 'Transported'
    