"""Git versioning utility for compute functions."""
from pathlib import Path
from typing import Dict, Optional

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


def get_git_info(repo_path: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """
    Get git information from the repository.
    
    Args:
        repo_path: Path to repository root. If None, searches from current file.
    
    Returns:
        Dictionary with git information:
        - commit_hash: Short commit hash (7 characters)
        - commit_hash_full: Full commit hash
        - branch: Current branch name
        - is_dirty: Boolean indicating if working directory has uncommitted changes
    """
    if not GIT_AVAILABLE:
        return {
            "commit_hash": None,
            "commit_hash_full": None,
            "branch": None,
            "is_dirty": None,
        }
    
    if repo_path is None:
        # Start from the functions directory and search up for .git
        repo_path = Path(__file__).parent.parent
    
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
        head = repo.head
        
        # Get commit hash
        commit_hash_full = head.commit.hexsha
        commit_hash = commit_hash_full[:7]
        
        # Get branch name
        try:
            branch = head.ref.name
        except AttributeError:
            # Detached HEAD state
            branch = None
        
        # Check if working directory is dirty
        is_dirty = repo.is_dirty()
        
        return {
            "commit_hash": commit_hash,
            "commit_hash_full": commit_hash_full,
            "branch": branch,
            "is_dirty": is_dirty,
        }
    except (git.exc.InvalidGitRepositoryError, git.exc.GitCommandError, Exception):
        # Not a git repository or git command failed
        return {
            "commit_hash": None,
            "commit_hash_full": None,
            "branch": None,
            "is_dirty": None,
        }


def get_versioning_info(function_version: str) -> Dict[str, Optional[str]]:
    """
    Get complete versioning information including function version and git info.
    
    Args:
        function_version: The __version__ string from the function module.
    
    Returns:
        Dictionary with versioning information:
        - function_version: The function's version string
        - commit_hash: Short commit hash (7 characters)
        - commit_hash_full: Full commit hash
        - branch: Current branch name
        - is_dirty: Boolean indicating if working directory has uncommitted changes
    """
    git_info = get_git_info()
    
    return {
        "function_version": function_version,
        "commit_hash": git_info.get("commit_hash"),
        "commit_hash_full": git_info.get("commit_hash_full"),
        "branch": git_info.get("branch"),
        "is_dirty": git_info.get("is_dirty"),
    }

