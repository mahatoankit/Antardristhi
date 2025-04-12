import subprocess
import os
from pathlib import Path
from .safe_packages import SAFE_PACKAGES, TEMP_CACHE_DIR

class PackageInstaller:
    def __init__(self):
        self.cache_dir = Path(TEMP_CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        
    async def install_package(self, package_name: str) -> bool:
        if package_name not in SAFE_PACKAGES:
            raise ValueError(f"Package {package_name} is not in the safe list")
            
        version = SAFE_PACKAGES[package_name]
        cache_path = self.cache_dir / f"{package_name}-{version}"
        
        if cache_path.exists():
            return True
            
        try:
            print(f"Installing {package_name} {version}...")
            subprocess.check_call([
                "pip", "install",
                f"{package_name}=={version}",
                "--target", str(cache_path)
            ])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}: {e}")
            return False