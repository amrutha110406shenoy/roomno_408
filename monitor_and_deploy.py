"""
Monitor training and auto-deploy when complete
"""
import time
from pathlib import Path
import subprocess
import sys

def check_model_exists():
    """Check if fine-tuned model exists"""
    model_path = Path("drone_finetuned_models/yolov8n_drone_fast/weights/best.pt")
    return model_path.exists()

def wait_for_training(max_wait=1800):
    """Wait for training to complete"""
    print("\n[+] Waiting for fine-tuning to complete...")
    print("[+] Max wait time: 30 minutes")

    start_time = time.time()
    check_interval = 10

    while time.time() - start_time < max_wait:
        if check_model_exists():
            print("\n[OK] ✓ Model training COMPLETE!")
            return True

        elapsed = time.time() - start_time
        print(f"  Still training... ({elapsed/60:.1f} min elapsed)", end='\r')
        time.sleep(check_interval)

    print("\n[!] Timeout: Training took too long")
    return False

def deploy():
    """Run deployment"""
    print("\n[STEP] Deploying fine-tuned model...")

    try:
        result = subprocess.run(
            [sys.executable, "deploy_finetuned_drone_detector.py"],
            capture_output=False,
            timeout=1200
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[!] Deployment error: {e}")
        return False

def main():
    print("="*80)
    print("AUTO-MONITOR & DEPLOY")
    print("="*80)

    # Wait for training
    if wait_for_training():
        # Deploy
        if deploy():
            print("\n" + "="*80)
            print("SUCCESS ✓ FINE-TUNING AND DEPLOYMENT COMPLETE")
            print("="*80)
            print("\n[OK] Videos ready in: drone_production_outputs/")
        else:
            print("\n[!] Deployment failed")
    else:
        print("\n[!] Training did not complete")

if __name__ == "__main__":
    main()
